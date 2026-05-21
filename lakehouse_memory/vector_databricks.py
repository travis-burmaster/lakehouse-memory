"""Real VectorIndex backed by Databricks Vector Search Delta Sync indexes.

`DatabricksVectorIndex` implements the `VectorIndex` protocol with `upsert`
and `delete` as no-ops (Delta Sync auto-syncs the source Delta table to the
managed index). `search` queries the index via the Vector Search SDK.

`ensure_indexes` is an idempotent factory that provisions the Vector Search
endpoint and the two Delta Sync indexes (episodic, semantic). Working
memory has no index.

Reads are eventually consistent: sync delay is typically seconds to minutes.
"""

from __future__ import annotations

import time
from typing import Any

from databricks.vector_search.client import VectorSearchClient

from lakehouse_memory.config import MemoryConfig

_ENDPOINT_POLL_INTERVAL_S = 10
_ENDPOINT_POLL_TIMEOUT_S = 600  # 10 min — endpoint cold start can be ~5 min
_INDEX_POLL_INTERVAL_S = 5
_INDEX_POLL_TIMEOUT_S = 1200  # 20 min — first sync for a fresh index (can be slow on cold endpoints)


class DatabricksVectorIndex:
    """`VectorIndex` Protocol implementation backed by a Delta Sync index."""

    # Default columns to return when none are specified explicitly.
    # In practice callers should always pass the correct columns for their store
    # (episodic: event_id; semantic: fact_id). The default covers only the
    # embedding column so a misconfigured instance fails visibly rather than
    # silently fetching cross-store columns that don't exist in the index.
    _DEFAULT_COLUMNS: list[str] = ["text"]

    def __init__(
        self,
        endpoint_name: str,
        index_name: str,
        embedding_column: str = "text",
        workspace_url: str | None = None,
        access_token: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self._endpoint_name = endpoint_name
        self._index_name = index_name
        self._embedding_column = embedding_column
        self._workspace_url = workspace_url
        self._access_token = access_token
        self._columns = columns if columns is not None else list(self._DEFAULT_COLUMNS)

    def upsert(self, records: list[dict[str, Any]]) -> None:
        # No-op: Delta Sync handles index population from the Delta source.
        return None

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        client = VectorSearchClient(
            workspace_url=self._workspace_url,
            personal_access_token=self._access_token,
        )
        index = client.get_index(
            endpoint_name=self._endpoint_name,
            index_name=self._index_name,
        )
        kwargs: dict[str, Any] = {
            "columns": self._columns,
            "query_text": query,
            "num_results": k,
        }
        if filter:
            kwargs["filters"] = filter
        response = index.similarity_search(**kwargs)
        return _unpack_search_response(response)

    def delete(self, ids: list[str]) -> None:
        # No-op: deletes propagate via Delta source.
        return None

    def trigger_sync(self) -> None:
        """Trigger an on-demand sync for a TRIGGERED Delta Sync index.

        Calls the Vector Search REST API ``POST .../sync`` endpoint so that
        data written to the source Delta table is picked up by the index.
        Only necessary for ``pipeline_type="TRIGGERED"`` indexes; CONTINUOUS
        indexes sync automatically.
        """
        vs_client = VectorSearchClient(
            workspace_url=self._workspace_url,
            personal_access_token=self._access_token,
        )
        index_obj = vs_client.get_index(
            endpoint_name=self._endpoint_name,
            index_name=self._index_name,
        )
        index_obj.sync()


def _unpack_search_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the Vector Search SDK response into a list of plain dicts.

    Response shape (per Databricks Vector Search REST API):
        {
            "result": {"row_count": N, "data_array": [[...], ...]},
            "manifest": {"columns": [{"name": "id"}, {"name": "text"}, ...]},
        }
    """
    result = response.get("result", {})
    manifest = response.get("manifest", {})
    columns = [c.get("name", "") for c in manifest.get("columns", [])]
    rows = result.get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


def ensure_indexes(
    workspace_url: str,
    access_token: str,
    endpoint_name: str,
    config: MemoryConfig,
) -> dict[str, DatabricksVectorIndex]:
    """Idempotently ensure the Vector Search endpoint and Delta Sync indexes exist.

    Returns {"episodic": ..., "semantic": ...}. WorkingStore has no index.

    Steps:
        1. If endpoint doesn't exist, create it and poll until ONLINE.
        2. For each of (episodic, semantic), try to create the Delta Sync
           index; if it already exists, swallow and proceed.
        3. Poll each index until READY.
    """
    client = VectorSearchClient(
        workspace_url=workspace_url,
        personal_access_token=access_token,
    )

    _ensure_endpoint(client, endpoint_name)

    # Columns to retrieve per store type (must match source Delta table columns)
    _STORE_COLUMNS: dict[str, list[str]] = {
        "episodic": ["event_id", "text", "user_id", "session_id", "agent_id"],
        "semantic": ["fact_id", "text", "user_id", "session_id", "agent_id"],
    }

    indexes: dict[str, DatabricksVectorIndex] = {}
    for table in ("episodic", "semantic"):
        index_name = f"{config.catalog}.{config.schema_name}.{table}_idx"
        source_table = config.fqn(table)
        _try_create_delta_sync_index(
            client=client,
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table=source_table,
            embedding_endpoint=config.embedding.endpoint_name,
        )
        _wait_for_index_ready(client, endpoint_name, index_name)
        indexes[table] = DatabricksVectorIndex(
            endpoint_name=endpoint_name,
            index_name=index_name,
            workspace_url=workspace_url,
            access_token=access_token,
            columns=_STORE_COLUMNS[table],
        )
    return indexes


def _ensure_endpoint(client: Any, endpoint_name: str) -> None:
    existing = client.list_endpoints().get("endpoints", [])
    by_name = {e.get("name"): e for e in existing}
    if endpoint_name not in by_name:
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    else:
        # If already listed and ONLINE, no need to poll.
        existing_state = by_name[endpoint_name].get("endpoint_status", {}).get("state")
        if existing_state == "ONLINE":
            return

    deadline = time.time() + _ENDPOINT_POLL_TIMEOUT_S
    while time.time() < deadline:
        info = client.get_endpoint(name=endpoint_name)
        state = info.get("endpoint_status", {}).get("state")
        if state == "ONLINE":
            return
        time.sleep(_ENDPOINT_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Vector Search endpoint {endpoint_name!r} did not become ONLINE within "
        f"{_ENDPOINT_POLL_TIMEOUT_S}s"
    )


def _try_create_delta_sync_index(
    client: Any,
    endpoint_name: str,
    index_name: str,
    source_table: str,
    embedding_endpoint: str,
    retries: int = 3,
    retry_delay_s: float = 15.0,
) -> None:
    """Create a Delta Sync index, retrying transient server errors.

    Swallows "already exists" errors (idempotency). Retries server-side
    failures (e.g. transient Model Serving endpoint errors) up to `retries`
    times with a fixed delay.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            client.create_delta_sync_index(
                endpoint_name=endpoint_name,
                index_name=index_name,
                source_table_name=source_table,
                pipeline_type="TRIGGERED",
                primary_key=_primary_key_for(source_table),
                embedding_source_column="text",
                embedding_model_endpoint_name=embedding_endpoint,
            )
            return
        except Exception as e:
            message = str(e).upper()
            # Handle both "ALREADY_EXISTS" error codes and natural-language messages
            # like "UC entity ... already exists."
            if (
                "ALREADY_EXISTS" in message
                or "RESOURCE_ALREADY_EXISTS" in message
                or "ALREADY EXISTS" in message
            ):
                return
            last_exc = e
            if attempt < retries:
                time.sleep(retry_delay_s)
    assert last_exc is not None
    raise last_exc


def _primary_key_for(source_table: str) -> str:
    """Pick the primary key column based on the table type embedded in the FQN."""
    table = source_table.rsplit(".", 1)[-1]
    if table == "episodic":
        return "event_id"
    if table == "semantic":
        return "fact_id"
    raise ValueError(f"No primary key configured for source table {source_table!r}")


def _wait_for_index_ready(client: Any, endpoint_name: str, index_name: str) -> None:
    """Poll until the index is ONLINE with no pending update.

    Waits for `detailed_state == "ONLINE_NO_PENDING_UPDATE"` (not just
    `ready=True`) so that the underlying DLT pipeline is fully IDLE before
    the next index creation starts. This avoids QUOTA_EXCEEDED errors on
    workspaces that limit concurrent BRICKINDEX pipelines to 1.

    The SDK's `get_index` returns a VectorSearchIndex object; use `.describe()`
    to get the status dict (shape: {"status": {"ready": bool, "detailed_state": str, ...}, ...}).
    """
    _TERMINAL_FAILED_STATES = {"OFFLINE_FAILED", "OFFLINE", "OFFLINE_DEGRADED"}
    deadline = time.time() + _INDEX_POLL_TIMEOUT_S
    while time.time() < deadline:
        index_obj = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        # index_obj is a VectorSearchIndex; .describe() returns a plain dict.
        info = index_obj.describe() if hasattr(index_obj, "describe") else index_obj
        if isinstance(info, dict):
            status = info.get("status", {})
            detailed_state = status.get("detailed_state", "")
        else:
            status_obj = getattr(info, "status", None)
            detailed_state = getattr(status_obj, "detailed_state", "") or ""
        if detailed_state == "ONLINE_NO_PENDING_UPDATE":
            return
        if detailed_state in _TERMINAL_FAILED_STATES:
            raise RuntimeError(
                f"Vector Search index {index_name!r} entered terminal state "
                f"{detailed_state!r}. Check the DLT pipeline in the workspace UI."
            )
        time.sleep(_INDEX_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Vector Search index {index_name!r} did not reach ONLINE_NO_PENDING_UPDATE "
        f"within {_INDEX_POLL_TIMEOUT_S}s"
    )
