"""Integration test fixtures.

These tests are skipped unless LAKEHOUSE_MEMORY_INTEGRATION=1 is set.
See SETUP.md for the required environment variables.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LAKEHOUSE_MEMORY_INTEGRATION"):
    pytest.skip(
        "Set LAKEHOUSE_MEMORY_INTEGRATION=1 to run integration tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def workspace_url() -> str:
    return os.environ["DATABRICKS_HOST"]


@pytest.fixture(scope="session")
def access_token() -> str:
    return os.environ["DATABRICKS_TOKEN"]


@pytest.fixture(scope="session")
def http_path() -> str:
    return os.environ["DATABRICKS_HTTP_PATH"]


@pytest.fixture(scope="session")
def vector_search_endpoint() -> str:
    return os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT"]


@pytest.fixture(scope="session")
def test_catalog() -> str:
    return os.environ["LAKEHOUSE_MEMORY_TEST_CATALOG"]


@pytest.fixture(scope="session")
def ephemeral_schema_name() -> str:
    prefix = os.environ.get("LAKEHOUSE_MEMORY_TEST_SCHEMA", "lakehouse_memory_test")
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def sql_client(workspace_url: str, access_token: str, http_path: str):
    from lakehouse_memory.client import SqlConnectorClient

    hostname = workspace_url.replace("https://", "").replace("http://", "").rstrip("/")
    return SqlConnectorClient(
        server_hostname=hostname,
        http_path=http_path,
        access_token=access_token,
    )


@pytest.fixture(scope="session")
def live_memory(
    workspace_url: str,
    access_token: str,
    http_path: str,
    vector_search_endpoint: str,
    test_catalog: str,
    ephemeral_schema_name: str,
) -> Iterator:
    """A Memory provisioned against the workspace with a temporary schema.

    On teardown: DROP SCHEMA CASCADE (drops the indexes inside it too).
    The Vector Search endpoint is NOT deleted.
    """
    from lakehouse_memory import Memory
    from lakehouse_memory.client import SqlConnectorClient

    mem = Memory.from_databricks(
        catalog=test_catalog,
        schema_name=ephemeral_schema_name,
        workspace_url=workspace_url,
        access_token=access_token,
        http_path=http_path,
        vector_search_endpoint=vector_search_endpoint,
    )
    mem.provision()  # uses stashed creds

    yield mem

    hostname = workspace_url.replace("https://", "").replace("http://", "").rstrip("/")
    teardown_client = SqlConnectorClient(
        server_hostname=hostname, http_path=http_path, access_token=access_token
    )
    try:
        teardown_client.execute(
            f"DROP SCHEMA IF EXISTS {test_catalog}.{ephemeral_schema_name} CASCADE"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Teardown DROP SCHEMA failed: {e}", flush=True)


def wait_for_searchable(store, query: str, expected_id: str, timeout_s: int = 360) -> None:
    """Poll store.search until expected_id appears in results or timeout.

    Checks both the generic ``id`` key and store-specific primary-key aliases
    (``event_id`` for episodic, ``fact_id`` for semantic) so this helper works
    for Delta Sync indexes whose column names come from the source Delta table.

    For TRIGGERED Delta Sync indexes, also calls ``trigger_sync()`` on the
    store's index each poll cycle so that new rows are picked up promptly.
    """
    _PK_ALIASES = ("id", "event_id", "fact_id")
    # Access underlying index to fire on-demand sync (TRIGGERED pipeline)
    _index = getattr(store, "_index", None)
    _trigger_sync = getattr(_index, "trigger_sync", None)

    # SemanticStore exposes retrieve(); EpisodicStore exposes search().
    _search_fn = getattr(store, "search", None) or getattr(store, "retrieve", None)
    if _search_fn is None:
        raise AttributeError(f"{type(store).__name__} has neither search() nor retrieve()")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _trigger_sync is not None:
            try:
                _trigger_sync()
            except Exception:
                pass  # best-effort; sync may already be in-progress
        results = _search_fn(query, k=10)
        if any(r.get(alias) == expected_id for r in results for alias in _PK_ALIASES):
            return
        time.sleep(5)
    pytest.fail(f"id {expected_id!r} not searchable after {timeout_s}s")
