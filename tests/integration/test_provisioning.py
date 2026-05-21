"""Integration tests: provisioning is idempotent and creates real resources."""
from __future__ import annotations

import pytest

from databricks.vector_search.client import VectorSearchClient


@pytest.mark.usefixtures("live_memory")
def test_schema_and_tables_exist(
    live_memory, sql_client, test_catalog, ephemeral_schema_name
) -> None:
    rows = sql_client.execute(
        f"SHOW TABLES IN {test_catalog}.{ephemeral_schema_name}"
    )
    table_names = {row.get("tableName") or row.get("table_name") for row in rows}
    assert {"episodic", "semantic", "working"}.issubset(table_names), (
        f"Expected episodic/semantic/working in {table_names}"
    )


def test_vector_search_indexes_exist(
    live_memory,
    workspace_url,
    access_token,
    vector_search_endpoint,
    test_catalog,
    ephemeral_schema_name,
) -> None:
    client = VectorSearchClient(
        workspace_url=workspace_url,
        personal_access_token=access_token,
    )
    for table in ("episodic", "semantic"):
        index_name = f"{test_catalog}.{ephemeral_schema_name}.{table}_idx"
        index_obj = client.get_index(endpoint_name=vector_search_endpoint, index_name=index_name)
        # get_index returns a VectorSearchIndex; use .describe() for the status dict.
        # Fall back to dict .get() if the SDK ever returns a plain dict.
        if hasattr(index_obj, "describe"):
            info = index_obj.describe()
        else:
            info = index_obj
        ready = (
            info.get("status", {}).get("ready")
            if isinstance(info, dict)
            else getattr(getattr(info, "status", None), "ready", None)
        )
        assert ready, f"Index {index_name} is not ready: {info}"


def test_provision_is_idempotent(
    live_memory,
    workspace_url,
    access_token,
    vector_search_endpoint,
) -> None:
    """Calling provision() again should not raise."""
    live_memory.provision(
        vector_search_endpoint=vector_search_endpoint,
        workspace_url=workspace_url,
        access_token=access_token,
    )
