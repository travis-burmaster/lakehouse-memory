"""Unit tests for the Databricks-backed VectorIndex implementation.

The Databricks Vector Search SDK (`databricks-vectorsearch`) is patched so
no network calls are made. We verify the wiring, not the SDK.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lakehouse_memory.config import MemoryConfig
from lakehouse_memory.vector_databricks import (
    DatabricksVectorIndex,
    ensure_indexes,
)


def test_databricks_vector_index_upsert_is_noop() -> None:
    """upsert is a no-op; Delta Sync handles index population from the Delta source."""
    idx = DatabricksVectorIndex(endpoint_name="ep", index_name="cat.sch.episodic_idx")
    idx.upsert([{"id": "1", "text": "anything", "user_id": "u_1"}])
    # No exception, no side effect — test passes by not raising.


def test_databricks_vector_index_delete_is_noop() -> None:
    """delete is a no-op; deletes propagate via Delta."""
    idx = DatabricksVectorIndex(endpoint_name="ep", index_name="cat.sch.episodic_idx")
    idx.delete(["any-id"])


def test_databricks_vector_index_search_queries_sdk_and_unpacks_results() -> None:
    """search() should call the SDK index's similarity_search and translate the response."""
    fake_index_obj = MagicMock()
    fake_index_obj.similarity_search.return_value = {
        "result": {
            "row_count": 2,
            "data_array": [
                ["id-1", "first text", 0.92],
                ["id-2", "second text", 0.84],
            ],
        },
        "manifest": {
            "columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}],
        },
    }
    fake_client = MagicMock()
    fake_client.get_index.return_value = fake_index_obj

    with patch(
        "lakehouse_memory.vector_databricks.VectorSearchClient",
        return_value=fake_client,
    ):
        idx = DatabricksVectorIndex(endpoint_name="ep", index_name="cat.sch.episodic_idx")
        results = idx.search(query="hello", k=5, filter={"user_id": "u_1"})

    fake_client.get_index.assert_called_once_with(
        endpoint_name="ep",
        index_name="cat.sch.episodic_idx",
    )
    fake_index_obj.similarity_search.assert_called_once()
    call_kwargs = fake_index_obj.similarity_search.call_args.kwargs
    assert call_kwargs["query_text"] == "hello"
    assert call_kwargs["num_results"] == 5
    assert json.loads(call_kwargs["filters_json"]) == {"user_id": "u_1"}

    assert results == [
        {"id": "id-1", "text": "first text", "score": 0.92},
        {"id": "id-2", "text": "second text", "score": 0.84},
    ]


def test_databricks_vector_index_search_without_filter() -> None:
    fake_index_obj = MagicMock()
    fake_index_obj.similarity_search.return_value = {
        "result": {"row_count": 0, "data_array": []},
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
    }
    fake_client = MagicMock()
    fake_client.get_index.return_value = fake_index_obj

    with patch(
        "lakehouse_memory.vector_databricks.VectorSearchClient",
        return_value=fake_client,
    ):
        idx = DatabricksVectorIndex(endpoint_name="ep", index_name="cat.sch.episodic_idx")
        idx.search(query="x", k=3, filter=None)

    call_kwargs = fake_index_obj.similarity_search.call_args.kwargs
    # filters_json should be absent or None/empty when no filter provided
    assert call_kwargs.get("filters_json") in (None, "{}", "")


def test_ensure_indexes_creates_endpoint_when_missing() -> None:
    """If the endpoint doesn't exist, create it and poll until ONLINE."""
    fake_client = MagicMock()
    fake_client.list_endpoints.return_value = {"endpoints": []}
    fake_client.get_endpoint.side_effect = [
        {"endpoint_status": {"state": "PROVISIONING"}},
        {"endpoint_status": {"state": "ONLINE"}},
    ]
    fake_client.create_delta_sync_index.return_value = {"status": {"ready": False}}
    fake_client.get_index.side_effect = [
        {"status": {"ready": False}},
        {"status": {"ready": True}},
        {"status": {"ready": False}},
        {"status": {"ready": True}},
    ]

    with (
        patch(
            "lakehouse_memory.vector_databricks.VectorSearchClient",
            return_value=fake_client,
        ),
        patch("lakehouse_memory.vector_databricks.time.sleep"),
    ):
        config = MemoryConfig(catalog="cat", schema_name="sch")
        result = ensure_indexes(
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
            endpoint_name="ep",
            config=config,
        )

    fake_client.create_endpoint.assert_called_once()
    create_kwargs = fake_client.create_endpoint.call_args.kwargs
    assert create_kwargs["name"] == "ep"

    assert fake_client.create_delta_sync_index.call_count == 2
    indexes_created = [
        c.kwargs["index_name"] for c in fake_client.create_delta_sync_index.call_args_list
    ]
    assert "cat.sch.episodic_idx" in indexes_created
    assert "cat.sch.semantic_idx" in indexes_created

    assert set(result.keys()) == {"episodic", "semantic"}
    assert isinstance(result["episodic"], DatabricksVectorIndex)
    assert isinstance(result["semantic"], DatabricksVectorIndex)


def test_ensure_indexes_skips_endpoint_creation_when_already_online() -> None:
    fake_client = MagicMock()
    fake_client.list_endpoints.return_value = {
        "endpoints": [{"name": "ep", "endpoint_status": {"state": "ONLINE"}}],
    }
    fake_client.get_index.side_effect = [
        {"status": {"ready": True}},
        {"status": {"ready": True}},
    ]

    with (
        patch(
            "lakehouse_memory.vector_databricks.VectorSearchClient",
            return_value=fake_client,
        ),
        patch("lakehouse_memory.vector_databricks.time.sleep"),
    ):
        config = MemoryConfig(catalog="cat", schema_name="sch")
        ensure_indexes(
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
            endpoint_name="ep",
            config=config,
        )

    fake_client.create_endpoint.assert_not_called()
    assert fake_client.create_delta_sync_index.call_count == 2


def test_ensure_indexes_is_idempotent_when_indexes_already_exist() -> None:
    """If indexes already exist (create_delta_sync_index raises a conflict-like error),
    ensure_indexes should swallow and proceed to readiness check."""
    fake_client = MagicMock()
    fake_client.list_endpoints.return_value = {
        "endpoints": [{"name": "ep", "endpoint_status": {"state": "ONLINE"}}],
    }
    fake_client.create_delta_sync_index.side_effect = Exception("RESOURCE_ALREADY_EXISTS")
    fake_client.get_index.side_effect = [
        {"status": {"ready": True}},
        {"status": {"ready": True}},
    ]

    with (
        patch(
            "lakehouse_memory.vector_databricks.VectorSearchClient",
            return_value=fake_client,
        ),
        patch("lakehouse_memory.vector_databricks.time.sleep"),
    ):
        config = MemoryConfig(catalog="cat", schema_name="sch")
        result = ensure_indexes(
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
            endpoint_name="ep",
            config=config,
        )

    assert set(result.keys()) == {"episodic", "semantic"}
