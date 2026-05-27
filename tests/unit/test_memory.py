"""Tests for the Memory composition root."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lakehouse_memory.config import MemoryConfig
from lakehouse_memory.memory import Memory
from lakehouse_memory.scope import Scope
from lakehouse_memory.vector import MockVectorIndex


def test_memory_exposes_three_stores() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(user_id="u_1"),
    )
    assert mem.episodic is not None
    assert mem.semantic is not None
    assert mem.working is not None


def test_memory_stores_use_correct_fqns() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(user_id="u_1"),
    )
    assert mem.episodic._fqn == "prod.mem.episodic"
    assert mem.semantic._fqn == "prod.mem.semantic"
    assert mem.working._fqn == "prod.mem.working"


def test_memory_provision_applies_schema() -> None:
    client = MagicMock()
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=client,
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(),
    )
    mem.provision()

    assert client.execute.call_count == 4
    statements = [c.args[0] for c in client.execute.call_args_list]
    assert any("CREATE SCHEMA IF NOT EXISTS" in s and "prod.mem" in s for s in statements)
    assert any("prod.mem.episodic" in s for s in statements)
    assert any("prod.mem.semantic" in s for s in statements)
    assert any("prod.mem.working" in s for s in statements)


def test_memory_with_scope_returns_new_memory_with_merged_scope() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(user_id="u_1"),
    )
    scoped = mem.with_scope(session_id="s_1")
    assert scoped.scope == Scope(user_id="u_1", session_id="s_1")
    assert mem.scope == Scope(user_id="u_1")  # original unchanged


def test_memory_provision_without_endpoint_runs_only_schema() -> None:
    """When vector_search_endpoint is None, provision() only applies the schema."""
    client = MagicMock()
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=client,
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(),
    )
    mem.provision()  # no vector_search_endpoint
    # Four DDL statements were issued: CREATE SCHEMA + three table DDLs
    assert client.execute.call_count == 4


def test_memory_provision_with_endpoint_calls_ensure_indexes() -> None:
    """When vector_search_endpoint is given, ensure_indexes is invoked."""
    client = MagicMock()
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=client,
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(),
    )

    with patch("lakehouse_memory.memory.ensure_indexes") as mock_ensure:
        mem.provision(
            vector_search_endpoint="vs_ep",
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
        )

    # Schema still applied (CREATE SCHEMA + three table DDLs)
    assert client.execute.call_count == 4
    # ensure_indexes called with the right args
    mock_ensure.assert_called_once()
    call_kwargs = mock_ensure.call_args.kwargs
    assert call_kwargs["endpoint_name"] == "vs_ep"
    assert call_kwargs["workspace_url"] == "https://example.cloud.databricks.com"
    assert call_kwargs["access_token"] == "dapi-test"
    assert call_kwargs["config"].catalog == "prod"
    assert call_kwargs["config"].schema_name == "mem"


def test_memory_provision_requires_creds_when_endpoint_given() -> None:
    """If vector_search_endpoint is set, workspace_url and access_token must be provided."""
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        episodic_index=MockVectorIndex(),
        semantic_index=MockVectorIndex(),
        scope=Scope(),
    )
    with pytest.raises(ValueError, match="workspace_url"):
        mem.provision(vector_search_endpoint="vs_ep")


def test_memory_explicit_per_store_indexes() -> None:
    ep = MockVectorIndex()
    sem = MockVectorIndex()
    mem = Memory(
        config=MemoryConfig(catalog="c", schema_name="s"),
        client=MagicMock(),
        episodic_index=ep,
        semantic_index=sem,
        scope=Scope(user_id="u_1"),
    )
    assert mem.episodic._index is ep
    assert mem.semantic._index is sem


def test_memory_requires_both_indexes() -> None:
    with pytest.raises(TypeError):
        Memory(
            config=MemoryConfig(catalog="c", schema_name="s"),
            client=MagicMock(),
        )


def test_memory_with_scope_preserves_per_store_indexes() -> None:
    ep = MockVectorIndex()
    sem = MockVectorIndex()
    mem = Memory(
        config=MemoryConfig(catalog="c", schema_name="s"),
        client=MagicMock(),
        episodic_index=ep,
        semantic_index=sem,
        scope=Scope(user_id="u_1"),
    )
    scoped = mem.with_scope(session_id="s_1")
    assert scoped.episodic._index is ep
    assert scoped.semantic._index is sem
    assert scoped.scope == Scope(user_id="u_1", session_id="s_1")


def test_from_databricks_builds_wired_memory() -> None:
    from unittest.mock import patch

    with (
        patch("lakehouse_memory.memory.SqlConnectorClient") as mock_client_cls,
        patch("lakehouse_memory.memory.DatabricksVectorIndex") as mock_idx_cls,
    ):
        mem = Memory.from_databricks(
            catalog="prod",
            schema_name="mem",
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
            http_path="/sql/1.0/warehouses/abc",
            vector_search_endpoint="vs_ep",
            scope=Scope(user_id="u_1"),
        )

    client_kwargs = mock_client_cls.call_args.kwargs
    assert client_kwargs["server_hostname"] == "example.cloud.databricks.com"
    assert client_kwargs["http_path"] == "/sql/1.0/warehouses/abc"
    assert client_kwargs["access_token"] == "dapi-test"

    assert mock_idx_cls.call_count == 2
    index_names = {c.kwargs["index_name"] for c in mock_idx_cls.call_args_list}
    assert index_names == {"prod.mem.episodic_idx", "prod.mem.semantic_idx"}

    assert mem._vs_endpoint == "vs_ep"
    assert mem._workspace_url == "https://example.cloud.databricks.com"
    assert mem._access_token == "dapi-test"
    assert mem.scope == Scope(user_id="u_1")


def test_from_databricks_provision_uses_stashed_creds() -> None:
    from unittest.mock import patch

    with (
        patch("lakehouse_memory.memory.SqlConnectorClient"),
        patch("lakehouse_memory.memory.DatabricksVectorIndex"),
        patch("lakehouse_memory.memory.SchemaProvisioner") as mock_prov,
        patch("lakehouse_memory.memory.ensure_indexes") as mock_ensure,
    ):
        mem = Memory.from_databricks(
            catalog="prod",
            schema_name="mem",
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
            http_path="/sql/1.0/warehouses/abc",
            vector_search_endpoint="vs_ep",
        )
        mem.provision()

    mock_prov.return_value.apply.assert_called_once()
    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.kwargs["endpoint_name"] == "vs_ep"
