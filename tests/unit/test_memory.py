"""Tests for the Memory composition root."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lakehouse_memory.config import MemoryConfig
from lakehouse_memory.memory import Memory
from lakehouse_memory.scope import Scope
from lakehouse_memory.vector import MockVectorIndex


def test_memory_exposes_three_stores() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        index=MockVectorIndex(),
        scope=Scope(user_id="u_1"),
    )
    assert mem.episodic is not None
    assert mem.semantic is not None
    assert mem.working is not None


def test_memory_stores_use_correct_fqns() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        index=MockVectorIndex(),
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
        index=MockVectorIndex(),
        scope=Scope(),
    )
    mem.provision()

    assert client.execute.call_count == 3
    statements = [c.args[0] for c in client.execute.call_args_list]
    assert any("prod.mem.episodic" in s for s in statements)
    assert any("prod.mem.semantic" in s for s in statements)
    assert any("prod.mem.working" in s for s in statements)


def test_memory_with_scope_returns_new_memory_with_merged_scope() -> None:
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=MagicMock(),
        index=MockVectorIndex(),
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
        index=MockVectorIndex(),
        scope=Scope(),
    )
    mem.provision()  # no vector_search_endpoint
    # Three DDL statements were issued (M1 behavior preserved)
    assert client.execute.call_count == 3


def test_memory_provision_with_endpoint_calls_ensure_indexes() -> None:
    """When vector_search_endpoint is given, ensure_indexes is invoked."""
    from unittest.mock import patch

    client = MagicMock()
    mem = Memory(
        config=MemoryConfig(catalog="prod", schema_name="mem"),
        client=client,
        index=MockVectorIndex(),
        scope=Scope(),
    )

    with patch("lakehouse_memory.vector_databricks.ensure_indexes") as mock_ensure:
        mem.provision(
            vector_search_endpoint="vs_ep",
            workspace_url="https://example.cloud.databricks.com",
            access_token="dapi-test",
        )

    # Schema still applied
    assert client.execute.call_count == 3
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
        index=MockVectorIndex(),
        scope=Scope(),
    )
    with pytest.raises(ValueError, match="workspace_url"):
        mem.provision(vector_search_endpoint="vs_ep")
