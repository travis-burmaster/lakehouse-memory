"""Tests for the Memory composition root."""

from __future__ import annotations

from unittest.mock import MagicMock

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
