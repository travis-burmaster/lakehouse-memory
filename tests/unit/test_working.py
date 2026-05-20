"""Tests for WorkingStore."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lakehouse_memory.scope import Scope
from lakehouse_memory.stores.working import WorkingStore


def _client_returning(rows: list[dict[str, object]]) -> MagicMock:
    client = MagicMock()
    client.execute.return_value = rows
    return client


def test_set_issues_merge_with_scope_columns() -> None:
    client = _client_returning([])
    store = WorkingStore(
        client=client,
        fqn="prod.mem.working",
        scope=Scope(user_id="u_1", session_id="s_1"),
    )
    store.set("current_task", "draft")

    assert client.execute.call_count == 1
    sql, params = client.execute.call_args.args
    assert "MERGE INTO prod.mem.working" in sql
    assert params["key"] == "current_task"
    assert params["value"] == "draft"
    assert params["user_id"] == "u_1"
    assert params["session_id"] == "s_1"


def test_get_issues_select_with_scope_filter() -> None:
    client = _client_returning([{"value": "draft"}])
    store = WorkingStore(
        client=client,
        fqn="prod.mem.working",
        scope=Scope(user_id="u_1", session_id="s_1"),
    )
    value = store.get("current_task")

    sql, params = client.execute.call_args.args
    assert "FROM prod.mem.working" in sql
    assert "WHERE key = :key" in sql
    assert "session_id = :session_id" in sql
    assert "user_id = :user_id" in sql
    assert params["key"] == "current_task"
    assert value == "draft"


def test_get_returns_none_when_missing() -> None:
    client = _client_returning([])
    store = WorkingStore(client=client, fqn="prod.mem.working", scope=Scope(session_id="s_1"))
    assert store.get("nope") is None


def test_clear_issues_delete_with_scope_filter() -> None:
    client = _client_returning([])
    store = WorkingStore(
        client=client,
        fqn="prod.mem.working",
        scope=Scope(session_id="s_1"),
    )
    store.clear()

    sql, params = client.execute.call_args.args
    assert "DELETE FROM prod.mem.working" in sql
    assert "session_id = :session_id" in sql
    assert params == {"session_id": "s_1"}


def test_clear_refuses_unscoped() -> None:
    """Refuse to delete the entire table when scope is empty."""
    client = _client_returning([])
    store = WorkingStore(client=client, fqn="prod.mem.working", scope=Scope())
    with pytest.raises(ValueError, match="empty scope"):
        store.clear()
