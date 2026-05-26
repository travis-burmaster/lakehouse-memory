"""Tests for SemanticStore."""

from __future__ import annotations

from unittest.mock import MagicMock

from lakehouse_memory.scope import Scope
from lakehouse_memory.stores.semantic import SemanticStore
from lakehouse_memory.vector import MockVectorIndex


def _client_returning(rows: list[dict[str, object]]) -> MagicMock:
    client = MagicMock()
    client.execute.return_value = rows
    return client


def test_upsert_issues_merge_and_upserts_vector() -> None:
    client = _client_returning([])
    idx = MockVectorIndex()
    store = SemanticStore(
        client=client,
        index=idx,
        fqn="prod.mem.semantic",
        scope=Scope(user_id="u_1"),
    )

    fact_id = store.upsert(fact="User prefers SQL.", source="conversation:s_1")

    sql, params = client.execute.call_args.args
    assert "MERGE INTO prod.mem.semantic" in sql
    assert params["fact"] == "User prefers SQL."
    assert params["source"] == "conversation:s_1"
    assert params["text"] == "User prefers SQL."
    assert params["user_id"] == "u_1"
    assert params["fact_id"] == fact_id

    results = idx.search("prefers SQL", k=5)
    assert any(r["id"] == fact_id for r in results)


def test_upsert_deduplicates_on_fact_within_scope() -> None:
    """Identical fact text in the same scope yields the same fact_id."""
    client = _client_returning([])
    idx = MockVectorIndex()
    store = SemanticStore(
        client=client,
        index=idx,
        fqn="prod.mem.semantic",
        scope=Scope(user_id="u_1"),
    )

    id_1 = store.upsert(fact="User prefers SQL.", source="a")
    id_2 = store.upsert(fact="User prefers SQL.", source="b")
    assert id_1 == id_2


def test_retrieve_uses_vector_index_with_scope_filter() -> None:
    idx = MockVectorIndex()
    idx.upsert(
        [
            {"id": "1", "text": "user prefers SQL", "user_id": "u_1"},
            {"id": "2", "text": "other user prefers Python", "user_id": "u_2"},
        ]
    )
    store = SemanticStore(
        client=_client_returning([]),
        index=idx,
        fqn="prod.mem.semantic",
        scope=Scope(user_id="u_1"),
    )

    results = store.retrieve("prefers", k=5)
    assert [r["id"] for r in results] == ["1"]


def test_forget_deletes_by_id_in_scope() -> None:
    client = _client_returning([])
    idx = MockVectorIndex()
    idx.upsert([{"id": "abc", "text": "x", "user_id": "u_1"}])
    store = SemanticStore(
        client=client,
        index=idx,
        fqn="prod.mem.semantic",
        scope=Scope(user_id="u_1"),
    )

    store.forget("abc")

    sql, params = client.execute.call_args.args
    assert "DELETE FROM prod.mem.semantic" in sql
    assert "fact_id = :fact_id" in sql
    assert "user_id = :user_id" in sql
    assert params == {"fact_id": "abc", "user_id": "u_1"}
    assert idx.search("x", k=5) == []


def test_semantic_trigger_sync_delegates_to_index() -> None:
    from unittest.mock import MagicMock

    idx = MagicMock()
    store = SemanticStore(client=MagicMock(), index=idx, fqn="c.s.semantic", scope=Scope())
    store.trigger_sync()
    idx.trigger_sync.assert_called_once_with()
