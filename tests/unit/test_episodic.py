"""Tests for EpisodicStore."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from lakehouse_memory.scope import Scope
from lakehouse_memory.stores.episodic import EpisodicStore
from lakehouse_memory.vector import MockVectorIndex


def _client_returning(rows: list[dict[str, object]]) -> MagicMock:
    client = MagicMock()
    client.execute.return_value = rows
    return client


def test_write_inserts_into_delta_and_upserts_vector_when_text_present() -> None:
    client = _client_returning([])
    idx = MockVectorIndex()
    store = EpisodicStore(
        client=client,
        index=idx,
        fqn="prod.mem.episodic",
        scope=Scope(user_id="u_1"),
    )

    event_id = store.write(
        event_type="page_view",
        payload={"page": "/billing"},
        text="user visited the billing page",
    )

    sql, params = client.execute.call_args.args
    assert "INSERT INTO prod.mem.episodic" in sql
    assert params["event_id"] == event_id
    assert params["event_type"] == "page_view"
    assert params["payload"] == json.dumps({"page": "/billing"})
    assert params["text"] == "user visited the billing page"
    assert params["user_id"] == "u_1"

    results = idx.search("billing", k=5)
    assert any(r["id"] == event_id for r in results)


def test_write_skips_vector_when_text_is_none() -> None:
    client = _client_returning([])
    idx = MockVectorIndex()
    store = EpisodicStore(
        client=client,
        index=idx,
        fqn="prod.mem.episodic",
        scope=Scope(user_id="u_1"),
    )

    store.write(event_type="ping", payload={"ts": 1})
    assert idx.search("anything", k=5) == []


def test_recent_orders_by_created_at_desc_and_limits() -> None:
    client = _client_returning([{"event_id": "1"}])
    store = EpisodicStore(
        client=client,
        index=MockVectorIndex(),
        fqn="prod.mem.episodic",
        scope=Scope(user_id="u_1"),
    )

    rows = store.recent(limit=10)

    sql, params = client.execute.call_args.args
    assert "FROM prod.mem.episodic" in sql
    assert "ORDER BY created_at DESC" in sql
    assert "LIMIT 10" in sql
    assert "user_id = :user_id" in sql
    assert params == {"user_id": "u_1"}
    assert rows == [{"event_id": "1"}]


def test_recent_with_event_type_filter() -> None:
    client = _client_returning([])
    store = EpisodicStore(
        client=client,
        index=MockVectorIndex(),
        fqn="prod.mem.episodic",
        scope=Scope(user_id="u_1"),
    )

    store.recent(limit=5, event_type="page_view")
    sql, params = client.execute.call_args.args
    assert "event_type = :event_type" in sql
    assert params["event_type"] == "page_view"


def test_search_queries_vector_index_with_scope_filter() -> None:
    idx = MockVectorIndex()
    idx.upsert(
        [
            {"id": "1", "text": "user struggled with billing", "user_id": "u_1"},
            {"id": "2", "text": "other user struggled", "user_id": "u_2"},
        ]
    )
    store = EpisodicStore(
        client=_client_returning([]),
        index=idx,
        fqn="prod.mem.episodic",
        scope=Scope(user_id="u_1"),
    )

    results = store.search("struggled", k=5)
    assert [r["id"] for r in results] == ["1"]
