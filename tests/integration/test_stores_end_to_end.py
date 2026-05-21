"""Integration tests: write → wait for sync → read across all three stores."""
from __future__ import annotations

import time

import pytest

from .conftest import wait_for_searchable


def test_episodic_write_then_search_finds_event(live_memory) -> None:
    scoped = live_memory.with_scope(user_id="u_test1", session_id="s_1")
    event_id = scoped.episodic.write(
        event_type="page_view",
        payload={"page": "/billing"},
        text="user visited the billing page",
    )
    wait_for_searchable(scoped.episodic, "billing", event_id)


def test_episodic_recent_returns_ordered_events(live_memory) -> None:
    scoped = live_memory.with_scope(user_id="u_test2", session_id="s_1")
    ids = []
    for i in range(3):
        ids.append(
            scoped.episodic.write(event_type="ping", payload={"i": i}, text=f"event {i}")
        )
        # tiny stagger so created_at differs
        time.sleep(0.5)

    events = scoped.episodic.recent(limit=10)
    assert len(events) >= 3
    # DESC order: most recent first; ids[2] should appear before ids[0]
    returned_ids_in_order = [e.get("event_id") for e in events[:3]]
    assert returned_ids_in_order == list(reversed(ids))


def test_semantic_upsert_writes_to_delta(live_memory) -> None:
    """Verify the semantic upsert produces a row in the Delta table.

    Note: the live_memory fixture wires episodic_idx as Memory.index for both
    episodic and semantic stores (known design wart documented at top of the
    M2 plan). So we verify the SQL-level write via a direct query rather than
    through mem.semantic.retrieve(). The retriever path is covered in
    test_langchain_adapters.py via a separately-built Memory.
    """
    scoped = live_memory.with_scope(user_id="u_test3")
    fact_id = scoped.semantic.upsert(
        fact="User likes integration tests.",
        source="conversation:s_1",
    )
    rows = live_memory._client.execute(
        f"SELECT fact_id, fact FROM {live_memory._config.fqn('semantic')} "
        f"WHERE fact_id = :fact_id",
        {"fact_id": fact_id},
    )
    assert rows, "semantic upsert did not produce a row"
    assert rows[0]["fact_id"] == fact_id


def test_working_store_set_get_clear(live_memory) -> None:
    scoped = live_memory.with_scope(session_id="s_working_test")
    assert scoped.working.get("task") is None

    scoped.working.set("task", "draft_email")
    assert scoped.working.get("task") == "draft_email"

    scoped.working.set("task", "send_email")  # overwrite
    assert scoped.working.get("task") == "send_email"

    scoped.working.clear()
    assert scoped.working.get("task") is None


def test_working_clear_refuses_empty_scope(live_memory) -> None:
    """Safety guard: clear() with an empty scope must raise, not nuke the table."""
    unscoped = live_memory  # default scope is empty
    with pytest.raises(ValueError, match="empty scope"):
        unscoped.working.clear()


def test_scope_isolation_episodic(live_memory) -> None:
    """Two scopes writing to episodic don't see each other's events via .recent."""
    a = live_memory.with_scope(user_id="u_scope_a")
    b = live_memory.with_scope(user_id="u_scope_b")

    a.episodic.write(event_type="probe", payload={"who": "a"})
    b.episodic.write(event_type="probe", payload={"who": "b"})

    a_events = a.episodic.recent(limit=20, event_type="probe")
    b_events = b.episodic.recent(limit=20, event_type="probe")

    assert all(e.get("user_id") == "u_scope_a" for e in a_events)
    assert all(e.get("user_id") == "u_scope_b" for e in b_events)
