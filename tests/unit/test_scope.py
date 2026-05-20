"""Tests for Scope filter construction."""

from __future__ import annotations

import dataclasses

import pytest

from lakehouse_memory.scope import Scope


def test_scope_empty_produces_no_filter() -> None:
    s = Scope()
    sql, params = s.to_where_clause()
    assert sql == ""
    assert params == {}


def test_scope_single_field_produces_clause() -> None:
    s = Scope(user_id="u_1")
    sql, params = s.to_where_clause()
    assert sql == "user_id = :user_id"
    assert params == {"user_id": "u_1"}


def test_scope_multi_field_anded_in_stable_order() -> None:
    s = Scope(user_id="u_1", session_id="s_1", agent_id="a_1")
    sql, params = s.to_where_clause()
    assert sql == "agent_id = :agent_id AND session_id = :session_id AND user_id = :user_id"
    assert params == {"agent_id": "a_1", "session_id": "s_1", "user_id": "u_1"}


def test_scope_metadata_filter_only_includes_set_fields() -> None:
    s = Scope(user_id="u_1", agent_id="a_1")
    assert s.to_metadata_filter() == {"user_id": "u_1", "agent_id": "a_1"}


def test_scope_merge_overrides_existing_fields() -> None:
    base = Scope(user_id="u_1", session_id="s_1")
    override = Scope(session_id="s_2", agent_id="a_1")
    merged = base.merge(override)
    assert merged == Scope(user_id="u_1", session_id="s_2", agent_id="a_1")


def test_scope_is_frozen() -> None:
    s = Scope(user_id="u_1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.user_id = "u_2"  # type: ignore[misc]
