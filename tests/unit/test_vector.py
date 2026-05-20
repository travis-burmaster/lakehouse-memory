"""Tests for VectorIndex protocol and MockVectorIndex."""

from __future__ import annotations

from lakehouse_memory.vector import MockVectorIndex


def test_mock_index_upsert_then_search_returns_matching_records() -> None:
    idx = MockVectorIndex()
    idx.upsert(
        [
            {"id": "1", "text": "blue car", "user_id": "u_1"},
            {"id": "2", "text": "red bike", "user_id": "u_2"},
        ]
    )
    results = idx.search(query="blue", k=5)
    assert any(r["id"] == "1" for r in results)


def test_mock_index_search_applies_metadata_filter() -> None:
    idx = MockVectorIndex()
    idx.upsert(
        [
            {"id": "1", "text": "blue car", "user_id": "u_1"},
            {"id": "2", "text": "blue plane", "user_id": "u_2"},
        ]
    )
    results = idx.search(query="blue", k=5, filter={"user_id": "u_1"})
    assert [r["id"] for r in results] == ["1"]


def test_mock_index_search_respects_k() -> None:
    idx = MockVectorIndex()
    idx.upsert([{"id": str(i), "text": "blue thing", "user_id": "u_1"} for i in range(10)])
    results = idx.search(query="blue", k=3)
    assert len(results) == 3


def test_mock_index_delete_removes_records() -> None:
    idx = MockVectorIndex()
    idx.upsert([{"id": "1", "text": "x"}, {"id": "2", "text": "y"}])
    idx.delete(["1"])
    results = idx.search(query="x", k=5)
    assert all(r["id"] != "1" for r in results)


def test_mock_index_upsert_with_existing_id_replaces() -> None:
    idx = MockVectorIndex()
    idx.upsert([{"id": "1", "text": "first"}])
    idx.upsert([{"id": "1", "text": "second"}])
    results = idx.search(query="second", k=5)
    assert results[0]["text"] == "second"
    assert len([r for r in results if r["id"] == "1"]) == 1
