"""Vector index abstractions.

`VectorIndex` is the structural Protocol stores use for embedding-aware
retrieval. `MockVectorIndex` is an in-memory implementation used by unit
tests; the real Databricks Vector Search-backed implementation lands in
Plan 2 (M2).
"""

from __future__ import annotations

from typing import Any, Protocol


class VectorIndex(Protocol):
    def upsert(self, records: list[dict[str, Any]]) -> None: ...

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def delete(self, ids: list[str]) -> None: ...


class MockVectorIndex:
    """In-memory mock used in unit tests.

    Search uses a naive substring match against the record's `text` column.
    This is sufficient for testing store behavior — real semantic retrieval
    is the production implementation's job.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for r in records:
            self._records[r["id"]] = dict(r)

    def search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = list(self._records.values())
        if filter:
            candidates = [r for r in candidates if all(r.get(f) == v for f, v in filter.items())]
        q = query.lower()
        scored = sorted(
            ((q in r.get("text", "").lower(), r) for r in candidates),
            key=lambda t: (not t[0],),
        )
        return [r for matched, r in scored if matched][:k] or [r for _, r in scored][:k]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._records.pop(i, None)
