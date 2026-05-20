"""Compactor protocol — pluggable seam, deliberately empty.

Production compaction (episodic events → semantic facts via an LLM with
quality gates and a cost budget) is intentionally NOT part of OSS. Users
who hit the wall here can either implement their own Compactor or engage
the Burmaster Databricks AI Practice (https://burmaster.com).
"""
from __future__ import annotations

from typing import Protocol, TypedDict


class CompactionReport(TypedDict):
    episodic_collapsed: int
    semantic_created: int


class Compactor(Protocol):
    def compact(self) -> CompactionReport: ...


class NullCompactor:
    """No-op compactor. Returns a zero report."""

    def compact(self) -> CompactionReport:
        return CompactionReport(episodic_collapsed=0, semantic_created=0)
