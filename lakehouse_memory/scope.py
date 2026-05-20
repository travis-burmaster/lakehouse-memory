"""Identity scoping for memory reads and writes.

`Scope` is the only place where scope filter SQL and metadata filters are
constructed. Every store applies these filters automatically to every read.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Scope:
    """Identity scope for memory operations.

    Any combination of `user_id`, `session_id`, and `agent_id` may be set.
    Unset fields are not included in filters (i.e., they match everything
    permitted by other set fields).
    """

    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None

    def to_where_clause(self) -> tuple[str, dict[str, str]]:
        """Build a SQL WHERE-clause fragment and parameter map.

        Returns ("", {}) if no fields are set. Clauses are AND-joined in a
        stable alphabetical order so generated SQL is deterministic.
        """
        params = self._set_fields()
        if not params:
            return "", {}
        clauses = [f"{name} = :{name}" for name in sorted(params)]
        return " AND ".join(clauses), params

    def to_metadata_filter(self) -> dict[str, str]:
        """Build a metadata filter dict for vector index queries."""
        return self._set_fields()

    def merge(self, other: Scope) -> Scope:
        """Return a new Scope where `other`'s set fields override self's."""
        updates: dict[str, Any] = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if getattr(other, f.name) is not None
        }
        return replace(self, **updates)

    def _set_fields(self) -> dict[str, str]:
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None
        }
