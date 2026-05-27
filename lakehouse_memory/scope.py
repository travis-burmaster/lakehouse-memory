"""Identity scoping for memory reads and writes.

`Scope` is the only place where scope filter SQL and metadata filters are
constructed. Every store applies these filters automatically to every read.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Scope:
    """Identity scope that constrains memory reads and tags memory writes.

    A ``Scope`` represents a specific combination of identity dimensions:
    ``user_id``, ``session_id``, and/or ``agent_id``.  Any subset of these
    may be set; unset fields act as wildcards — they are absent from SQL
    ``WHERE`` clauses and vector metadata filters, so they match all values.

    ``Scope`` is the single source of truth for scope-related SQL and vector
    filter construction.  Every store applies these filters automatically to
    every read operation and includes all set fields as columns on every write.

    Instances are frozen (immutable).  Use ``merge`` to derive a new ``Scope``
    with some fields overridden.

    Attributes:
        user_id: Identifies the end-user whose memory is being accessed.
        session_id: Identifies the conversation session.
        agent_id: Identifies the agent (or agent variant) operating on memory.
    """

    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None

    def to_where_clause(self) -> tuple[str, dict[str, str]]:
        """Build a SQL WHERE-clause fragment and a bound-parameter map.

        Clauses are AND-joined in stable alphabetical field order so that the
        generated SQL is deterministic across Python versions and runtimes.

        Returns:
            A 2-tuple ``(clause, params)`` where *clause* is a non-empty SQL
            fragment like ``"agent_id = :agent_id AND user_id = :user_id"``
            and *params* is the corresponding ``{name: value}`` dict for
            parameterised queries.  Both are empty (``""`` and ``{}``) when no
            fields are set.
        """
        params = self._set_fields()
        if not params:
            return "", {}
        clauses = [f"{name} = :{name}" for name in sorted(params)]
        return " AND ".join(clauses), params

    def to_metadata_filter(self) -> dict[str, str]:
        """Build a metadata filter dict for Databricks Vector Search queries.

        Returns:
            A ``{field_name: value}`` dict containing only the fields that are
            set on this scope.  An empty dict is returned when no fields are
            set (i.e., no filtering is applied).
        """
        return self._set_fields()

    def merge(self, other: Scope) -> Scope:
        """Return a new Scope with *other*'s set fields overriding self's.

        Fields that are ``None`` on *other* are inherited from ``self``
        unchanged.  This allows incremental narrowing of scope without losing
        previously set dimensions.

        Args:
            other: A ``Scope`` whose non-``None`` fields will override the
                corresponding fields on ``self``.

        Returns:
            A new frozen ``Scope`` instance with the merged field values.
        """
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
