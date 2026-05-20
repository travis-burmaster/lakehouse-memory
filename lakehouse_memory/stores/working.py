"""Working memory: session-scoped key/value state.

No vector index. Overwrite semantics. `clear()` requires a non-empty scope
to prevent accidental whole-table deletes.
"""
from __future__ import annotations

from typing import Any

from lakehouse_memory.client import DatabricksClient
from lakehouse_memory.scope import Scope


class WorkingStore:
    def __init__(self, client: DatabricksClient, fqn: str, scope: Scope) -> None:
        self._client = client
        self._fqn = fqn
        self._scope = scope

    def set(self, key: str, value: str) -> None:
        scope_params = self._scope.to_metadata_filter()
        params: dict[str, Any] = {"key": key, "value": value, **scope_params}
        scope_cols = sorted(scope_params)

        source_aliases = ", ".join([
            ":key AS key",
            ":value AS value",
            *[f":{c} AS {c}" for c in scope_cols],
        ])
        match_conds = " AND ".join(["t.key = s.key", *[f"t.{c} = s.{c}" for c in scope_cols]])
        insert_cols = ", ".join(["key", "value", "updated_at", *scope_cols])
        insert_vals = ", ".join([
            "s.key",
            "s.value",
            "CURRENT_TIMESTAMP()",
            *[f"s.{c}" for c in scope_cols],
        ])

        sql = (
            f"MERGE INTO {self._fqn} t "
            f"USING (SELECT {source_aliases}) s "
            f"ON {match_conds} "
            f"WHEN MATCHED THEN UPDATE SET t.value = s.value, t.updated_at = CURRENT_TIMESTAMP() "
            f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
        )
        self._client.execute(sql, params)

    def get(self, key: str) -> str | None:
        scope_sql, scope_params = self._scope.to_where_clause()
        where = "WHERE key = :key" + (f" AND {scope_sql}" if scope_sql else "")
        sql = f"SELECT value FROM {self._fqn} {where} LIMIT 1"
        rows = self._client.execute(sql, {"key": key, **scope_params})
        if not rows:
            return None
        value = rows[0].get("value")
        return str(value) if value is not None else None

    def clear(self) -> None:
        scope_sql, scope_params = self._scope.to_where_clause()
        if not scope_sql:
            raise ValueError("WorkingStore.clear() refuses to run with an empty scope")
        sql = f"DELETE FROM {self._fqn} WHERE {scope_sql}"
        self._client.execute(sql, scope_params)
