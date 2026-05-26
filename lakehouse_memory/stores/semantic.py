"""Semantic memory: durable, upsertable, vector-searched facts."""

from __future__ import annotations

import hashlib
from typing import Any

from lakehouse_memory.client import DatabricksClient
from lakehouse_memory.scope import Scope
from lakehouse_memory.vector import VectorIndex


def _fact_id(fact: str, scope: Scope) -> str:
    """Deterministic id so identical fact text within a scope dedupes."""
    scope_key = "|".join(f"{k}={v}" for k, v in sorted(scope.to_metadata_filter().items()))
    payload = f"{scope_key}:{fact}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class SemanticStore:
    def __init__(
        self,
        client: DatabricksClient,
        index: VectorIndex,
        fqn: str,
        scope: Scope,
    ) -> None:
        self._client = client
        self._index = index
        self._fqn = fqn
        self._scope = scope

    def upsert(self, fact: str, source: str | None = None) -> str:
        fact_id = _fact_id(fact, self._scope)
        scope_params = self._scope.to_metadata_filter()
        params: dict[str, Any] = {
            "fact_id": fact_id,
            "fact": fact,
            "source": source,
            "text": fact,
            **scope_params,
        }
        scope_cols = sorted(scope_params)

        source_aliases = ", ".join(
            [
                ":fact_id AS fact_id",
                ":fact AS fact",
                ":source AS source",
                ":text AS text",
                *[f":{c} AS {c}" for c in scope_cols],
            ]
        )
        insert_cols = ", ".join(["fact_id", "fact", "source", "text", "updated_at", *scope_cols])
        insert_vals = ", ".join(
            [
                "s.fact_id",
                "s.fact",
                "s.source",
                "s.text",
                "CURRENT_TIMESTAMP()",
                *[f"s.{c}" for c in scope_cols],
            ]
        )

        sql = (
            f"MERGE INTO {self._fqn} t "
            f"USING (SELECT {source_aliases}) s "
            f"ON t.fact_id = s.fact_id "
            f"WHEN MATCHED THEN UPDATE SET "
            f"t.fact = s.fact, t.source = s.source, t.text = s.text, "
            f"t.updated_at = CURRENT_TIMESTAMP() "
            f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
        )
        self._client.execute(sql, params)

        record: dict[str, Any] = {"id": fact_id, "text": fact, **scope_params}
        self._index.upsert([record])
        return fact_id

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self._index.search(query=query, k=k, filter=self._scope.to_metadata_filter() or None)

    def trigger_sync(self) -> None:
        """Trigger the underlying Vector Search index sync (no-op for mock/CONTINUOUS)."""
        self._index.trigger_sync()

    def forget(self, fact_id: str) -> None:
        scope_sql, scope_params = self._scope.to_where_clause()
        conditions = ["fact_id = :fact_id"]
        params: dict[str, Any] = {"fact_id": fact_id, **scope_params}
        if scope_sql:
            conditions.append(scope_sql)
        sql = f"DELETE FROM {self._fqn} WHERE {' AND '.join(conditions)}"
        self._client.execute(sql, params)
        self._index.delete([fact_id])
