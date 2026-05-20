"""Episodic memory: append-only time-ordered events."""
from __future__ import annotations

import json
import uuid
from typing import Any

from lakehouse_memory.client import DatabricksClient
from lakehouse_memory.scope import Scope
from lakehouse_memory.vector import VectorIndex


class EpisodicStore:
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

    def write(
        self,
        event_type: str,
        payload: dict[str, Any],
        text: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        scope_params = self._scope.to_metadata_filter()
        params: dict[str, Any] = {
            "event_id": event_id,
            "event_type": event_type,
            "payload": json.dumps(payload),
            "text": text,
            **scope_params,
        }
        scope_cols = sorted(scope_params)
        col_list = ", ".join(["event_id", "event_type", "payload", "text", "created_at", *scope_cols])
        val_list = ", ".join([
            ":event_id",
            ":event_type",
            ":payload",
            ":text",
            "CURRENT_TIMESTAMP()",
            *[f":{c}" for c in scope_cols],
        ])
        self._client.execute(
            f"INSERT INTO {self._fqn} ({col_list}) VALUES ({val_list})",
            params,
        )

        if text is not None:
            record: dict[str, Any] = {"id": event_id, "text": text, **scope_params}
            self._index.upsert([record])

        return event_id

    def recent(self, limit: int = 20, event_type: str | None = None) -> list[dict[str, Any]]:
        scope_sql, scope_params = self._scope.to_where_clause()
        conditions: list[str] = []
        params: dict[str, Any] = dict(scope_params)
        if scope_sql:
            conditions.append(scope_sql)
        if event_type is not None:
            conditions.append("event_type = :event_type")
            params["event_type"] = event_type
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM {self._fqn} {where} ORDER BY created_at DESC LIMIT {int(limit)}"
        return self._client.execute(sql, params)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self._index.search(query=query, k=k, filter=self._scope.to_metadata_filter() or None)
