"""Databricks SQL client abstractions.

`DatabricksClient` is the structural Protocol the rest of the library uses.
`SqlConnectorClient` is the production implementation backed by
`databricks-sql-connector`. Tests use mocks that satisfy the Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

from databricks import sql


class DatabricksClient(Protocol):
    """Minimal SQL-execution surface the library needs."""

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def execute_many(self, sql: str, rows: list[dict[str, Any]]) -> None: ...


class SqlConnectorClient:
    """Production client backed by `databricks-sql-connector`."""

    def __init__(self, server_hostname: str, http_path: str, access_token: str) -> None:
        self._server_hostname = server_hostname
        self._http_path = http_path
        self._access_token = access_token

    def _connect(self) -> Any:
        return sql.connect(
            server_hostname=self._server_hostname,
            http_path=self._http_path,
            access_token=self._access_token,
        )

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            if not cur.description:
                return []
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def execute_many(self, sql: str, rows: list[dict[str, Any]]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
