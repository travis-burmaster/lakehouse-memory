"""Unity Catalog DDL templates and idempotent provisioning."""

from __future__ import annotations

from typing import Protocol


class _Executor(Protocol):
    def execute(
        self, sql: str, params: dict[str, object] | None = None
    ) -> list[dict[str, object]]: ...


_EPISODIC_TEMPLATE = """\
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.episodic (
    event_id STRING NOT NULL,
    event_type STRING NOT NULL,
    payload STRING NOT NULL,
    text STRING,
    user_id STRING,
    session_id STRING,
    agent_id STRING,
    created_at TIMESTAMP NOT NULL
) USING DELTA"""

_SEMANTIC_TEMPLATE = """\
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.semantic (
    fact_id STRING NOT NULL,
    fact STRING NOT NULL,
    source STRING,
    text STRING NOT NULL,
    user_id STRING,
    session_id STRING,
    agent_id STRING,
    updated_at TIMESTAMP NOT NULL
) USING DELTA"""

_WORKING_TEMPLATE = """\
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.working (
    key STRING NOT NULL,
    value STRING NOT NULL,
    user_id STRING,
    session_id STRING,
    agent_id STRING,
    updated_at TIMESTAMP NOT NULL
) USING DELTA"""


def episodic_ddl(catalog: str, schema: str) -> str:
    return _EPISODIC_TEMPLATE.format(catalog=catalog, schema=schema)


def semantic_ddl(catalog: str, schema: str) -> str:
    return _SEMANTIC_TEMPLATE.format(catalog=catalog, schema=schema)


def working_ddl(catalog: str, schema: str) -> str:
    return _WORKING_TEMPLATE.format(catalog=catalog, schema=schema)


class SchemaProvisioner:
    """Applies the three memory table DDLs idempotently."""

    def __init__(self, client: _Executor, catalog: str, schema: str) -> None:
        self._client = client
        self._catalog = catalog
        self._schema = schema

    def apply(self) -> None:
        self._client.execute(episodic_ddl(self._catalog, self._schema))
        self._client.execute(semantic_ddl(self._catalog, self._schema))
        self._client.execute(working_ddl(self._catalog, self._schema))
