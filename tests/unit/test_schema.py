"""Tests for DDL templates and SchemaProvisioner."""

from __future__ import annotations

from unittest.mock import MagicMock

from lakehouse_memory.schema import (
    SchemaProvisioner,
    episodic_ddl,
    semantic_ddl,
    working_ddl,
)


def test_episodic_ddl_includes_fqn_and_required_columns() -> None:
    sql = episodic_ddl("prod", "mem")
    assert "prod.mem.episodic" in sql
    assert "event_id STRING" in sql
    assert "event_type STRING" in sql
    assert "payload STRING" in sql
    assert "text STRING" in sql
    assert "user_id STRING" in sql
    assert "session_id STRING" in sql
    assert "agent_id STRING" in sql
    assert "created_at TIMESTAMP" in sql
    assert "USING DELTA" in sql
    assert "CREATE TABLE IF NOT EXISTS" in sql


def test_semantic_ddl_includes_fqn_and_required_columns() -> None:
    sql = semantic_ddl("prod", "mem")
    assert "prod.mem.semantic" in sql
    assert "fact_id STRING" in sql
    assert "fact STRING" in sql
    assert "source STRING" in sql
    assert "text STRING" in sql
    assert "user_id STRING" in sql
    assert "updated_at TIMESTAMP" in sql


def test_working_ddl_includes_fqn_and_required_columns() -> None:
    sql = working_ddl("prod", "mem")
    assert "prod.mem.working" in sql
    assert "key STRING" in sql
    assert "value STRING" in sql
    assert "session_id STRING" in sql
    assert "updated_at TIMESTAMP" in sql


def test_provisioner_apply_executes_four_ddls_in_order() -> None:
    client = MagicMock()
    provisioner = SchemaProvisioner(client=client, catalog="prod", schema="mem")
    provisioner.apply()

    assert client.execute.call_count == 4
    statements = [call.args[0] for call in client.execute.call_args_list]
    assert "CREATE SCHEMA IF NOT EXISTS" in statements[0]
    assert "prod.mem" in statements[0]
    assert "prod.mem.episodic" in statements[1]
    assert "prod.mem.semantic" in statements[2]
    assert "prod.mem.working" in statements[3]


def test_provisioner_apply_is_idempotent_via_if_not_exists() -> None:
    client = MagicMock()
    provisioner = SchemaProvisioner(client=client, catalog="prod", schema="mem")
    provisioner.apply()
    statements = [call.args[0] for call in client.execute.call_args_list]
    # First statement creates the schema idempotently
    assert "CREATE SCHEMA IF NOT EXISTS" in statements[0]
    # Remaining statements create tables idempotently
    for stmt in statements[1:]:
        assert "CREATE TABLE IF NOT EXISTS" in stmt
