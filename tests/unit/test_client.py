"""Tests for the DatabricksClient protocol and SqlConnectorClient wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from lakehouse_memory.client import DatabricksClient, SqlConnectorClient


def test_databricks_client_is_a_protocol() -> None:
    # SqlConnectorClient satisfies the DatabricksClient protocol structurally.
    client: DatabricksClient = SqlConnectorClient(
        server_hostname="example.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/abc",
        access_token="dapi-test",
    )
    assert hasattr(client, "execute")


def test_sql_connector_client_execute_returns_rows_as_dicts() -> None:
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [("v_1", 1), ("v_2", 2)]
    fake_cursor.description = [("name",), ("count",)]
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    with patch("lakehouse_memory.client.sql.connect", return_value=fake_conn):
        client = SqlConnectorClient(
            server_hostname="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/abc",
            access_token="dapi-test",
        )
        rows = client.execute("SELECT name, count FROM t")

    assert rows == [{"name": "v_1", "count": 1}, {"name": "v_2", "count": 2}]
    fake_cursor.execute.assert_called_once_with("SELECT name, count FROM t", {})


def test_sql_connector_client_passes_named_params() -> None:
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_cursor.description = []
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    with patch("lakehouse_memory.client.sql.connect", return_value=fake_conn):
        client = SqlConnectorClient(
            server_hostname="h",
            http_path="/p",
            access_token="t",
        )
        client.execute("SELECT * FROM t WHERE id = :id", {"id": "x"})

    fake_cursor.execute.assert_called_once_with("SELECT * FROM t WHERE id = :id", {"id": "x"})


def test_sql_connector_client_execute_many_dispatches_rows() -> None:
    fake_cursor = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    with patch("lakehouse_memory.client.sql.connect", return_value=fake_conn):
        client = SqlConnectorClient(
            server_hostname="h",
            http_path="/p",
            access_token="t",
        )
        client.execute_many(
            "INSERT INTO t (a, b) VALUES (:a, :b)",
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        )

    fake_cursor.executemany.assert_called_once_with(
        "INSERT INTO t (a, b) VALUES (:a, :b)",
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
    )
