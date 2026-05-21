"""Integration test fixtures.

These tests are skipped unless LAKEHOUSE_MEMORY_INTEGRATION=1 is set.
See SETUP.md for the required environment variables.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LAKEHOUSE_MEMORY_INTEGRATION"):
    pytest.skip(
        "Set LAKEHOUSE_MEMORY_INTEGRATION=1 to run integration tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def workspace_url() -> str:
    return os.environ["DATABRICKS_HOST"]


@pytest.fixture(scope="session")
def access_token() -> str:
    return os.environ["DATABRICKS_TOKEN"]


@pytest.fixture(scope="session")
def http_path() -> str:
    return os.environ["DATABRICKS_HTTP_PATH"]


@pytest.fixture(scope="session")
def vector_search_endpoint() -> str:
    return os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT"]


@pytest.fixture(scope="session")
def test_catalog() -> str:
    return os.environ["LAKEHOUSE_MEMORY_TEST_CATALOG"]


@pytest.fixture(scope="session")
def ephemeral_schema_name() -> str:
    prefix = os.environ.get("LAKEHOUSE_MEMORY_TEST_SCHEMA", "lakehouse_memory_test")
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def sql_client(workspace_url: str, access_token: str, http_path: str):
    from lakehouse_memory.client import SqlConnectorClient

    hostname = workspace_url.replace("https://", "").replace("http://", "").rstrip("/")
    return SqlConnectorClient(
        server_hostname=hostname,
        http_path=http_path,
        access_token=access_token,
    )


@pytest.fixture(scope="session")
def live_memory(
    sql_client,
    workspace_url: str,
    access_token: str,
    vector_search_endpoint: str,
    test_catalog: str,
    ephemeral_schema_name: str,
) -> Iterator:
    """A Memory provisioned against the workspace with a temporary schema.

    On teardown: DROP SCHEMA CASCADE (drops the indexes inside it too).
    The Vector Search endpoint is NOT deleted.
    """
    from lakehouse_memory import Memory, MemoryConfig, Scope
    from lakehouse_memory.vector_databricks import DatabricksVectorIndex

    config = MemoryConfig(catalog=test_catalog, schema_name=ephemeral_schema_name)

    episodic_index = DatabricksVectorIndex(
        endpoint_name=vector_search_endpoint,
        index_name=f"{test_catalog}.{ephemeral_schema_name}.episodic_idx",
        workspace_url=workspace_url,
        access_token=access_token,
    )

    mem = Memory(
        config=config,
        client=sql_client,
        index=episodic_index,
        scope=Scope(),
    )

    mem.provision(
        vector_search_endpoint=vector_search_endpoint,
        workspace_url=workspace_url,
        access_token=access_token,
    )

    yield mem

    try:
        sql_client.execute(f"DROP SCHEMA IF EXISTS {test_catalog}.{ephemeral_schema_name} CASCADE")
    except Exception as e:
        print(f"Teardown DROP SCHEMA failed: {e}", flush=True)


def wait_for_searchable(store, query: str, expected_id: str, timeout_s: int = 180) -> None:
    """Poll store.search until expected_id appears in results or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        results = store.search(query, k=10)
        if any(r.get("id") == expected_id for r in results):
            return
        time.sleep(5)
    pytest.fail(f"id {expected_id!r} not searchable after {timeout_s}s")
