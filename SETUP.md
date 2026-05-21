# Integration test setup

The integration test suite at `tests/integration/` runs the library against
a real Databricks workspace. This document walks you through obtaining
each credential listed in `.env.example`.

## Prerequisites

- A Databricks workspace with **Unity Catalog enabled** and **Vector Search
  enabled** (the latter is a paid add-on on some plans).
- A user account that can:
  - Create and drop schemas in some catalog.
  - Use a SQL warehouse.
  - Create / read Vector Search indexes (typically requires the
    "Workspace user" role plus catalog `USE_CATALOG` + `USE_SCHEMA` privileges).

## Steps

1. **Copy the template:**
   ```
   cp .env.example .env
   ```

2. **Get `DATABRICKS_HOST`:** It's the URL of your workspace
   (e.g. `https://dbc-12345abc-6789.cloud.databricks.com`).

3. **Get `DATABRICKS_TOKEN`:** In the workspace UI, click your username
   (top-right) → User Settings → Developer → Access tokens → Generate new
   token. Give it a description like "lakehouse-memory integration tests"
   and a short lifetime (e.g. 7 days). Copy the token.

4. **Get `DATABRICKS_HTTP_PATH`:** In the workspace UI, SQL Warehouses →
   pick one → Connection details → HTTP path
   (e.g. `/sql/1.0/warehouses/abc123def456`).

5. **Get `DATABRICKS_VECTOR_SEARCH_ENDPOINT`:** In the workspace UI,
   Compute → Vector Search. If no endpoint exists, create one (any
   `STANDARD` endpoint will do; cold start is ~5 min). Copy the endpoint
   name (not the URL).

6. **Pick `LAKEHOUSE_MEMORY_TEST_CATALOG`:** A Unity Catalog catalog you
   can write to. Often `main` (for personal workspaces) or a dedicated
   sandbox catalog.

7. **Leave `LAKEHOUSE_MEMORY_TEST_SCHEMA` as-is** unless you want a
   different prefix. The test conftest appends a random suffix per run so
   no two runs collide.

## Running the integration tests

```
source .venv/bin/activate
LAKEHOUSE_MEMORY_INTEGRATION=1 pytest tests/integration -v
```

Or, since `.env` already sets `LAKEHOUSE_MEMORY_INTEGRATION=1`:

```
source .venv/bin/activate
pytest tests/integration -v
```

The first run will take 1–5 minutes per test that polls for sync. Subsequent
runs against the same endpoint are faster.

## Cost / safety

- The test session creates an ephemeral schema per run and `DROP SCHEMA
  ... CASCADE` on teardown. No state persists.
- The Vector Search endpoint is treated as shared infrastructure and is
  **not** deleted by tests (creating an endpoint takes ~5 min).
- SQL warehouse usage is metered. Tests issue ~10–20 small queries; the
  cost should be cents at most for a serverless warehouse.

## Troubleshooting

- **`pyodbc` / `thrift` import errors:** Make sure the venv has
  `databricks-sql-connector>=3.0.0` installed.
- **`PERMISSION_DENIED` on Vector Search:** Your user needs the
  `Vector Search` workspace entitlement plus catalog privileges.
- **Index creation hangs past the 5-min timeout:** Bump
  `_INDEX_POLL_TIMEOUT_S` in `lakehouse_memory/vector_databricks.py`
  (the integration test polling helpers also have their own timeout).
