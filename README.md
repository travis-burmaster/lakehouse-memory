# lakehouse-memory

[![PyPI](https://img.shields.io/pypi/v/lakehouse-memory.svg?label=pypi&include_prereleases)](https://pypi.org/project/lakehouse-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/travis-burmaster/lakehouse-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/travis-burmaster/lakehouse-memory/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://travis-burmaster.github.io/lakehouse-memory/)

Unity Catalog-native episodic, semantic, and working memory for AI agents on Databricks.

> **Status:** Stable (`0.1.0`). Public from day one. The core library, LangChain adapters, DAB starter (M3), and docs site (M4) are shipped. See [the docs](https://travis-burmaster.github.io/lakehouse-memory/) for full documentation.

## The pitch

Memory is the missing Databricks layer. The standard workaround is a sidecar vector DB with its own governance, access control, and lineage — a system you can't ship. Memory belongs in Unity Catalog, where your data already lives.

`lakehouse-memory` gives AI agents on Databricks three first-class memory primitives — episodic, semantic, and working — backed by Unity Catalog tables and Databricks Vector Search.

## Install

```bash
pip install lakehouse-memory
```

> **Migrating from a pre-release:** `Memory(index=...)` was removed in 0.1.0. Use
> `Memory(config, client, episodic_index=idx, semantic_index=idx)` or, preferably,
> `Memory.from_databricks(...)`.

## Quickstart with the DAB starter (recommended)

Bootstrap the whole reference architecture — UC tables, Vector Search indexes,
and a working chat agent — in your Databricks workspace:

```bash
databricks bundle init https://github.com/travis-burmaster/lakehouse-memory \
  --template-dir templates/lakehouse-memory-bundle \
  --output-dir my-memory-demo
cd my-memory-demo
databricks bundle deploy
databricks bundle run setup_job
```

You'll be prompted for your catalog, schema, Vector Search endpoint, SQL
warehouse HTTP path, and LLM serving endpoint. (`--output-dir` is the project
root itself, not a parent directory.) After `setup_job` finishes, open
`notebooks/02_chat_agent.ipynb` and run all cells.

The setup job typically takes ~15 minutes end-to-end: the bulk is the library
install plus the one-time provisioning of two Delta Sync Vector Search indexes
(which sync serially on workspaces with a single-pipeline quota). Subsequent
runs against an already-provisioned schema are fast.

## Manual setup (advanced)

```python
from lakehouse_memory import Memory, MemoryConfig, Scope
from lakehouse_memory.client import SqlConnectorClient
from lakehouse_memory.vector_databricks import DatabricksVectorIndex
import os

config = MemoryConfig(catalog="main", schema_name="agent_memory")

client = SqlConnectorClient(
    server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
)

index = DatabricksVectorIndex(
    endpoint_name=os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT"],
    index_name=f"{config.catalog}.{config.schema_name}.episodic_idx",
    workspace_url=os.environ["DATABRICKS_HOST"],
    access_token=os.environ["DATABRICKS_TOKEN"],
    columns=["event_id", "text", "user_id", "session_id", "agent_id"],
)

mem = Memory(config=config, client=client, index=index, scope=Scope(user_id="u_1"))
mem.provision(
    vector_search_endpoint=os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT"],
    workspace_url=os.environ["DATABRICKS_HOST"],
    access_token=os.environ["DATABRICKS_TOKEN"],
)

# Write a fact
mem.semantic.upsert(fact="User prefers SQL over Python.")

# Delta Sync indexes are TRIGGERED — explicitly fire the sync after writes.
# (For production, consider switching to CONTINUOUS pipelines.)
mem.semantic._index.trigger_sync()

# Wait for sync; production code would use exponential backoff
import time; time.sleep(15)

facts = mem.semantic.retrieve("language preferences", k=3)
```

**LangChain integration:**

```python
chat = mem.as_langchain_chat_history(limit=50)
retriever = mem.as_langchain_retriever(k=5)
```

## Production gaps

(Coming in M4. Short version: compaction at scale, multi-tenant RLS, regression evals, observability, and custom retrieval strategies are deliberately not in OSS. If you want help building past those, the [Burmaster Databricks AI Practice](https://burmaster.com) does this for a living.)

## License

Apache 2.0. See [LICENSE](LICENSE).
