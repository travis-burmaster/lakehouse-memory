# Quickstart

There are three paths to a running `Memory` instance:

1. **DAB starter** (recommended) — one command bootstraps the entire reference architecture
2. **Manual setup** — wire up a `Memory` directly in Python
3. **LangChain integration** — drop into `RunnableWithMessageHistory` or a retrieval chain

---

## DAB starter (recommended)

The Databricks Asset Bundle starter provisions Unity Catalog tables, Vector
Search indexes, and a working chat agent notebook in your workspace:

```bash
databricks bundle init https://github.com/travis-burmaster/lakehouse-memory \
  --template-dir templates/lakehouse-memory-bundle \
  --output-dir my-memory-demo
cd my-memory-demo
databricks bundle deploy
databricks bundle run setup_job
```

You will be prompted for:

- Unity Catalog catalog and schema names
- Vector Search endpoint name
- SQL Warehouse HTTP path
- LLM serving endpoint name

After `setup_job` completes (~15 minutes end-to-end, mostly Vector Search index
provisioning), open `notebooks/02_chat_agent.ipynb` and run all cells.

---

## Manual setup

```python
from lakehouse_memory import Memory, MemoryConfig, Scope
import os

mem = Memory.from_databricks(
    catalog="my_catalog",
    schema_name="agent_memory",
    workspace_url=os.environ["DATABRICKS_HOST"],
    access_token=os.environ["DATABRICKS_TOKEN"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    vector_search_endpoint=os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT"],
)

# Idempotently create UC tables + Vector Search indexes
mem.provision()

# Scope to a specific user and session
scoped = mem.with_scope(user_id="u_1", session_id="s_1")

# Write a semantic fact
scoped.semantic.upsert(fact="User prefers SQL over Python.")

# Delta Sync indexes are triggered — fire the sync after writes
scoped.semantic._index.trigger_sync()

# Wait for sync (production code uses exponential backoff)
import time; time.sleep(15)

# Retrieve similar facts
facts = scoped.semantic.retrieve("language preferences", k=3)
print(facts)
```

---

## LangChain integration

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# Wire up the LangChain adapters
chat_history = mem.as_langchain_chat_history(limit=50)
retriever = mem.as_langchain_retriever(k=5)

# Use in a chain
chain_with_history = RunnableWithMessageHistory(
    your_chain,
    lambda session_id: mem.with_scope(session_id=session_id).as_langchain_chat_history(),
    input_messages_key="input",
    history_messages_key="history",
)
```

The `LakehouseChatHistory` adapter persists each turn as an episodic event.
The `LakehouseSemanticRetriever` adapter returns `Document` objects from the
semantic store.

!!! note "Scope per session"
    Call `mem.with_scope(session_id="...")` before constructing LangChain adapters
    to isolate history and retrieval per conversation.
