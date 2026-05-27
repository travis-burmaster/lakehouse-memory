# Concepts

`lakehouse-memory` models agent memory as three distinct primitives, each backed
by a Unity Catalog Delta table.

---

## Episodic memory

**What happened.** An append-only log of raw events — chat messages, tool
invocations, observations, errors — ordered by wall-clock time.

| Property | Value |
|----------|-------|
| Table | `<catalog>.<schema>.episodic` |
| Write pattern | append-only |
| Read pattern | `recent(limit=N, event_type=...)` |
| Index | Delta Sync Vector Search (`episodic_idx`) |

Use episodic memory to reconstruct a conversation, replay events for debugging,
or feed a LangChain `BaseChatMessageHistory`.

---

## Semantic memory

**What is true.** A upsert-friendly store of facts — user preferences, domain
knowledge, learned rules — keyed by `(user_id, fact_id)`.

| Property | Value |
|----------|-------|
| Table | `<catalog>.<schema>.semantic` |
| Write pattern | upsert by `fact_id` |
| Read pattern | `retrieve(query, k=N)` via vector similarity |
| Index | Delta Sync Vector Search (`semantic_idx`) |

Use semantic memory to store "User prefers SQL over Python" or "This project
uses Unity Catalog 13.3 LTS" and retrieve the most relevant facts at query time.

---

## Working memory

**What is active right now.** A key-value scratchpad scoped to the current agent
run — no vector index, no append overhead, just fast gets and sets.

| Property | Value |
|----------|-------|
| Table | `<catalog>.<schema>.working` |
| Write pattern | upsert by `key` |
| Read pattern | `get(key)` / `all()` |
| Index | none |

Use working memory to carry intermediate results, partial plans, or structured
state between agent steps within a single session.

---

## Eventual consistency

Databricks Vector Search Delta Sync indexes are **triggered** — they do not
replicate writes instantly. After calling `episodic.write()` or
`semantic.upsert()`, call `index.trigger_sync()` and wait for the sync to
complete before expecting the new data to appear in vector search results.

For high-write workloads, consider switching to **CONTINUOUS** pipeline mode
(see [Production Gaps](production-gaps.md)).

---

## Scoping

Every read and write is filtered by a `Scope` — a frozen triple of
`(user_id, session_id, agent_id)`. Any dimension left `None` acts as a wildcard.

```python
# Scope to a specific user and session
scoped = mem.with_scope(user_id="u_1", session_id="s_42")

# All reads and writes are now filtered/tagged to that scope
scoped.episodic.write(event_type="chat_message", payload={}, text="Hello")
scoped.semantic.retrieve("language preferences", k=3)
```

`Memory.with_scope()` returns a new `Memory` instance and shares the underlying
SQL client and vector indexes — no new connections are opened.
