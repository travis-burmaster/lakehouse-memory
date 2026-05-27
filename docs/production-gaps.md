# Production Gaps

`lakehouse-memory` is a pre-release OSS library. The core primitives work and
are workspace-validated, but several concerns are deliberately out of scope for
the open-source release.

---

## Gap table

| Gap | Why it's out of scope | What you'd need |
|-----|----------------------|-----------------|
| **Compaction at scale** | Delta Sync indexes don't auto-compact episodic tables; long-running agents accumulate millions of rows | Scheduled Spark jobs or DLT pipelines to summarise and prune old events |
| **Multi-tenant row-level security** | Unity Catalog RLS is workspace-specific; the library uses `Scope` for logical filtering only | Column masking + row filter policies, tested per-tenant |
| **Regression evals** | No golden-set harness ships with the library | MLflow evaluation runs comparing retrieval quality across versions |
| **Observability** | No tracing, latency histograms, or cache-hit metrics | MLflow tracing, custom dashboards, alerting on VS sync lag |
| **Custom retrieval strategies** | Only top-k cosine similarity is supported | MMR, hybrid BM25+vector, re-ranking with a cross-encoder |
| **Continuous VS pipelines** | Default indexes are TRIGGERED; writes are not immediately searchable | Switch to CONTINUOUS Delta Sync pipelines for sub-second freshness |

---

## The compactor seam

The library exposes a `Compactor` class in `lakehouse_memory.compactor` as a
seam for implementing summarisation and pruning. It is not fully implemented in
the OSS release — it defines the interface but defers the scheduling logic to
the operator.

A production compactor would:

1. Scan episodic events older than a configurable window
2. Summarise them with an LLM call (e.g., "sessions from last week said X")
3. Upsert the summary as a semantic fact
4. Delete or archive the original episodic rows

---

## Practice CTA

If you need help implementing past these gaps — multi-tenant RLS, continuous
pipelines, retrieval evals, or production observability — the
[Burmaster Databricks AI Practice](https://burmaster.com) does this for a living.
