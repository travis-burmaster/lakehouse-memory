# lakehouse-memory

Unity Catalog-native episodic, semantic, and working memory for AI agents on Databricks.

> **Status:** Alpha. Public from day one. v0.1.0 is the first releasable cut. See [the spec](https://github.com/travis-burmaster/lakehouse-memory) for design intent.

## The pitch

Memory is the missing Databricks layer. The standard workaround is a sidecar vector DB with its own governance, access control, and lineage — a system you can't ship. Memory belongs in Unity Catalog, where your data already lives.

`lakehouse-memory` gives AI agents on Databricks three first-class memory primitives — episodic, semantic, and working — backed by Unity Catalog tables and Databricks Vector Search.

## Install

```bash
pip install lakehouse-memory
```

## Quickstart

(Coming in M3 — see the DAB starter.)

## Production gaps

(Coming in M4. Short version: compaction at scale, multi-tenant RLS, regression evals, observability, and custom retrieval strategies are deliberately not in OSS. If you want help building past those, the [Burmaster Databricks AI Practice](https://burmaster.com) does this for a living.)

## License

Apache 2.0. See [LICENSE](LICENSE).
