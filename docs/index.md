# lakehouse-memory

[![PyPI](https://img.shields.io/pypi/v/lakehouse-memory.svg?label=pypi&include_prereleases)](https://pypi.org/project/lakehouse-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/travis-burmaster/lakehouse-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/travis-burmaster/lakehouse-memory/actions/workflows/ci.yml)

**Unity Catalog-native episodic, semantic, and working memory for AI agents on Databricks.**

---

## The thesis

Memory is the missing Databricks layer. The standard workaround is a sidecar
vector database with its own governance, access control, and lineage — a system
you can't ship.

Memory belongs in Unity Catalog, where your data already lives.

`lakehouse-memory` gives AI agents on Databricks three first-class memory
primitives — **episodic**, **semantic**, and **working** — backed by Unity
Catalog tables and Databricks Vector Search.

---

## Install

```bash
pip install --pre lakehouse-memory
```

> The `--pre` flag is required while the package is in pre-release. Once `0.1.0`
> ships, `pip install lakehouse-memory` will work without the flag.

For LangChain integration:

```bash
pip install --pre "lakehouse-memory[langchain]"
```

---

## Next steps

- **[Concepts](concepts.md)** — understand episodic, semantic, and working memory
- **[Quickstart](quickstart.md)** — bootstrap the DAB starter or wire up manually
- **[API Reference](api.md)** — full API docs auto-generated from source
- **[Production Gaps](production-gaps.md)** — what's intentionally left out of OSS
