# Contributing to lakehouse-memory

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the test suite

Unit tests (no Databricks workspace needed):
```bash
pytest tests/unit -v
```

Integration tests (requires a workspace) — arriving in v0.2:
```bash
LAKEHOUSE_MEMORY_INTEGRATION=1 pytest tests/integration -v
```

## Lint, type check

```bash
ruff check .
ruff format --check .
mypy lakehouse_memory
```

## Commit style

Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.

## License

By contributing, you agree your contributions are licensed under Apache 2.0.
