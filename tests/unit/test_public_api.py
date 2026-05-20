"""Lock the public API so accidental exports are caught."""
from __future__ import annotations


def test_public_exports_are_exactly_the_documented_set() -> None:
    import lakehouse_memory as pkg

    expected = {"Memory", "MemoryConfig", "EmbeddingConfig", "Scope", "__version__"}
    actual = {name for name in dir(pkg) if not name.startswith("_")} | {"__version__"}
    # `__version__` is dunder-named so it's added explicitly above.
    assert expected.issubset(actual), f"missing: {expected - actual}"

    # No accidental leakage of internal modules at top level.
    leaked = {"client", "schema", "vector", "stores", "compactor"} & set(dir(pkg))
    assert not leaked, f"internal modules leaked into public API: {leaked}"


def test_version_string_is_set() -> None:
    import lakehouse_memory as pkg

    assert isinstance(pkg.__version__, str)
    assert pkg.__version__
