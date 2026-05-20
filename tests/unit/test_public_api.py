"""Lock the public API so accidental exports are caught."""

from __future__ import annotations


def test_public_exports_are_exactly_the_documented_set() -> None:
    import lakehouse_memory as pkg

    expected = {"Memory", "MemoryConfig", "EmbeddingConfig", "Scope", "__version__"}
    actual = set(pkg.__all__) | {"__version__"}
    # `__version__` is dunder-named so it's added explicitly above.
    assert expected == actual, f"expected {expected}, got {actual}"

    # Verify that the intended public names are actually accessible
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"public export {name} not found on package"


def test_version_string_is_set() -> None:
    import lakehouse_memory as pkg

    assert isinstance(pkg.__version__, str)
    assert pkg.__version__
