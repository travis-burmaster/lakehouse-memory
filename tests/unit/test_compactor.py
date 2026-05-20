"""Tests for the Compactor protocol stub."""
from __future__ import annotations

from lakehouse_memory.compactor import Compactor, NullCompactor


def test_null_compactor_satisfies_protocol() -> None:
    compactor: Compactor = NullCompactor()
    assert compactor.compact() == {"episodic_collapsed": 0, "semantic_created": 0}


def test_null_compactor_is_noop() -> None:
    """The stub explicitly does nothing — production compactor lives behind the practice."""
    NullCompactor().compact()
