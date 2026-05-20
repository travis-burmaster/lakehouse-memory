"""Shared pytest fixtures for the lakehouse-memory test suite."""
from __future__ import annotations

import pytest


@pytest.fixture
def catalog() -> str:
    return "test_catalog"


@pytest.fixture
def schema_name() -> str:
    return "test_schema"
