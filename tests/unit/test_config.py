"""Tests for configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lakehouse_memory.config import EmbeddingConfig, MemoryConfig


def test_embedding_config_defaults_match_databricks_default() -> None:
    cfg = EmbeddingConfig()
    assert cfg.endpoint_name == "databricks-gte-large-en"
    assert cfg.dimensions == 1024


def test_embedding_config_accepts_override() -> None:
    cfg = EmbeddingConfig(endpoint_name="my-endpoint", dimensions=768)
    assert cfg.endpoint_name == "my-endpoint"
    assert cfg.dimensions == 768


def test_memory_config_requires_catalog_and_schema() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig()  # type: ignore[call-arg]


def test_memory_config_rejects_empty_strings() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(catalog="", schema="my_schema")
    with pytest.raises(ValidationError):
        MemoryConfig(catalog="my_catalog", schema="")


def test_memory_config_defaults_embedding() -> None:
    cfg = MemoryConfig(catalog="prod", schema="mem")
    assert cfg.embedding.endpoint_name == "databricks-gte-large-en"


def test_memory_config_accepts_custom_embedding() -> None:
    cfg = MemoryConfig(
        catalog="prod",
        schema="mem",
        embedding=EmbeddingConfig(endpoint_name="my-endpoint", dimensions=512),
    )
    assert cfg.embedding.endpoint_name == "my-endpoint"
    assert cfg.embedding.dimensions == 512


def test_memory_config_fqn_helper() -> None:
    cfg = MemoryConfig(catalog="prod", schema="mem")
    assert cfg.fqn("episodic") == "prod.mem.episodic"
