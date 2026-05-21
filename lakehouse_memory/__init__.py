"""lakehouse-memory: Unity Catalog-native memory for AI agents on Databricks."""

from __future__ import annotations

from lakehouse_memory.config import EmbeddingConfig, MemoryConfig
from lakehouse_memory.memory import Memory
from lakehouse_memory.scope import Scope

__version__ = "0.1.0b1"

__all__ = ["EmbeddingConfig", "Memory", "MemoryConfig", "Scope", "__version__"]
