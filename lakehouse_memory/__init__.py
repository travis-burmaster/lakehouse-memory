"""lakehouse-memory: Unity Catalog-native memory for AI agents on Databricks."""
from __future__ import annotations

import sys
from types import ModuleType

from lakehouse_memory.config import EmbeddingConfig, MemoryConfig
from lakehouse_memory.memory import Memory
from lakehouse_memory.scope import Scope

__version__ = "0.1.0a0"

__all__ = ["EmbeddingConfig", "Memory", "MemoryConfig", "Scope", "__version__"]

# Clean up namespace to prevent internal modules from leaking
_current_module = sys.modules[__name__]
_public_names = set(__all__)
for _name in list(vars(_current_module).keys()):
    if _name.startswith("_") or _name in _public_names or _name == "__version__":
        continue
    if isinstance(vars(_current_module)[_name], ModuleType):
        delattr(_current_module, _name)
