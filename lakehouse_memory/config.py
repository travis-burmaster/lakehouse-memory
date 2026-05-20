"""Configuration models for lakehouse_memory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding endpoint Vector Search uses."""

    model_config = ConfigDict(frozen=True)

    endpoint_name: str = Field(default="databricks-gte-large-en", min_length=1)
    dimensions: int = Field(default=1024, gt=0)


class MemoryConfig(BaseModel):
    """Top-level configuration for a Memory instance."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    catalog: str = Field(min_length=1)
    schema: str = Field(min_length=1)  # type: ignore[assignment]
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    def fqn(self, table: str) -> str:
        """Return the fully-qualified Unity Catalog name for a table."""
        return f"{self.catalog}.{self.schema}.{table}"
