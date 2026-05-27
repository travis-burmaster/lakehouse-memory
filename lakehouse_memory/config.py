"""Configuration models for lakehouse_memory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model used by Databricks Vector Search.

    Specifies which Foundation Model API endpoint produces the embeddings that
    back the episodic and semantic vector indexes, along with the expected
    output dimensionality.

    Attributes:
        endpoint_name: Databricks Foundation Model API endpoint name that
            generates embeddings.  Must match the endpoint used when the Vector
            Search index was created.  Defaults to
            ``"databricks-gte-large-en"``.
        dimensions: Dimensionality of the embedding vectors produced by
            *endpoint_name*.  Must be positive.  Defaults to ``1024``.
    """

    model_config = ConfigDict(frozen=True)

    endpoint_name: str = Field(default="databricks-gte-large-en", min_length=1)
    dimensions: int = Field(default=1024, gt=0)


class MemoryConfig(BaseModel):
    """Top-level configuration for a ``Memory`` instance.

    Holds the Unity Catalog coordinates (catalog + schema) that determine where
    the three memory tables are stored, plus the embedding configuration used
    by Vector Search.  Instances are frozen (immutable) after construction.

    Attributes:
        catalog: Unity Catalog catalog name.  Must be non-empty.
        schema_name: Schema inside *catalog* where the ``episodic``,
            ``semantic``, and ``working`` tables reside.  Must be non-empty.
        embedding: Embedding endpoint configuration used when creating and
            querying Vector Search indexes.  Defaults to
            ``EmbeddingConfig()``.
    """

    model_config = ConfigDict(frozen=True)

    catalog: str = Field(min_length=1)
    schema_name: str = Field(min_length=1)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    def fqn(self, table: str) -> str:
        """Return the fully-qualified Unity Catalog name for a table.

        Args:
            table: Unqualified table name (e.g. ``"episodic"``).

        Returns:
            Three-part identifier ``<catalog>.<schema_name>.<table>`` suitable
            for use in SQL statements and Vector Search index names.
        """
        return f"{self.catalog}.{self.schema_name}.{table}"
