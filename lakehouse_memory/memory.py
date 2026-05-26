"""Memory: the composition root.

Wires the three stores against a shared client, vector index, and scope.
Provides idempotent UC table provisioning and ergonomic scope refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lakehouse_memory.client import DatabricksClient
from lakehouse_memory.config import MemoryConfig
from lakehouse_memory.schema import SchemaProvisioner
from lakehouse_memory.scope import Scope
from lakehouse_memory.stores.episodic import EpisodicStore
from lakehouse_memory.stores.semantic import SemanticStore
from lakehouse_memory.stores.working import WorkingStore
from lakehouse_memory.vector import VectorIndex

if TYPE_CHECKING:
    from lakehouse_memory.adapters.langchain import (
        LakehouseChatHistory,
        LakehouseSemanticRetriever,
    )


class Memory:
    def __init__(
        self,
        config: MemoryConfig,
        client: DatabricksClient,
        index: VectorIndex | None = None,
        *,
        episodic_index: VectorIndex | None = None,
        semantic_index: VectorIndex | None = None,
        scope: Scope | None = None,
    ) -> None:
        if index is not None and (episodic_index is not None or semantic_index is not None):
            raise TypeError(
                "Pass either index= (deprecated) or episodic_index=/semantic_index=, not both."
            )
        if index is not None:
            import warnings

            warnings.warn(
                "Memory(index=...) is deprecated; use episodic_index= and semantic_index=, "
                "or Memory.from_databricks(...). Will be removed in 0.2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            episodic_index = index
            semantic_index = index
        if episodic_index is None or semantic_index is None:
            raise TypeError(
                "Memory requires episodic_index= and semantic_index= (or the deprecated index=)."
            )

        self._config = config
        self._client = client
        self._episodic_index = episodic_index
        self._semantic_index = semantic_index
        self._scope = scope or Scope()
        self._vs_endpoint: str | None = None
        self._workspace_url: str | None = None
        self._access_token: str | None = None

        self.episodic = EpisodicStore(
            client=client,
            index=episodic_index,
            fqn=config.fqn("episodic"),
            scope=self._scope,
        )
        self.semantic = SemanticStore(
            client=client,
            index=semantic_index,
            fqn=config.fqn("semantic"),
            scope=self._scope,
        )
        self.working = WorkingStore(
            client=client,
            fqn=config.fqn("working"),
            scope=self._scope,
        )

    @property
    def scope(self) -> Scope:
        return self._scope

    def provision(
        self,
        *,
        vector_search_endpoint: str | None = None,
        workspace_url: str | None = None,
        access_token: str | None = None,
    ) -> None:
        """Idempotently create the three memory tables and, optionally, the Vector Search indexes.

        When no Vector Search endpoint is available (neither passed nor stashed
        by from_databricks), only the UC tables are provisioned.
        """
        SchemaProvisioner(
            client=self._client,
            catalog=self._config.catalog,
            schema=self._config.schema_name,
        ).apply()

        endpoint = vector_search_endpoint or self._vs_endpoint
        ws = workspace_url or self._workspace_url
        tok = access_token or self._access_token

        if endpoint is not None:
            if not ws or not tok:
                raise ValueError("vector_search_endpoint requires workspace_url and access_token")
            from lakehouse_memory.vector_databricks import ensure_indexes

            ensure_indexes(
                workspace_url=ws,
                access_token=tok,
                endpoint_name=endpoint,
                config=self._config,
            )

    def with_scope(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> Memory:
        """Return a new Memory with scope merged from the given fields."""
        override = Scope(user_id=user_id, session_id=session_id, agent_id=agent_id)
        new = Memory(
            config=self._config,
            client=self._client,
            episodic_index=self._episodic_index,
            semantic_index=self._semantic_index,
            scope=self._scope.merge(override),
        )
        new._vs_endpoint = self._vs_endpoint
        new._workspace_url = self._workspace_url
        new._access_token = self._access_token
        return new

    def as_langchain_chat_history(self, limit: int = 100) -> LakehouseChatHistory:
        """Return a LangChain BaseChatMessageHistory wired to this Memory's episodic store.

        Requires the `[langchain]` optional extra: `pip install lakehouse-memory[langchain]`.
        """
        from lakehouse_memory.adapters.langchain import LakehouseChatHistory

        return LakehouseChatHistory(self, limit=limit)

    def as_langchain_retriever(self, k: int = 5) -> LakehouseSemanticRetriever:
        """Return a LangChain BaseRetriever wired to this Memory's semantic store.

        Requires the `[langchain]` optional extra: `pip install lakehouse-memory[langchain]`.
        """
        from lakehouse_memory.adapters.langchain import LakehouseSemanticRetriever

        return LakehouseSemanticRetriever(memory=self, k=k)
