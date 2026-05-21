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
        index: VectorIndex,
        scope: Scope | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._index = index
        self._scope = scope or Scope()

        self.episodic = EpisodicStore(
            client=client,
            index=index,
            fqn=config.fqn("episodic"),
            scope=self._scope,
        )
        self.semantic = SemanticStore(
            client=client,
            index=index,
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

    def provision(self) -> None:
        """Idempotently create the three memory tables."""
        SchemaProvisioner(
            client=self._client,
            catalog=self._config.catalog,
            schema=self._config.schema_name,
        ).apply()

    def with_scope(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> Memory:
        """Return a new Memory with scope merged from the given fields."""
        override = Scope(user_id=user_id, session_id=session_id, agent_id=agent_id)
        return Memory(
            config=self._config,
            client=self._client,
            index=self._index,
            scope=self._scope.merge(override),
        )

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
