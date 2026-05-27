"""Memory: the composition root.

Wires the three stores against a shared client, per-store vector indexes, and scope.
Provides idempotent UC table provisioning and ergonomic scope refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lakehouse_memory.client import DatabricksClient, SqlConnectorClient
from lakehouse_memory.config import EmbeddingConfig, MemoryConfig
from lakehouse_memory.schema import SchemaProvisioner
from lakehouse_memory.scope import Scope
from lakehouse_memory.stores.episodic import EpisodicStore
from lakehouse_memory.stores.semantic import SemanticStore
from lakehouse_memory.stores.working import WorkingStore
from lakehouse_memory.vector import VectorIndex
from lakehouse_memory.vector_databricks import DatabricksVectorIndex, ensure_indexes

if TYPE_CHECKING:
    from lakehouse_memory.adapters.langchain import (
        LakehouseChatHistory,
        LakehouseSemanticRetriever,
    )


class Memory:
    """Composition root that wires episodic, semantic, and working stores.

    Each ``Memory`` instance holds a SQL client, one vector index per store,
    and a ``Scope`` that filters all reads and tags all writes to a specific
    user / session / agent combination.

    The canonical way to create a ``Memory`` wired to real Databricks resources is
    ``Memory.from_databricks``.  After construction, call ``provision()`` once to
    create the underlying Unity Catalog tables and Vector Search indexes.

    Example::

        mem = Memory.from_databricks(
            catalog="my_catalog",
            schema_name="agent_memory",
            workspace_url="https://my-workspace.azuredatabricks.net",
            access_token="dapi...",
            http_path="/sql/1.0/warehouses/abc123",
            vector_search_endpoint="my_vs_endpoint",
        )
        mem.provision()
        scoped = mem.with_scope(user_id="u1", session_id="s1")
        scoped.episodic.write(event_type="chat_message", payload={}, text="Hello")
    """

    def __init__(
        self,
        config: MemoryConfig,
        client: DatabricksClient,
        *,
        episodic_index: VectorIndex,
        semantic_index: VectorIndex,
        scope: Scope | None = None,
    ) -> None:
        """Initialise Memory with explicit collaborators.

        Prefer ``Memory.from_databricks`` for production use.  This constructor
        is useful in tests where you supply stub clients and no-op indexes.

        Args:
            config: Catalog/schema/embedding settings for this Memory instance.
            client: SQL client used by all three stores for DDL and DML.
            episodic_index: Vector index used by the episodic store for
                similarity search.  Pass a no-op ``VectorIndex`` to skip
                vector search for episodic events.
            semantic_index: Vector index used by the semantic store for
                similarity search.  Pass a no-op ``VectorIndex`` to skip
                vector search for facts.
            scope: Optional identity scope applied to every read and write.
                Defaults to an empty ``Scope()`` (no filtering).
        """
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

    @classmethod
    def from_databricks(
        cls,
        *,
        catalog: str,
        schema_name: str,
        workspace_url: str,
        access_token: str,
        http_path: str,
        vector_search_endpoint: str,
        scope: Scope | None = None,
        embedding: EmbeddingConfig | None = None,
    ) -> Memory:
        """Build a Memory wired to real Databricks resources.

        Constructs a ``SqlConnectorClient`` and two Delta Sync-backed
        ``DatabricksVectorIndex`` objects (one for episodic events, one for
        semantic facts) and stashes the VS credentials so that a later call to
        ``provision()`` can create the indexes without repeating them.

        Does **not** provision — call ``mem.provision()`` after construction to
        idempotently create the Unity Catalog tables and Vector Search indexes.

        Args:
            catalog: Unity Catalog catalog name (e.g. ``"my_catalog"``).
            schema_name: Schema inside *catalog* where memory tables live
                (e.g. ``"agent_memory"``).
            workspace_url: Full Databricks workspace URL, including scheme
                (e.g. ``"https://my-workspace.azuredatabricks.net"``).
            access_token: Databricks personal-access token or service-principal
                secret used for both SQL Warehouse and Vector Search API calls.
            http_path: SQL Warehouse HTTP path
                (e.g. ``"/sql/1.0/warehouses/abc123"``).
            vector_search_endpoint: Name of the existing Databricks Vector
                Search endpoint to back both indexes.
            scope: Optional identity scope to pre-apply to every store.
                Defaults to an empty ``Scope()`` (no filtering).
            embedding: Optional embedding endpoint configuration.  Defaults to
                ``EmbeddingConfig()`` (``databricks-gte-large-en``, 1024 dims).

        Returns:
            A fully-wired ``Memory`` instance.  Call ``provision()`` before
            reading or writing to ensure the underlying tables and indexes exist.
        """
        config = MemoryConfig(
            catalog=catalog,
            schema_name=schema_name,
            embedding=embedding or EmbeddingConfig(),
        )
        hostname = workspace_url.replace("https://", "").replace("http://", "").rstrip("/")
        client = SqlConnectorClient(
            server_hostname=hostname,
            http_path=http_path,
            access_token=access_token,
        )
        episodic_index = DatabricksVectorIndex(
            endpoint_name=vector_search_endpoint,
            index_name=f"{catalog}.{schema_name}.episodic_idx",
            workspace_url=workspace_url,
            access_token=access_token,
            columns=["event_id", "text", "user_id", "session_id", "agent_id"],
        )
        semantic_index = DatabricksVectorIndex(
            endpoint_name=vector_search_endpoint,
            index_name=f"{catalog}.{schema_name}.semantic_idx",
            workspace_url=workspace_url,
            access_token=access_token,
            columns=["fact_id", "text", "user_id", "session_id", "agent_id"],
        )
        mem = cls(
            config=config,
            client=client,
            episodic_index=episodic_index,
            semantic_index=semantic_index,
            scope=scope,
        )
        mem._vs_endpoint = vector_search_endpoint
        mem._workspace_url = workspace_url
        mem._access_token = access_token
        return mem

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
        """Idempotently create the UC schema + tables and, optionally, the Vector Search indexes.

        Always creates the Unity Catalog schema (if absent) and the three memory
        tables (``episodic``, ``semantic``, ``working``).  When a Vector Search
        endpoint is available — either supplied here or stashed by
        ``from_databricks`` — also creates the two Delta Sync indexes.

        Safe to call multiple times; existing tables and indexes are left
        untouched.

        Args:
            vector_search_endpoint: Name of the Databricks Vector Search endpoint
                to use when creating indexes.  Falls back to the value stashed by
                ``from_databricks``, if any.  Pass ``None`` (and provide no
                stashed value) to skip index creation entirely.
            workspace_url: Workspace URL needed for Vector Search API calls.
                Falls back to the value stashed by ``from_databricks``.
            access_token: Databricks PAT or service-principal secret for Vector
                Search API calls.  Falls back to the value stashed by
                ``from_databricks``.

        Raises:
            ValueError: If *vector_search_endpoint* is resolved but
                *workspace_url* or *access_token* cannot be determined.
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
        """Return a new Memory with scope fields merged from the given arguments.

        Any field you pass overrides the corresponding field on the current
        scope; fields you omit (or pass as ``None``) are inherited unchanged.
        The new instance shares the same SQL client and vector indexes as the
        original — no new connections are opened.  Stashed VS credentials
        (workspace_url, access_token, endpoint) are forwarded so that
        ``provision()`` may still be called on the derived instance.

        Args:
            user_id: Override the ``user_id`` dimension of the scope.
            session_id: Override the ``session_id`` dimension of the scope.
            agent_id: Override the ``agent_id`` dimension of the scope.

        Returns:
            A new ``Memory`` instance with the merged scope applied to all
            three stores.
        """
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
        """Return a LangChain ``BaseChatMessageHistory`` wired to the episodic store.

        Requires the ``[langchain]`` optional extra::

            pip install lakehouse-memory[langchain]

        Args:
            limit: Maximum number of recent chat messages to return when
                ``messages`` is accessed.  Defaults to ``100``.

        Returns:
            A ``LakehouseChatHistory`` instance scoped to this Memory's scope.
        """
        from lakehouse_memory.adapters.langchain import LakehouseChatHistory

        return LakehouseChatHistory(self, limit=limit)

    def as_langchain_retriever(self, k: int = 5) -> LakehouseSemanticRetriever:
        """Return a LangChain ``BaseRetriever`` wired to the semantic store.

        Requires the ``[langchain]`` optional extra::

            pip install lakehouse-memory[langchain]

        Args:
            k: Number of semantically-similar facts to return per query.
                Defaults to ``5``.

        Returns:
            A ``LakehouseSemanticRetriever`` instance scoped to this Memory's
            scope.
        """
        from lakehouse_memory.adapters.langchain import LakehouseSemanticRetriever

        return LakehouseSemanticRetriever(memory=self, k=k)
