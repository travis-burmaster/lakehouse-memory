"""LangChain integration adapters.

Two adapters bridge lakehouse_memory primitives to LangChain interfaces:

- LakehouseChatHistory: BaseChatMessageHistory backed by episodic memory.
  Drops into RunnableWithMessageHistory.
- LakehouseSemanticRetriever: BaseRetriever backed by semantic memory.
  Drops into any retrieval chain.

Both pull scope from the Memory instance they wrap. Use
`memory.with_scope(session_id=...)` upstream of these adapters to scope
chat history per session, etc.

This module imports langchain_core; install with `pip install lakehouse-memory[langchain]`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

if TYPE_CHECKING:
    from lakehouse_memory.memory import Memory
else:
    Memory = object  # type: ignore[assignment]


class LakehouseChatHistory(BaseChatMessageHistory):
    """``BaseChatMessageHistory`` backed by the episodic memory store.

    Bridges ``Memory.episodic`` to the LangChain
    ``BaseChatMessageHistory`` interface so that ``LakehouseChatHistory`` can
    be dropped directly into ``RunnableWithMessageHistory``.

    Each chat turn is persisted as an episodic event with
    ``event_type="chat_message"`` and
    ``payload={"role": "human"|"ai", "content": ...}``.

    Scope is inherited from the ``Memory`` instance supplied at construction
    time.  To isolate history per session, call
    ``memory.with_scope(session_id="new-session-id")`` before constructing
    this object.

    Note:
        ``clear()`` is an intentional no-op.  Episodic memory is append-only
        by design; deleting history is not supported.  To start a fresh
        conversation, change the ``session_id`` via
        ``memory.with_scope(session_id=...)``.
    """

    def __init__(self, memory: Memory, limit: int = 100) -> None:
        """Initialise a ``LakehouseChatHistory``.

        Args:
            memory: A ``Memory`` instance (typically already scoped to a
                session via ``memory.with_scope(session_id=...)``).
            limit: Maximum number of most-recent messages to return when
                ``messages`` is accessed.  Defaults to ``100``.
        """
        self._memory = memory
        self._limit = limit

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        events = self._memory.episodic.recent(limit=self._limit, event_type="chat_message")
        events.reverse()  # recent() returns DESC; LangChain expects chronological
        return [_event_to_message(e) for e in events]

    def add_message(self, message: BaseMessage) -> None:
        role = "human" if isinstance(message, HumanMessage) else "ai"
        self._memory.episodic.write(
            event_type="chat_message",
            payload={"role": role, "content": message.content},
            text=str(message.content),
        )

    def clear(self) -> None:
        """No-op: episodic memory is append-only and does not support deletion.

        For a fresh conversation, derive a new scoped instance via
        ``memory.with_scope(session_id="<new-session-id>")`` and pass it to a
        new ``LakehouseChatHistory``.
        """
        return None


class LakehouseSemanticRetriever(BaseRetriever):
    """``BaseRetriever`` backed by the semantic memory store.

    Bridges ``Memory.semantic`` to the LangChain ``BaseRetriever`` interface
    so that ``LakehouseSemanticRetriever`` can be dropped into any retrieval
    chain or used with ``create_retrieval_chain``.

    Each retrieved fact becomes a ``Document`` whose ``page_content`` is the
    fact text; all other fact columns (source, scope fields, timestamps, etc.)
    are placed in ``metadata`` (``text`` is excluded since it is promoted to
    ``page_content``).

    Scope is inherited from the ``Memory`` instance supplied at construction
    time.

    Attributes:
        memory: The ``Memory`` instance whose semantic store is queried.
            Scope filtering (user / session / agent) is applied automatically
            from ``memory.scope``.
        k: Number of semantically-similar facts to return per query.
            Defaults to ``5``.
    """

    memory: Memory
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        facts = self.memory.semantic.retrieve(query, k=self.k)
        return [
            Document(
                page_content=str(f.get("text", "")),
                metadata={k: v for k, v in f.items() if k != "text"},
            )
            for f in facts
        ]


def _event_to_message(event: dict[str, Any]) -> BaseMessage:
    payload = event.get("payload")
    data: dict[str, Any]
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}
    elif isinstance(payload, dict):
        data = payload
    else:
        data = {}
    role = data.get("role", "human")
    content = data.get("content", "")
    if role == "human":
        return HumanMessage(content=content)
    return AIMessage(content=content)
