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
    """LangChain chat history backed by `Memory.episodic`.

    Each turn is one episodic event with event_type="chat_message" and
    payload={"role": "human"|"ai", "content": ...}.
    """

    def __init__(self, memory: Memory, limit: int = 100) -> None:
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
        # Episodic is append-only by design — clear() does not delete history.
        # For a fresh session, change session_id via memory.with_scope(session_id=...).
        return None


class LakehouseSemanticRetriever(BaseRetriever):
    """LangChain retriever backed by `Memory.semantic`.

    Returns Documents whose page_content is the fact text and metadata
    carries source + scope columns (with `text` removed since it's the page
    content).
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
