"""Unit tests for LangChain adapters.

These tests use the real Memory class wired to a MagicMock SQL client and a
MockVectorIndex — exercising the adapter logic without touching any real
Databricks resources.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from lakehouse_memory import Memory, MemoryConfig, Scope
from lakehouse_memory.adapters.langchain import (
    LakehouseChatHistory,
    LakehouseSemanticRetriever,
)
from lakehouse_memory.vector import MockVectorIndex


def _make_memory(scope: Scope | None = None) -> tuple[Memory, MagicMock]:
    client = MagicMock()
    client.execute.return_value = []
    mem = Memory(
        config=MemoryConfig(catalog="t", schema_name="t"),
        client=client,
        index=MockVectorIndex(),
        scope=scope or Scope(session_id="s_1"),
    )
    return mem, client


def test_chat_history_add_human_message_writes_episodic_event() -> None:
    mem, client = _make_memory()
    chat = LakehouseChatHistory(mem)
    chat.add_message(HumanMessage(content="hi there"))

    sql, params = client.execute.call_args.args
    assert "INSERT INTO t.t.episodic" in sql
    assert params["event_type"] == "chat_message"
    payload = json.loads(params["payload"])
    assert payload == {"role": "human", "content": "hi there"}
    assert params["text"] == "hi there"


def test_chat_history_add_ai_message_writes_episodic_event() -> None:
    mem, client = _make_memory()
    chat = LakehouseChatHistory(mem)
    chat.add_message(AIMessage(content="hello back"))

    _sql, params = client.execute.call_args.args
    payload = json.loads(params["payload"])
    assert payload == {"role": "ai", "content": "hello back"}


def test_chat_history_messages_returns_chronological_messages() -> None:
    mem, client = _make_memory()
    # episodic.recent returns DESC order; adapter must reverse to chronological
    client.execute.return_value = [
        {"event_id": "2", "payload": json.dumps({"role": "ai", "content": "hello back"})},
        {"event_id": "1", "payload": json.dumps({"role": "human", "content": "hi there"})},
    ]
    chat = LakehouseChatHistory(mem)
    msgs = chat.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hi there"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "hello back"


def test_chat_history_messages_filters_to_chat_message_event_type() -> None:
    mem, client = _make_memory()
    client.execute.return_value = []
    _ = LakehouseChatHistory(mem).messages

    sql, params = client.execute.call_args.args
    assert "event_type = :event_type" in sql
    assert params["event_type"] == "chat_message"


def test_chat_history_clear_is_documented_noop() -> None:
    mem, client = _make_memory()
    chat = LakehouseChatHistory(mem)
    chat.clear()
    # No DELETE issued; clear is an intentional no-op for append-only episodic
    assert client.execute.call_count == 0


def test_chat_history_handles_dict_payload_when_recent_returns_dict() -> None:
    """Tolerate payload coming back as a dict rather than a JSON string."""
    mem, client = _make_memory()
    client.execute.return_value = [
        {"event_id": "1", "payload": {"role": "human", "content": "raw dict"}},
    ]
    msgs = LakehouseChatHistory(mem).messages
    assert msgs[0].content == "raw dict"


def test_semantic_retriever_returns_documents_from_semantic_retrieve() -> None:
    mem, _ = _make_memory(scope=Scope(user_id="u_1"))
    # Seed the mock index so semantic.retrieve returns results
    mem.semantic._index.upsert(
        [
            {"id": "1", "text": "user prefers SQL", "user_id": "u_1", "source": "conv:s_1"},
        ]
    )
    retriever = LakehouseSemanticRetriever(memory=mem, k=5)
    docs = retriever.invoke("prefers")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "user prefers SQL"
    assert docs[0].metadata["id"] == "1"
    assert docs[0].metadata["source"] == "conv:s_1"
    assert "text" not in docs[0].metadata  # text moved to page_content


def test_semantic_retriever_respects_k_parameter() -> None:
    mem, _ = _make_memory(scope=Scope(user_id="u_1"))
    mem.semantic._index.upsert(
        [{"id": str(i), "text": "topic", "user_id": "u_1"} for i in range(10)]
    )
    docs = LakehouseSemanticRetriever(memory=mem, k=3).invoke("topic")
    assert len(docs) == 3


def test_memory_as_langchain_chat_history_returns_adapter() -> None:
    mem, _ = _make_memory()
    chat = mem.as_langchain_chat_history(limit=50)
    assert isinstance(chat, LakehouseChatHistory)
    assert chat._limit == 50


def test_memory_as_langchain_retriever_returns_adapter() -> None:
    mem, _ = _make_memory()
    retriever = mem.as_langchain_retriever(k=7)
    assert isinstance(retriever, LakehouseSemanticRetriever)
    assert retriever.k == 7
    assert retriever.memory is mem
