"""Integration tests: LangChain adapters work against the real workspace.

LakehouseChatHistory.add_message writes through episodic → Delta Sync; the
messages property reads directly from the Delta table (no sync wait needed
for chat-history round-trip).

LakehouseSemanticRetriever wraps semantic.retrieve which reads through the
semantic Vector Search index; we wait for sync after upsert before retrieval.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from lakehouse_memory import Memory, Scope
from lakehouse_memory.vector_databricks import DatabricksVectorIndex

from .conftest import wait_for_searchable


def _scoped_memory_with_index(
    live_memory,
    table: str,
    workspace_url: str,
    access_token: str,
    vector_search_endpoint: str,
    user_id: str,
):
    """Build a Memory scoped to `user_id` and wired with the index for the given table.

    Each store needs its own column list because episodic uses event_id and semantic
    uses fact_id as the primary key.
    """
    columns_by_table = {
        "episodic": ["event_id", "text", "user_id", "session_id", "agent_id"],
        "semantic": ["fact_id", "text", "user_id", "session_id", "agent_id", "source"],
    }
    index_name = f"{live_memory._config.catalog}.{live_memory._config.schema_name}.{table}_idx"
    index = DatabricksVectorIndex(
        endpoint_name=vector_search_endpoint,
        index_name=index_name,
        workspace_url=workspace_url,
        access_token=access_token,
        columns=columns_by_table[table],
    )
    return Memory(
        config=live_memory._config,
        client=live_memory._client,
        index=index,
        scope=Scope(user_id=user_id, session_id="lc_test"),
    )


def test_chat_history_round_trip(
    live_memory,
    workspace_url,
    access_token,
    vector_search_endpoint,
) -> None:
    """add_message → messages returns the same content (no vector sync needed).

    chat.messages reads from episodic.recent via SQL, not via the vector
    index, so there's no Delta Sync wait here.
    """
    mem = _scoped_memory_with_index(
        live_memory, "episodic", workspace_url, access_token, vector_search_endpoint,
        user_id="u_lc_chat",
    )
    chat = mem.as_langchain_chat_history(limit=10)
    chat.add_message(HumanMessage(content="hello workspace"))
    chat.add_message(AIMessage(content="hi back"))

    msgs = chat.messages
    contents = [m.content for m in msgs]
    assert "hello workspace" in contents
    assert "hi back" in contents


def test_semantic_retriever_returns_documents_after_sync(
    live_memory,
    workspace_url,
    access_token,
    vector_search_endpoint,
) -> None:
    mem = _scoped_memory_with_index(
        live_memory, "semantic", workspace_url, access_token, vector_search_endpoint,
        user_id="u_lc_retriever",
    )
    fact_id = mem.semantic.upsert(fact="Integration tests cover the LC retriever.")
    # Wait for sync via the semantic store's underlying index
    wait_for_searchable(mem.semantic, "integration tests", fact_id)

    retriever = mem.as_langchain_retriever(k=5)
    docs = retriever.invoke("integration tests")
    contents = [d.page_content for d in docs]
    assert any("integration tests" in c.lower() for c in contents)


def test_runnable_with_message_history_end_to_end(
    live_memory,
    workspace_url,
    access_token,
    vector_search_endpoint,
) -> None:
    """A trivial RunnableLambda + LakehouseChatHistory through RunnableWithMessageHistory."""
    mem = _scoped_memory_with_index(
        live_memory, "episodic", workspace_url, access_token, vector_search_endpoint,
        user_id="u_lc_runnable",
    )

    def _echo(inputs):
        # Inputs is {"input": str, "history": list[BaseMessage]}
        return AIMessage(content=f"echo: {inputs['input']}")

    chain = RunnableLambda(_echo)

    def _get_history(session_id):
        return mem.with_scope(session_id=session_id).as_langchain_chat_history()

    with_history = RunnableWithMessageHistory(
        chain,
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "lc_rwmh_session"}}
    response = with_history.invoke({"input": "first turn"}, config=config)
    assert response.content == "echo: first turn"

    # Second turn should see "first turn" in episodic chat history
    history_after = mem.with_scope(session_id="lc_rwmh_session").as_langchain_chat_history()
    contents = [m.content for m in history_after.messages]
    assert "first turn" in contents
