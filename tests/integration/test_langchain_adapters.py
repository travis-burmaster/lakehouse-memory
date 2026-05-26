"""Integration tests: LangChain adapters work against the real workspace.

LakehouseChatHistory.add_message writes through episodic → Delta Sync; the
messages property reads directly from the Delta table (no sync wait needed
for chat-history round-trip).

LakehouseSemanticRetriever wraps semantic.retrieve which reads through the
semantic Vector Search index; we wait for sync after upsert before retrieval.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from lakehouse_memory import Memory

from .conftest import wait_for_searchable


def test_chat_history_round_trip(
    live_memory,
    workspace_url,
    access_token,
    http_path,
    vector_search_endpoint,
    test_catalog,
) -> None:
    """add_message → messages returns the same content (no vector sync needed).

    chat.messages reads from episodic.recent via SQL, not via the vector
    index, so there's no Delta Sync wait here.
    """
    mem = Memory.from_databricks(
        catalog=test_catalog,
        schema_name=live_memory._config.schema_name,
        workspace_url=workspace_url,
        access_token=access_token,
        http_path=http_path,
        vector_search_endpoint=vector_search_endpoint,
    ).with_scope(user_id="u_lc_chat", session_id="lc_test")

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
    http_path,
    vector_search_endpoint,
    test_catalog,
) -> None:
    mem = Memory.from_databricks(
        catalog=test_catalog,
        schema_name=live_memory._config.schema_name,
        workspace_url=workspace_url,
        access_token=access_token,
        http_path=http_path,
        vector_search_endpoint=vector_search_endpoint,
    ).with_scope(user_id="u_lc_retriever")

    fact_id = mem.semantic.upsert(fact="Integration tests cover the LC retriever.")
    mem.semantic.trigger_sync()
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
    http_path,
    vector_search_endpoint,
    test_catalog,
) -> None:
    """A trivial RunnableLambda + LakehouseChatHistory through RunnableWithMessageHistory."""
    mem = Memory.from_databricks(
        catalog=test_catalog,
        schema_name=live_memory._config.schema_name,
        workspace_url=workspace_url,
        access_token=access_token,
        http_path=http_path,
        vector_search_endpoint=vector_search_endpoint,
    ).with_scope(user_id="u_lc_runnable")

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
