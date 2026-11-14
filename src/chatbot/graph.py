"""The LangGraph workflow itself: START -> call_model -> END."""

from __future__ import annotations

import sqlite3
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot import config
from chatbot.state import ChatState


def get_llm() -> ChatDeepSeek:
    """DeepSeek chat model, configured from the environment."""
    return ChatDeepSeek(
        model=config.model_name(), temperature=0, extra_body={"thinking": {"type": "disabled"}}
    )


def get_checkpointer() -> SqliteSaver:
    """Checkpointer that stores every conversation in the SQLite file.

    `check_same_thread=False` is needed because Streamlit runs each rerun on its own thread.
    """
    connection = sqlite3.connect(config.db_path(), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


def thread_config(thread_id: str) -> RunnableConfig:
    """The `config` LangGraph uses to know which conversation to read and write."""
    return {"configurable": {"thread_id": thread_id}}


def load_history(graph: CompiledStateGraph, thread_id: str) -> list[AnyMessage]:
    """All messages the checkpointer has stored for this thread."""
    return graph.get_state(thread_config(thread_id)).values.get("messages", [])


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Wire the single node between START and END and compile it with persistence."""

    def call_model(state: ChatState) -> dict[str, Any]:
        """The only node: send the conversation to the LLM, return its reply."""
        messages = [SystemMessage(content=config.SYSTEM_PROMPT), *state["messages"]]
        reply = llm.invoke(messages)
        return {"messages": [AIMessage(content=reply.text)]}

    # ty 0.0.104 does not yet match TypedDicts against LangGraph's `StateLike` protocol
    # (`MessagesState` from langgraph fails the same way), so the state type is ignored here.
    builder = StateGraph(ChatState)  # ty: ignore[invalid-argument-type]
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(checkpointer=checkpointer)
