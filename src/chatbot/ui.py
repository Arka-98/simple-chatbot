"""Everything Streamlit renders: the sidebar with threads and the chat area."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from chatbot import config, db
from chatbot import graph as workflow


@st.cache_resource
def load_checkpointer() -> SqliteSaver:
    """The SQLite store behind every conversation, built once per server process."""
    db.init_db()
    return workflow.get_checkpointer()


@st.cache_resource
def load_graph() -> CompiledStateGraph:
    """Built once per server process and shared by every rerun and browser tab."""
    return workflow.build_graph(workflow.get_llm(), load_checkpointer())


def new_thread_id() -> str:
    return str(uuid.uuid4())


def render() -> None:
    st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="wide")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()

    st.title("💬 Simple LangGraph Chatbot")
    st.caption("One node between START and END, persisted in SQLite.")

    if config.missing_api_key():
        st.error(
            "`DEEPSEEK_API_KEY` is not set. Copy `.env.example` to `.env`, add your key, "
            "and restart the app."
        )
        st.stop()

    graph = load_graph()
    render_sidebar()
    render_chat(graph, st.session_state.thread_id)


def render_sidebar() -> None:
    """New chat button plus every thread stored in the database."""
    with st.sidebar:
        st.subheader("Conversations")

        if st.button("➕ New chat", use_container_width=True):
            st.session_state.thread_id = new_thread_id()
            st.rerun()

        threads = db.list_threads()
        if not threads:
            st.caption("No conversations yet — say hello!")

        for thread in threads:
            is_active = thread.id == st.session_state.thread_id
            open_column, delete_column = st.columns([0.8, 0.2], vertical_alignment="center")

            label = f"• {thread.title}" if is_active else thread.title
            if open_column.button(
                label,
                key=f"open-{thread.id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.thread_id = thread.id
                st.rerun()

            if delete_column.button("🗑", key=f"delete-{thread.id}", help="Delete conversation"):
                delete_thread(load_checkpointer(), thread.id)
                if is_active:
                    st.session_state.thread_id = new_thread_id()
                st.rerun()


def render_chat(graph: CompiledStateGraph, thread_id: str) -> None:
    for message in workflow.load_history(graph, thread_id):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        st.chat_message(role).write(message.text)

    prompt = st.chat_input("Type a message and press Enter…")
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    db.ensure_thread(thread_id, prompt)

    with st.chat_message("assistant"):
        try:
            st.write_stream(stream_reply(graph, thread_id, prompt))
        except Exception as error:
            st.error(f"Could not reach the model: {error}")
            return

    # Redraw from the database so the transcript, titles and ordering all match what is stored.
    st.rerun()


def stream_reply(graph: CompiledStateGraph, thread_id: str, prompt: str) -> Iterator[str]:
    """Run the graph and yield the assistant's tokens as they arrive."""
    inputs = {"messages": [HumanMessage(content=prompt)]}
    for chunk, _metadata in graph.stream(
        inputs, workflow.thread_config(thread_id), stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            yield chunk.text


def delete_thread(checkpointer: BaseCheckpointSaver, thread_id: str) -> None:
    """Remove the conversation from both stores: LangGraph checkpoints and our threads table."""
    checkpointer.delete_thread(thread_id)
    db.delete_thread(thread_id)
