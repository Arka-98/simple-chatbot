"""The reducer is what makes a conversation remember itself: prove that here."""

from __future__ import annotations

import itertools
import sqlite3
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from chatbot import db
from chatbot.config import SYSTEM_PROMPT
from chatbot.graph import build_graph, load_history, thread_config
from chatbot.ui import delete_thread, stream_reply


class ScriptedChatModel(BaseChatModel):
    """Fake chat model: records what it was asked, answers with a counter."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        reply = AIMessage(content=f"reply {len(self.calls)}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_messages_accumulate_across_invocations_on_one_thread() -> None:
    llm = ScriptedChatModel()
    graph = build_graph(llm, InMemorySaver())
    run_config = thread_config("thread-1")

    graph.invoke({"messages": [HumanMessage(content="hi")]}, run_config)
    graph.invoke({"messages": [HumanMessage(content="and again")]}, run_config)

    assert [message.text for message in load_history(graph, "thread-1")] == [
        "hi",
        "reply 1",
        "and again",
        "reply 2",
    ]
    assert [message.text for message in llm.calls[1]] == [
        SYSTEM_PROMPT,
        "hi",
        "reply 1",
        "and again",
    ]


def test_stream_reply_streams_tokens_and_persists_the_turn(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "chat.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    llm = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    graph = build_graph(llm, checkpointer)

    chunks = list(stream_reply(graph, "thread-1", "hi"))

    assert len(chunks) > 1
    assert "".join(chunks) == "hello world"
    assert [message.text for message in load_history(graph, "thread-1")] == ["hi", "hello world"]


def test_delete_thread_clears_checkpoints_and_the_thread_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "app.db"))
    db.init_db()
    checkpointer = SqliteSaver(sqlite3.connect(tmp_path / "app.db", check_same_thread=False))
    checkpointer.setup()
    llm = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="yo")]))
    graph = build_graph(llm, checkpointer)

    db.ensure_thread("t1", "hello")
    graph.invoke({"messages": [HumanMessage(content="hello")]}, thread_config("t1"))

    delete_thread(checkpointer, "t1")

    assert load_history(graph, "t1") == []
    assert db.list_threads() == []


def test_threads_keep_separate_histories() -> None:
    llm = ScriptedChatModel()
    graph = build_graph(llm, InMemorySaver())

    graph.invoke({"messages": [HumanMessage(content="thread A")]}, thread_config("a"))
    graph.invoke({"messages": [HumanMessage(content="thread B")]}, thread_config("b"))

    assert [message.text for message in load_history(graph, "a")] == ["thread A", "reply 1"]
    assert [message.text for message in load_history(graph, "b")] == ["thread B", "reply 2"]
