"""The state schema shared by every node in the graph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """The conversation.

    `add_messages` is a *reducer*: a function LangGraph calls to merge an update into the
    existing value instead of replacing it. Without it, every update (and every new user
    message) would overwrite the whole message list and the chatbot would forget the
    conversation. See the "Why reducers?" section of the README.

    `MessagesState` from `langgraph.graph` is the built-in shortcut for exactly this field;
    it is spelled out here so the reducer is visible while learning.
    """

    messages: Annotated[list[AnyMessage], add_messages]
