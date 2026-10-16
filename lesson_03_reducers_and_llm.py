"""
LESSON 3 — REDUCERS, MESSAGES, and your first LLM node
======================================================

Problem: in Lesson 2 we wrote  `state["rolls"] + [roll]`  by hand to append.
By default, when a node returns {"key": value}, LangGraph OVERWRITES that key.

A REDUCER changes that rule for a key: "here's how to COMBINE the old value
with the new update". You attach it with Annotated[type, reducer].

For chat, LangGraph ships a ready-made reducer: `add_messages`.
  - It APPENDS new messages to the list (instead of overwriting).
  - If a message has the same id as an existing one, it REPLACES it.

    START ──► chatbot ──► END

Run:  uv run lesson_03_reducers_and_llm.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm import llm


# ---------------------------------------------------------------------------
# Part A: Reducers without any AI — see the difference.
# ---------------------------------------------------------------------------
class DemoState(TypedDict):
    overwritten: list[str]  # no reducer -> last write wins
    appended: Annotated[list[str], operator.add]  # reducer -> old + new


def node_a(state: DemoState) -> dict:
    return {"overwritten": ["from A"], "appended": ["from A"]}


def node_b(state: DemoState) -> dict:
    return {"overwritten": ["from B"], "appended": ["from B"]}


demo = (
    StateGraph(DemoState)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)


# ---------------------------------------------------------------------------
# Part B: A chatbot. State = a list of messages with the add_messages reducer.
#
# Tip: this exact state is so common that LangGraph provides it pre-built:
#     from langgraph.graph import MessagesState
# We write it out here so you can see there's no magic.
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState) -> dict:
    # The node reads the conversation, calls the LLM, and returns ONLY the
    # new reply. add_messages appends it to the history for us.
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}


chat_graph = (
    StateGraph(ChatState)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)


if __name__ == "__main__":
    print("--- Part A: reducers ---")
    print(demo.invoke({"overwritten": [], "appended": []}))
    # -> overwritten: ['from B']            (B replaced A)
    # -> appended:    ['from A', 'from B']  (reducer combined them)

    print("\n--- Part B: chatbot ---")
    result = chat_graph.invoke(
        {
            "messages": [
                SystemMessage("You are a friendly teacher. Answer in 2 sentences."),
                HumanMessage("What is a graph in computer science?"),
            ]
        }
    )
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
