"""
LESSON 4 — THE AGENT: an LLM that uses TOOLS in a LOOP
======================================================

An "agent" is just Lessons 1-3 combined:
  - STATE       = the message history                (Lesson 3)
  - NODES       = "agent" (calls LLM) + "tools" (runs Python functions)
  - CONDITIONAL = "did the LLM ask for a tool?"      (Lesson 2)
  - LOOP        = after running tools, go back to the LLM

                   ┌──────────────────────────┐
                   ▼                          │
    START ──► agent ──► (tool calls?) ──yes──► tools
                   │
                   └──no──► END

How tool calling works:
  1. We tell the LLM which tools exist (llm.bind_tools).
  2. The LLM doesn't run anything. It replies with a REQUEST like
     "please call multiply(a=23, b=47)"  -> stored in reply.tool_calls
  3. OUR "tools" node runs the real function and appends the result
     as a ToolMessage.
  4. Back to the LLM, which now sees the result and either asks for more
     tools or writes the final answer.

Run:  uv run lesson_04_agent_with_tools.py
"""

from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm import llm


# ---------------------------------------------------------------------------
# 1. TOOLS — normal Python functions. The @tool decorator reads the name,
#    type hints and docstring and turns them into a description the LLM sees.
#    So the docstring matters: it's how the LLM knows WHEN to use the tool.
# ---------------------------------------------------------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [add, multiply, get_current_time]
tools_by_name = {t.name: t for t in tools}

# Give the LLM the "menu" of tools it may request.
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 2. STATE
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = SystemMessage(
    "You are a helpful assistant. Use the tools for every arithmetic step "
    "(never compute in your head) and when asked about the time."
)


# ---------------------------------------------------------------------------
# 3. NODES
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """The 'brain': ask the LLM what to do next."""
    reply = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [reply]}


def tools_node(state: AgentState) -> dict:
    """The 'hands': run every tool the LLM asked for."""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        output = tools_by_name[call["name"]].invoke(call["args"])
        # tool_call_id links this result to the request that asked for it.
        results.append(
            ToolMessage(content=str(output), name=call["name"], tool_call_id=call["id"])
        )
    return {"messages": results}

# (LangGraph has a ready-made version of this node:
#      from langgraph.prebuilt import ToolNode;  ToolNode(tools)
#  We wrote our own so you can see it's only a few lines.)


# ---------------------------------------------------------------------------
# 4. ROUTER — the decision that makes it an agent instead of a pipeline.
# ---------------------------------------------------------------------------
def should_use_tools(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. WIRE IT UP
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_use_tools)
builder.add_edge("tools", "agent")  # <- the loop: results go back to the brain

graph = builder.compile()


if __name__ == "__main__":
    question = "What time is it right now? Also, what is 23 * 47 + 12?"
    print(f"You: {question}\n")

    # Stream so we can watch the agent think -> act -> think -> answer.
    for step in graph.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="updates"
    ):
        for node_name, update in step.items():
            for msg in update["messages"]:
                if msg.type == "ai" and msg.tool_calls:
                    print(f"[{node_name}] wants tools: {[c['name'] for c in msg.tool_calls]}")
                elif msg.type == "tool":
                    print(f"[{node_name}] result: {msg.content}")
                else:
                    print(f"\n[{node_name}] FINAL ANSWER: {msg.content}")
