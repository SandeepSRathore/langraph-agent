"""
LESSON 2 — Making decisions: CONDITIONAL EDGES (and loops)
==========================================================

Lesson 1 was a straight line. Real agents need to DECIDE where to go next.
A conditional edge is a function that looks at the state and returns the
NAME of the next node.

              ┌──────────── "again" ───────────┐
              ▼                                │
    START ──► roll_dice ──► check ─────────────┘
                              │
                              └── "done" ──► END

This is exactly the shape of an AI agent (Lesson 4):
    "call LLM ──► did it ask for a tool? ──► yes: run tool, loop back / no: END"

Run:  uv run lesson_02_conditional_edges.py
"""

import random
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    rolls: list[int]
    attempts: int


def roll_dice(state: State) -> dict:
    roll = random.randint(1, 6)
    print(f"  [roll_dice] rolled a {roll}")
    return {
        "rolls": state["rolls"] + [roll],
        "attempts": state["attempts"] + 1,
    }


# ---------------------------------------------------------------------------
# The ROUTER function. It does NOT change state — it only returns a decision.
# The Literal[...] type hint tells LangGraph (and the diagram) the possible
# destinations.
# ---------------------------------------------------------------------------
def should_continue(state: State) -> Literal["roll_dice", "__end__"]:
    last = state["rolls"][-1]
    if last == 6:
        print("  [router] got a 6 -> stop")
        return END  # END is just the string "__end__"
    if state["attempts"] >= 10:
        print("  [router] too many attempts -> stop")
        return END
    print("  [router] not a 6 -> roll again")
    return "roll_dice"  # going BACK to a previous node = a loop


builder = StateGraph(State)
builder.add_node("roll_dice", roll_dice)

builder.add_edge(START, "roll_dice")
# After "roll_dice", call should_continue() to pick the next node.
builder.add_conditional_edges("roll_dice", should_continue)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"rolls": [], "attempts": 0})
    print(f"\nFinal state: {result}")

    # STREAMING: instead of waiting for the final state, watch each step.
    # stream_mode="updates" yields {node_name: update_it_returned} per step.
    print("\n--- Same graph, streamed step by step ---")
    for step in graph.stream({"rolls": [], "attempts": 0}, stream_mode="updates"):
        print("  step:", step)
