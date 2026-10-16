"""
LESSON 1 — The three building blocks: STATE, NODES, EDGES
=========================================================

No LLM here on purpose. LangGraph is NOT an "AI library" at its core —
it's a way to run functions as a graph. Once this clicks, the AI part is easy.

    START ──► greet ──► shout ──► END

Concepts:
  1. STATE  : a shared dictionary that flows through the graph.
  2. NODE   : a plain Python function. Receives the state, returns an UPDATE.
  3. EDGE   : "after this node, go to that node".
  4. COMPILE: turns your blueprint into a runnable app.
  5. INVOKE : runs the app with an initial state.

Run:  uv run lesson_01_state_nodes_edges.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE — define the "shape" of the data that moves through the graph.
#    Every node can read any key and update any key.
# ---------------------------------------------------------------------------
class State(TypedDict):
    name: str
    message: str


# ---------------------------------------------------------------------------
# 2. NODES — ordinary functions: (state) -> partial update.
#    KEY IDEA: you return ONLY the keys you want to change.
#    LangGraph merges that update into the state for you.
# ---------------------------------------------------------------------------
def greet(state: State) -> dict:
    print(f"  [greet] received state: {state}")
    return {"message": f"Hello, {state['name']}!"}


def shout(state: State) -> dict:
    print(f"  [shout] received state: {state}")
    return {"message": state["message"].upper()}


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH — a blueprint: register nodes, then wire them with edges.
# ---------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("greet", greet)  # "greet" is the node's name in the graph
builder.add_node("shout", shout)

builder.add_edge(START, "greet")  # START is a special entry point
builder.add_edge("greet", "shout")
builder.add_edge("shout", END)  # END is a special exit point

# ---------------------------------------------------------------------------
# 4. COMPILE — validates the wiring (e.g. no dangling nodes) and gives you
#    a runnable object with .invoke(), .stream(), etc.
# ---------------------------------------------------------------------------
graph = builder.compile()


if __name__ == "__main__":
    # 5. INVOKE — pass the initial state, get the FINAL state back.
    print("Running graph...")
    result = graph.invoke({"name": "Sandeep", "message": ""})
    print("\nFinal state:", result)

    # BONUS: LangGraph can draw your graph. Paste this at https://mermaid.live
    print("\nGraph diagram (Mermaid):")
    print(graph.get_graph().draw_mermaid())
