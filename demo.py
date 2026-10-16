"""
DEMO — watch the LangGraph agent from the lessons run, step by step.

Every lesson shows up here:
  - the graph structure                       (Lesson 1)
  - the route it takes, including the loop    (Lesson 2 + 4)
  - the message history growing               (Lesson 3)
  - memory per thread                          (Lesson 5)

Run:
  uv run demo.py           # scripted scenarios, then a live chat
  uv run demo.py --chat    # skip straight to the live chat
"""

import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from lesson_04_agent_with_tools import builder

graph = builder.compile(checkpointer=InMemorySaver())

# --- tiny color helpers (colors switch off when output isn't a terminal) ---
USE_COLOR = sys.stdout.isatty()


def color(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t): return color(t, "1")
def dim(t): return color(t, "2")
def cyan(t): return color(t, "36")
def green(t): return color(t, "32")
def yellow(t): return color(t, "33")
def magenta(t): return color(t, "35")


GRAPH_PICTURE = """
                 ┌──────────────────────────┐
                 ▼                          │
  START ──► [ agent ] ──tool calls?──yes──► [ tools ]
                 │
                 └──no──► END
"""


def run_turn(text: str, thread_id: str, echo: bool = True) -> None:
    """Send one message and narrate everything the graph does."""
    config = {"configurable": {"thread_id": thread_id}}
    path = ["START"]

    if echo:
        print(f"\n{bold('You')} {dim(f'(thread {thread_id})')}: {text}")

    for step in graph.stream(
        {"messages": [HumanMessage(text)]}, config, stream_mode="updates"
    ):
        for node, update in step.items():
            path.append(node)
            for msg in update["messages"]:
                if msg.type == "ai" and msg.tool_calls:
                    for call in msg.tool_calls:
                        args = ", ".join(f"{k}={v}" for k, v in call["args"].items())
                        print(f"  {cyan('agent')} ─► asks for {bold(call['name'])}({args})")
                elif msg.type == "tool":
                    print(f"  {yellow('tools')} ─► {msg.name} returned {bold(msg.content)}")
                else:
                    print(f"  {cyan('agent')} ─► decides it's done")
                    print(f"\n{bold(green('Agent'))}: {msg.content}")

    path.append("END")
    saved = graph.get_state(config).values["messages"]
    print(dim(f"\n  route : {' → '.join(path)}"))
    print(dim(f"  memory: {len(saved)} messages saved in thread {thread_id}"))


SCENARIOS = [
    (
        "1. No tools needed: agent goes straight to END",
        "1",
        "Hi! My name is Sandeep and I'm learning LangGraph.",
    ),
    (
        "2. Tools + the loop: agent → tools → agent → tools → agent",
        "1",
        "What is (23 * 47) + 12, and what time is it?",
    ),
    (
        "3. Memory: same thread, so it remembers",
        "1",
        "What's my name, and what was the answer to the math question?",
    ),
    (
        "4. New thread = fresh memory",
        "2",
        "What's my name?",
    ),
]


def scripted() -> None:
    for title, thread_id, text in SCENARIOS:
        print("\n" + magenta("━" * 70))
        print(bold(magenta(f"SCENARIO {title}")))
        print(magenta("━" * 70))
        run_turn(text, thread_id)


def live_chat() -> None:
    print("\n" + magenta("━" * 70))
    print(bold(magenta("LIVE CHAT")) + dim("  —  /thread <id> to switch, quit to exit"))
    print(magenta("━" * 70))
    thread_id = "1"
    while True:
        try:
            text = input(f"\n{bold('You')} {dim(f'(thread {thread_id})')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.startswith("/thread"):
            parts = text.split(maxsplit=1)
            thread_id = parts[1] if len(parts) > 1 else "1"
            print(dim(f"  switched to thread {thread_id}"))
            continue
        run_turn(text, thread_id, echo=False)
    print("\nBye!")


if __name__ == "__main__":
    print(bold("LangGraph agent demo"))
    print(GRAPH_PICTURE)
    if "--chat" not in sys.argv:
        scripted()
    live_chat()
