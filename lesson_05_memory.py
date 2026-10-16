"""
LESSON 5 — MEMORY with CHECKPOINTERS and THREADS
================================================

Problem: each graph.invoke() starts from scratch. Ask "my name is Sandeep",
then "what's my name?" in a second call, and the agent has no idea.

Solution: a CHECKPOINTER. When you compile with one, LangGraph SAVES the
state after every step, keyed by a THREAD ID. Next time you invoke with the
same thread_id, it loads the saved state and your new message is appended
(thanks to the add_messages reducer from Lesson 3).

  thread "1": [hi, I'm Sandeep] -> [what's my name?] -> "Sandeep!"
  thread "2": [what's my name?] -> "I don't know"   (separate conversation)

Notice: we DON'T rebuild the agent. We reuse the exact same blueprint
(builder) from Lesson 4 and only compile it differently. Memory is a
property of how you RUN a graph, not of the graph itself.

Run:  uv run lesson_05_memory.py
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from lesson_04_agent_with_tools import builder

# InMemorySaver keeps checkpoints in RAM (lost when the program exits).
# For real apps swap in a persistent one, e.g. SqliteSaver or PostgresSaver.
graph = builder.compile(checkpointer=InMemorySaver())


def chat(thread_id: str, text: str) -> str:
    # The thread_id goes in the "config", not the state.
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Type a message. Commands: '/thread <id>' to switch conversation,")
    print("'/history' to see saved state, 'quit' to exit.\n")

    thread_id = "1"
    while True:
        text = input(f"[thread {thread_id}] You: ").strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.startswith("/thread"):
            thread_id = text.split(maxsplit=1)[1] if " " in text else "1"
            print(f"  -> switched to thread {thread_id}\n")
            continue
        if text == "/history":
            # get_state() reads the latest checkpoint for a thread.
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            for m in snapshot.values.get("messages", []):
                print(f"  [{m.type}] {m.content or m.tool_calls}")
            print()
            continue

        print(f"Agent: {chat(thread_id, text)}\n")
