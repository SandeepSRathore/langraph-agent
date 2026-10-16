# Learn LangGraph by building an agent

Five small files. Each one adds **one** concept. Read the comments at the top
of each file, run it, then tweak something and run it again.

## Setup

```bash
uv sync                       # installs langgraph + langchain-openai
export OPENAI_API_KEY=...     # already set on this machine
```

Run any lesson with `uv run <file>`.

## Quick demo

```bash
uv run demo.py           # 4 scripted scenarios, then a live chat
uv run demo.py --chat    # straight to the live chat
```

The demo narrates every step: which node ran, which tools were requested,
the route taken (`START → agent → tools → agent → END`) and how many
messages are saved in memory for the current thread.

## The learning path

| # | File | Concept you learn | Uses LLM? |
|---|------|-------------------|-----------|
| 1 | `lesson_01_state_nodes_edges.py` | **State**, **Nodes**, **Edges**, `compile()`, `invoke()` | No |
| 2 | `lesson_02_conditional_edges.py` | **Conditional edges** (routing), **loops**, `stream()` | No |
| 3 | `lesson_03_reducers_and_llm.py` | **Reducers** (`add_messages`), message state, an LLM node | Yes |
| 4 | `lesson_04_agent_with_tools.py` | **Tools** + the **agent loop** (ReAct) — the actual agent | Yes |
| 5 | `lesson_05_memory.py` | **Checkpointers**, **threads**, interactive chat with memory | Yes |

`llm.py` is the one place the model is chosen. Swap it to change providers.

## The whole framework in 6 sentences

1. **State** is a shared dict (a `TypedDict`) that flows through the graph.
2. A **node** is a plain function: it gets the state and returns *only the keys it changes*.
3. An **edge** says what runs next; a **conditional edge** is a function that *returns the name* of what runs next.
4. A **reducer** decides how an update merges into state (default: overwrite; `add_messages`: append).
5. You **compile** the blueprint (`StateGraph`) into a runnable app, then `invoke` / `stream` it.
6. A **checkpointer** saves state after every step per `thread_id`, which gives you memory.

## The agent you end up with (Lesson 4)

```
                ┌───────────────────────┐
                ▼                       │
START ──► agent (LLM) ──tool calls?──► tools
                │
                └── no ──► END
```

The LLM never runs code itself. It *requests* tool calls; the `tools` node runs
them and feeds results back; the loop repeats until the LLM answers in plain text.

## Exercises (try these to check your understanding)

1. **Lesson 1:** add a third node `add_emoji` between `shout` and `END`.
2. **Lesson 2:** stop when the *total* of all rolls passes 15 instead of on a 6.
3. **Lesson 3:** remove `add_messages` from `ChatState` — what breaks, and why?
4. **Lesson 4:** add a `@tool def word_count(text: str) -> int` and ask the agent to use it.
5. **Lesson 4:** replace `tools_node` with `langgraph.prebuilt.ToolNode(tools)` — it should behave identically.
6. **Lesson 5:** in the chat, tell it your name, type `/thread 2`, ask again, then `/thread 1`.

## Where to go next

- **Human-in-the-loop:** `interrupt()` to pause the graph and ask a human before a tool runs.
- **Persistent memory:** `SqliteSaver` instead of `InMemorySaver`.
- **Subgraphs / multi-agent:** a compiled graph can itself be used as a node.
- **Shortcut:** `langgraph.prebuilt.create_react_agent` builds Lesson 4 in one line —
  now you know what it does under the hood.
