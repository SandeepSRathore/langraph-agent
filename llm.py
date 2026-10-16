"""
The ONE place where we pick an LLM.

LangGraph doesn't care which model you use — a node is just a function, and
whatever happens inside it is up to you. We use langchain's ChatOpenAI wrapper
only because it gives us a standard message format and easy tool-calling.

Want a different provider? Swap this file, e.g.:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-5-5")
"""

import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
