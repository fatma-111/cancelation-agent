"""
LangGraph graph for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see graph.py.pre_rewrite_backup for the old deterministic
20-node router). The graph is now a standard two-node ReAct-style loop:

    START -> load_config -> agent <-> tools -> END

  - load_config: loads/caches the tenant's client_config.csv +
    dialect_templates.csv (config.get_messages, UNCHANGED function) and
    builds the system prompt (prompts.build_system_prompt) - but only
    once per thread (checked via state.get("templates")), not every turn.
  - agent: calls the LLM, bound to every tool in tools.ALL_TOOLS. The LLM
    decides which tool(s) to call, if any.
  - tools: executes whatever tool call(s) the LLM just requested
    (LangGraph's prebuilt ToolNode - handles InjectedState wiring
    automatically) and appends the results as ToolMessages.
  - Routing: after "agent", if the LLM's latest message contains
    tool_calls, go to "tools"; otherwise the LLM's message IS the reply
    to the user this turn, so the graph ends (control returns to
    whichever caller - main.py's CLI or app.py's HTTP layer - the same
    way for both).

NOTE ON INTERRUPTS: the old graph used explicit interrupt() calls at
wait_for_otp/wait_for_selection/wait_for_confirmation/etc. Those are GONE
- there is no longer a fixed set of "steps that pause". Any time the
agent's response has no tool_calls (e.g. it's asking "which one?" or
"please confirm" or "what's the OTP?"), that's a natural, implicit pause:
the graph reaches END, and the next incoming HumanMessage (via app.py or
main.py) simply gets appended and the graph invoked again - the
checkpointer (MemorySaver, PRESERVED exactly per requirements) restores
the full chat history for that thread_id automatically. This IS how
interrupt/resume is achieved now - it's just no longer a graph-level
primitive, because the "steps" themselves no longer exist as distinct
nodes; the LLM decides turn-by-turn whether it needs another tool call
or needs to ask the user something.
"""

import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import config
import tools
from prompts import build_system_prompt
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# LLM (bound to every tool) - api.py/config.py's OPENAI_* settings,
# UNCHANGED from the old project
# ==========================================================

_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY or "sk-not-configured",
    timeout=config.OPENAI_TIMEOUT_SECONDS,
)

_llm_with_tools = _llm.bind_tools(tools.ALL_TOOLS)


# ==========================================================
# Nodes
# ==========================================================

def load_config(state: AgentState) -> AgentState:
    """Loads client_config.csv/dialect_templates.csv (config.get_messages,
    UNCHANGED) and builds the system prompt EVERY turn.

    CHANGED: this used to skip rebuilding once state["templates"] was
    already set for a thread (a caching optimization). The real-world
    cost of that: any update to prompts.py or the CSVs would NEVER take
    effect for a conversation already in progress - only brand new
    threads picked it up - which is exactly what caused a live
    conversation to keep using stale, pre-fix wording long after
    multiple prompt improvements had already been deployed. Rebuilding
    every turn is cheap (in-memory CSV lookups, already cached at the
    row level by config.py's lru_cache, plus plain string formatting -
    no network calls), so there's no real performance reason to keep
    the old per-thread caching."""

    templates = config.get_messages(state["client_id"])

    state["templates"] = templates
    state["system_prompt"] = build_system_prompt(templates)

    return state


def agent(state: AgentState) -> dict:
    """Calls the LLM with the cached system prompt + full chat history.
    The LLM decides whether to call a tool or reply directly."""

    system_message = SystemMessage(content=state["system_prompt"])
    response = _llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


def route_after_agent(state: AgentState) -> str:
    """If the LLM's latest message requested tool call(s), run them.
    Otherwise its message is the reply for this turn - end the graph."""

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ToolNode automatically injects graph state into any tool parameter
# annotated with InjectedState (see tools.py's `state` params) without
# exposing it to the LLM's function-calling schema.
_tool_node = ToolNode(tools.ALL_TOOLS)


# ==========================================================
# Build graph
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("load_config", load_config)
builder.add_node("agent", agent)
builder.add_node("tools", _tool_node)

builder.set_entry_point("load_config")
builder.add_edge("load_config", "agent")
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
