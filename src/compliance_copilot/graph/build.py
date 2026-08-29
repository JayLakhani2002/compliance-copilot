# src/compliance_copilot/graph/build.py — assembles the two nodes into the
# LangGraph `StateGraph` (docs/ARCHITECTURE.md §4: today's graph is just
# `retrieve -> answer`; the router/critic/interrupt nodes shown in that
# diagram are later features that get added as nodes+edges here, not a
# restructure).
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from compliance_copilot.graph.nodes import (
    answer_node,
    guard_in_node,
    guard_out_node,
    refuse_node,
    retrieve_node,
)
from compliance_copilot.graph.state import AnswerSchema, CitationError, GraphContext, GraphState
from compliance_copilot.settings import settings

# 2 = one retry (the initial attempt + one self-correction) — ADR-0015. One
# place, read by `route_after_answer` only; `answer_node` doesn't need to
# know the cap, it just increments `attempts` each time it runs.
MAX_ATTEMPTS = 2


def route_after_guard(state: GraphState) -> str:
    """Conditional edge after `guard_in` — mirrors `route_after_answer`
    below: a plain state check reading one `GraphState` key, no `runtime`
    param needed (same precedent)."""
    return "refuse" if state["guard"].flagged else "retrieve"


def route_after_answer(state: GraphState) -> str:
    """Conditional edge after `answer` — a plain state check, no `runtime`
    param needed (routing only reads two `GraphState` keys; `path` callables
    passed to `add_conditional_edges` support a `runtime` param the same way
    nodes do, via LangGraph's `RunnableCallable` wrapping, but this routing
    decision doesn't need one). Success -> `guard_out` (ADR-0021: the final
    output-side gate runs on every path, including a good answer — it's no
    longer a direct route to END); a citation failure with a retry left ->
    back to `answer` with the failed draft now in state; out of retries ->
    `fail` (which raises, never reaching `guard_out`)."""
    if state.get("answer") is not None:
        return "guard_out"
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "answer"
    return "fail"


def fail_node(state: GraphState) -> dict:
    """Reached only after MAX_ATTEMPTS failed citation checks. Re-raises so
    `ask()`/`cli.py` see the same hard `CitationError` as before Day 7 —
    the retry loop gives the model one extra chance, it doesn't soften the
    "guard blocks, never swaps" rule ADR-0014 already established."""
    raise CitationError(state["citation_error"])


def _mcp_connection() -> dict[str, Any]:
    """The stdio connection config for `MultiServerMCPClient` (ADR-0007's
    Day-17 amendment) — spawns `python -m compliance_copilot.mcp_server` as
    a subprocess via `uv run --frozen` (never re-resolves/re-locks the
    environment on every graph invocation — the lockfile is already
    committed, so this is a pure perf/determinism win).

    `env=dict(os.environ)`, not a curated allowlist: the stdio SDK only
    inherits a small OS-dependent safe subset by default (verified in the
    installed `langchain_mcp_adapters.sessions.StdioConnection` docstring),
    which would drop `DATABASE_URL`/`OPENAI_API_KEY`/`PATH`/`uv` itself —
    the same full-passthrough-plus-override pattern
    `tests/test_mcp_server_integration.py`'s stdio subprocess test already
    uses, reused here rather than re-inventing a second allowlist."""
    return {
        "copilot": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "--frozen", "python", "-m", "compliance_copilot.mcp_server"],
            "env": dict(os.environ),
            "cwd": os.getcwd(),
        }
    }


async def make_mcp_tools() -> dict[str, Any] | None:
    """Spawns the MCP server subprocess and loads its tools as LangChain
    `BaseTool`s, keyed by name for `GraphContext.tools` (state.py) — the
    context factory ADR-0007's Day-17 amendment calls for. Per-call
    sessions (`MultiServerMCPClient`'s documented default, lesson 17): each
    tool's own `.ainvoke()` opens and tears down its own subprocess+session,
    so building the tool LIST once here (cheap — no session is held open by
    this call) and reusing it across requests is safe; a persistent shared
    session is the upgrade to reach for once connection-setup latency
    actually measures as a problem, not before.

    `None` when `settings.mcp_enabled=False` — the "how to reverse" lever
    (settings.py) for an MCP-outage incident. NOT a fallback to direct
    retrieval: `retrieve_node` still raises `ToolCallError` the moment a
    real question reaches it with no tools (the lesson's fail-loud rule) —
    this just skips ever spawning a doomed subprocess first."""
    if not settings.mcp_enabled:
        return None
    client = MultiServerMCPClient(_mcp_connection())
    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}


@lru_cache(maxsize=1)
def build_graph():
    """Builds and compiles the graph once. Safe to cache module-wide (no
    per-request objects live on the compiled graph itself — those arrive via
    `context=` at `.invoke()` time, see `state.py`'s `GraphContext`).

    No checkpointer is passed to `.compile()`: this feature has no pause/
    resume step yet (ADR-0001 flags a Postgres checkpointer as needed once
    `interrupt()`-based human review lands, not before)."""
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("guard_in", guard_in_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("fail", fail_node)
    builder.add_node("guard_out", guard_out_node)
    builder.add_edge(START, "guard_in")
    builder.add_conditional_edges(
        "guard_in", route_after_guard, {"retrieve": "retrieve", "refuse": "refuse"}
    )
    # ADR-0021: `refuse` no longer goes straight to END — `guard_out` is the
    # final gate on EVERY path, a `guard_in` refusal included.
    builder.add_edge("refuse", "guard_out")
    builder.add_edge("retrieve", "answer")
    builder.add_conditional_edges(
        "answer", route_after_answer, {"answer": "answer", "fail": "fail", "guard_out": "guard_out"}
    )
    # `fail_node` always raises (never returns), so this edge is unreachable
    # in practice — kept for the same reason it always was: an explicit
    # graph shape, not a dangling node.
    builder.add_edge("fail", END)
    builder.add_edge("guard_out", END)
    return builder.compile()


async def ask(
    question: str,
    *,
    session: Session,
    embeddings: Embeddings,
    llm: Any,
    classifier: Any | None = None,
    tools: dict[str, Any] | None = None,
    config: dict | None = None,
) -> AnswerSchema:
    """Convenience entry point: runs the compiled graph for one question and
    returns just the final `AnswerSchema` (rather than making every caller —
    cli.py, tests — reach into the raw state dict).

    `async def`, using `graph.ainvoke(...)` (ADR-0007's Day-17 amendment):
    `retrieve_node` is now `async def` (it awaits an MCP tool call), and a
    real installed-`langgraph` smoke test confirmed the sync `graph.invoke()`
    entrypoint raises `TypeError: No synchronous function provided` the
    moment a run actually reaches an async node — there is no working sync
    path left once `retrieve_node` is async, so every caller converges on
    the async entrypoint (the researcher handoff's simpler alternative to
    maintaining two entrypoints).

    `classifier`: ADR-0019's layer-2 guard, forwarded into `GraphContext`.
    `None` (the default) disables it — `guard_in_node` skips straight past
    the classifier check, same as before this feature existed, so every
    existing caller that doesn't pass one is unaffected.

    `tools`: ADR-0007's MCP tool mapping (`make_mcp_tools()` above), keyed
    by tool name — forwarded into `GraphContext.tools`. `None` (the
    default) means a real question that reaches `retrieve_node` raises
    `ToolCallError` (fail loudly, never a silent fallback) — callers that
    only ever exercise `guard_in`'s refusal path are unaffected either way.

    `config`: optional LangChain `RunnableConfig` dict (ADR-0009 amendment —
    `tracing.run_config()` builds one carrying a Langfuse callback when
    enabled). `None` here just means "no callbacks" — `graph.ainvoke()`
    already defaults to that with no `config=` kwarg at all, so this is
    additive, not a behaviour change for existing callers/tests that don't
    pass one."""
    graph = build_graph()
    context = GraphContext(
        session=session, embeddings=embeddings, llm=llm, classifier=classifier, tools=tools
    )
    state = await graph.ainvoke({"question": question}, context=context, config=config)
    return state["answer"]
