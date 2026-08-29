# src/compliance_copilot/graph/build.py — assembles the graph's nodes into
# the LangGraph `StateGraph` (docs/ARCHITECTURE.md §4). ADR-0023 adds
# `router` (before `retrieve`) and `critic` (before `guard_out`); ADR-0025
# adds `hitl` between `critic` and `guard_out` — an `interrupt()`-based pause
# on a low-confidence critic score. Same incremental pattern every prior
# guard/gate feature (ADR-0018/0019/0020/0021) added its own node with — no
# restructure.
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
    critic_node,
    guard_in_node,
    guard_out_node,
    hitl_node,
    refuse_node,
    retrieve_node,
    router_node,
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
    param needed (same precedent). ADR-0023: a clean question now goes to
    `router`, not straight to `retrieve` — `router_node` decides the search
    scope (or no-ops when disabled) before retrieval runs."""
    return "refuse" if state["guard"].flagged else "router"


def route_after_router(state: GraphState) -> str:
    """Conditional edge after `router` (ADR-0023) — `out_of_scope` short-
    circuits straight to `refuse` (no retrieval/answer spend on a question
    about neither regulation); every other label, INCLUDING an absent
    `state["router"]` key (router disabled, or a fail-open outage already
    normalised to `both` by `router_node`), goes to `retrieve`. Two
    different sources (`route_after_guard` above, this function) both map
    into the same `refuse` node — confirmed safe: `add_conditional_edges`
    just adds graph edges, no exclusivity constraint between them (verified
    against installed `langgraph/graph/state.py`, and smoke-tested in
    tests/test_graph.py)."""
    router_verdict = state.get("router")
    if router_verdict is not None and router_verdict.regulation == "out_of_scope":
        return "refuse"
    return "retrieve"


def route_after_answer(state: GraphState) -> str:
    """Conditional edge after `answer` — a plain state check, no `runtime`
    param needed (routing only reads two `GraphState` keys; `path` callables
    passed to `add_conditional_edges` support a `runtime` param the same way
    nodes do, via LangGraph's `RunnableCallable` wrapping, but this routing
    decision doesn't need one). Success -> `critic` (ADR-0023: records a
    faithfulness verdict, never blocks yet) -> `guard_out` (ADR-0021: the
    final output-side gate runs on every path, including a good answer); a
    citation failure with a retry left -> back to `answer` with the failed
    draft now in state; out of retries -> `fail` (which raises, never
    reaching `critic`/`guard_out`)."""
    if state.get("answer") is not None:
        return "critic"
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
def build_graph(checkpointer: Any | None = None):
    """Builds and compiles the graph once per distinct `checkpointer`
    (`lru_cache` keys on its arguments — `checkpointer` is hashed by object
    identity, the default for a plain object with no custom `__eq__`, which
    every saver class here is). Safe to cache module-wide (no per-request
    objects live on the compiled graph itself — those arrive via `context=`
    at `.invoke()` time, see `state.py`'s `GraphContext`).

    ADR-0024: `checkpointer` is `None` by default (every existing caller
    that calls `build_graph()` with no args is unaffected — same "existing
    caller keeps working" contract `GraphContext`'s optional fields already
    give). The app passes a real saver — `InMemorySaver()` in unit tests,
    `AsyncPostgresSaver` in the API/CLI (`checkpointer.py`) — so a
    `thread_id` in `config["configurable"]` actually persists state across
    turns; `None` keeps today's stateless-per-call behaviour (a `thread_id`
    is accepted but nothing durable happens with it)."""
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("guard_in", guard_in_node)
    builder.add_node("router", router_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("fail", fail_node)
    builder.add_node("critic", critic_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("guard_out", guard_out_node)
    builder.add_edge(START, "guard_in")
    # ADR-0023: a clean question now goes to `router`, not straight to
    # `retrieve` — `router_node` decides the search scope before retrieval.
    builder.add_conditional_edges(
        "guard_in", route_after_guard, {"router": "router", "refuse": "refuse"}
    )
    builder.add_conditional_edges(
        "router", route_after_router, {"retrieve": "retrieve", "refuse": "refuse"}
    )
    # ADR-0021: `refuse` no longer goes straight to END — `guard_out` is the
    # final gate on EVERY path, a `guard_in`/`router` refusal included.
    builder.add_edge("refuse", "guard_out")
    builder.add_edge("retrieve", "answer")
    builder.add_conditional_edges(
        "answer", route_after_answer, {"answer": "answer", "fail": "fail", "critic": "critic"}
    )
    # ADR-0025: `critic` -> `hitl` -> `guard_out`, a single unconditional
    # edge into each. `hitl_node` pauses (interrupt()) only when the critic
    # ran and scored below `settings.critic_confidence_min`; otherwise it's
    # a pass-through (`return {}`, which follows this static edge to
    # `guard_out`). When it DOES pause, the resumed node instead returns a
    # `Command(goto="guard_out")` — that dynamic routing overrides this
    # static edge (verified: a node's `Command.goto` wins regardless of any
    # static edge also registered for that node), so both the pass-through
    # and every resume decision converge on `guard_out`, which stays the
    # final gate on every path (ADR-0021's invariant, unchanged).
    builder.add_edge("critic", "hitl")
    builder.add_edge("hitl", "guard_out")
    # `fail_node` always raises (never returns), so this edge is unreachable
    # in practice — kept for the same reason it always was: an explicit
    # graph shape, not a dangling node.
    builder.add_edge("fail", END)
    builder.add_edge("guard_out", END)
    return builder.compile(checkpointer=checkpointer)


async def ask(
    question: str,
    *,
    session: Session,
    embeddings: Embeddings,
    llm: Any,
    classifier: Any | None = None,
    router: Any | None = None,
    critic: Any | None = None,
    tools: dict[str, Any] | None = None,
    config: dict | None = None,
    checkpointer: Any | None = None,
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

    `router`/`critic`: ADR-0023's two new cheap-LLM calls, forwarded into
    `GraphContext`. `None` (the default) disables each — `router_node`/
    `critic_node` no-op, same "existing caller unaffected" contract
    `classifier` above already gives.

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
    pass one.

    `checkpointer`: ADR-0024's durable-state saver, forwarded to
    `build_graph()`. `None` (the default) is today's pre-ADR-0024 behaviour
    — every existing caller of `ask()` is unaffected."""
    graph = build_graph(checkpointer=checkpointer)
    context = GraphContext(
        session=session,
        embeddings=embeddings,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )
    state = await graph.ainvoke({"question": question}, context=context, config=config)
    return state["answer"]
