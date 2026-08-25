# src/compliance_copilot/graph/build.py — assembles the two nodes into the
# LangGraph `StateGraph` (docs/ARCHITECTURE.md §4: today's graph is just
# `retrieve -> answer`; the router/critic/interrupt nodes shown in that
# diagram are later features that get added as nodes+edges here, not a
# restructure).
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
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


def ask(
    question: str,
    *,
    session: Session,
    embeddings: Embeddings,
    llm: Any,
    classifier: Any | None = None,
    config: dict | None = None,
) -> AnswerSchema:
    """Convenience entry point: runs the compiled graph for one question and
    returns just the final `AnswerSchema` (rather than making every caller —
    cli.py, tests — reach into the raw state dict).

    `classifier`: ADR-0019's layer-2 guard, forwarded into `GraphContext`.
    `None` (the default) disables it — `guard_in_node` skips straight past
    the classifier check, same as before this feature existed, so every
    existing caller that doesn't pass one is unaffected.

    `config`: optional LangChain `RunnableConfig` dict (ADR-0009 amendment —
    `tracing.run_config()` builds one carrying a Langfuse callback when
    enabled). `None` here just means "no callbacks" — `graph.invoke()`
    already defaults to that with no `config=` kwarg at all, so this is
    additive, not a behaviour change for existing callers/tests that don't
    pass one."""
    graph = build_graph()
    context = GraphContext(session=session, embeddings=embeddings, llm=llm, classifier=classifier)
    state = graph.invoke({"question": question}, context=context, config=config)
    return state["answer"]
