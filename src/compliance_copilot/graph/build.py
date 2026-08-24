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

from compliance_copilot.graph.nodes import answer_node, retrieve_node
from compliance_copilot.graph.state import AnswerSchema, CitationError, GraphContext, GraphState

# 2 = one retry (the initial attempt + one self-correction) — ADR-0015. One
# place, read by `route_after_answer` only; `answer_node` doesn't need to
# know the cap, it just increments `attempts` each time it runs.
MAX_ATTEMPTS = 2


def route_after_answer(state: GraphState) -> str:
    """Conditional edge after `answer` — a plain state check, no `runtime`
    param needed (routing only reads two `GraphState` keys; `path` callables
    passed to `add_conditional_edges` support a `runtime` param the same way
    nodes do, via LangGraph's `RunnableCallable` wrapping, but this routing
    decision doesn't need one). Success -> END; a citation failure with a
    retry left -> back to `answer` with the failed draft now in state; out
    of retries -> `fail`."""
    if state.get("answer") is not None:
        return END
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
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("fail", fail_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_conditional_edges(
        "answer", route_after_answer, {"answer": "answer", "fail": "fail", END: END}
    )
    builder.add_edge("fail", END)
    return builder.compile()


def ask(question: str, *, session: Session, embeddings: Embeddings, llm: Any) -> AnswerSchema:
    """Convenience entry point: runs the compiled graph for one question and
    returns just the final `AnswerSchema` (rather than making every caller —
    cli.py, tests — reach into the raw state dict)."""
    graph = build_graph()
    context = GraphContext(session=session, embeddings=embeddings, llm=llm)
    state = graph.invoke({"question": question}, context=context)
    return state["answer"]
