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
from compliance_copilot.graph.state import AnswerSchema, GraphContext, GraphState


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
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def ask(question: str, *, session: Session, embeddings: Embeddings, llm: Any) -> AnswerSchema:
    """Convenience entry point: runs the compiled graph for one question and
    returns just the final `AnswerSchema` (rather than making every caller —
    cli.py, tests — reach into the raw state dict)."""
    graph = build_graph()
    context = GraphContext(session=session, embeddings=embeddings, llm=llm)
    state = graph.invoke({"question": question}, context=context)
    return state["answer"]
