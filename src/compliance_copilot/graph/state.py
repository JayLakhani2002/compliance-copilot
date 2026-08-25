# src/compliance_copilot/graph/state.py — the shared state schema and the
# citation/answer types for the LangGraph graph (docs/ARCHITECTURE.md §4).
#
# `GraphState` is a plain TypedDict, not a class with methods: LangGraph
# passes it between nodes as data, and a node returns only the keys it
# changed (a partial update) which LangGraph merges into the running state.
# `question` is always present (the caller supplies it); `articles`,
# `recitals`, `answer` are filled in by nodes as the graph runs, so they're
# marked `NotRequired` rather than making the whole dict `total=False`
# (both forms are supported by LangGraph's schema handling — see
# `langgraph/_internal/_fields.py` in the installed package).
#
# `AnswerSchema`/`Citation` are Pydantic v2 models used as the LLM's forced
# output shape (`with_structured_output`, see nodes.py). Each field's
# `Field(description=...)` text is sent to the model as part of the schema
# it's filling in — so the description IS the instruction the model reads,
# not just documentation for humans.
#
# `GraphContext` carries the run's dependencies (DB session, embeddings,
# LLM client) into nodes via LangGraph's `context_schema`/`Runtime[...]`
# mechanism (ADR-0014) instead of living inside `GraphState`. Why: state is
# data that (once ADR-0001's checkpointer is wired in for the HITL feature)
# gets serialised to Postgres — a SQLAlchemy `Session` or an LLM client
# object is neither serialisable nor something that should be persisted per
# request, so it's passed at `.invoke()` time instead of stored in state.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from compliance_copilot.guards.injection import GuardResult
from compliance_copilot.retriever import RetrievedChunk


class Citation(BaseModel):
    """One citation backing a claim in `AnswerSchema.answer`."""

    regulation: str = Field(
        description="Must be exactly one of the retrieved articles' `regulation` "
        "values (e.g. 'ai_act' or 'gdpr') — never a value you haven't seen."
    )
    anchor: str = Field(
        description="Must be exactly one of the retrieved articles' anchor ids "
        "(e.g. 'art_6') — never a recital anchor, and never an anchor not "
        "present in the provided excerpts."
    )
    quote: str = Field(
        description="A SHORT verbatim excerpt (one sentence or clause, at most "
        "~300 characters) copied word-for-word from the cited article's "
        "excerpt — no paraphrasing, no summarising, no whole paragraphs."
    )


class AnswerSchema(BaseModel):
    """The compliance assistant's answer, forced into a checkable shape: every
    factual claim traces back to a retrieved article through `citations`."""

    answer: str = Field(
        description="The answer to the user's question, written only from the "
        "provided excerpts. If the excerpts don't answer the question, say so "
        "here plainly and leave `citations` empty — never guess."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Every citation backing a factual claim made in `answer`. "
        "Empty if the excerpts didn't answer the question.",
    )


class CitationError(ValueError):
    """Raised when `AnswerSchema` cites something `retrieve` didn't actually
    fetch, or misquotes a retrieved chunk. This is a hard failure, not a
    warning — a bad citation is blocked, never silently swapped for a real
    one (ADR-0014; same "guard blocks, never swaps" rule ADR-0013 already
    applies to retrieval). Message deliberately omits the user's question —
    it's built only from citation/anchor data, so logging or displaying this
    error can't leak the question text."""


@dataclass
class GraphContext:
    """Run-scoped dependencies, supplied per `.invoke(..., context=...)` call
    (LangGraph's `context_schema`/`Runtime[GraphContext]` mechanism) rather
    than stored in `GraphState` — see the module docstring above for why."""

    session: Session
    embeddings: Embeddings
    # `Any`, not `ChatAnthropic`: the only contract nodes.answer() needs is
    # `.invoke(messages) -> AnswerSchema`, which is exactly what
    # `ChatAnthropic(...).with_structured_output(...)` returns — typing this
    # as the concrete provider class would leak an implementation detail
    # into the type nodes.py depends on, and would make the test fake in
    # tests/test_graph.py fail a strict type check for no real benefit.
    llm: Any
    # ADR-0019: layer 2 of `guard_in` — an object with `.invoke(messages)
    # -> Verdict` (guards/classifier.py's `make_classifier_llm()` return
    # value), or `None` to disable the classifier entirely (settings.
    # classifier_enabled=False). Defaults `None` so every existing caller
    # that builds a `GraphContext` without it (tests, any code that hasn't
    # been touched by this feature) keeps working unchanged.
    classifier: Any | None = None


class GraphState(TypedDict):
    """Shared state threaded through `guard_in` -> (`retrieve` -> `answer` ->
    (`answer` | `fail`)) | `refuse`. Keys nodes don't fill in until they run
    are `NotRequired` so the initial `{"question": ...}` dict passed to
    `.invoke()` is still valid input.

    `guard`/`refused` back Day 11's input guard (ADR-0018): `guard_in_node`
    (nodes.py) always fills `guard`; `refused` is only set `True` by
    `refuse_node`, so its absence/`False` means the question passed the
    heuristic check (or refusal never applies before that node runs).

    `draft`/`citation_error`/`attempts` back the Day 7 retry-once loop
    (ADR-0015): a failed `answer_node` call stores its rejected `AnswerSchema`
    and the validation error here instead of raising immediately, so
    `route_after_answer` (build.py) can send the graph back to `answer` once
    with that context appended as extra message turns."""

    question: str
    guard: NotRequired[GuardResult]
    refused: NotRequired[bool]
    articles: NotRequired[list[RetrievedChunk]]
    recitals: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[AnswerSchema | None]
    draft: NotRequired[AnswerSchema | None]
    citation_error: NotRequired[str | None]
    attempts: NotRequired[int]
