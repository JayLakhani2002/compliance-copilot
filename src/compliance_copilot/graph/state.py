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
# data that (ADR-0024's Postgres checkpointer, wired in for durable
# multi-turn conversations) gets serialised to Postgres — a SQLAlchemy
# `Session` or an LLM client object is neither serialisable nor something
# that should be persisted per request, so it's passed at `.invoke()` time
# instead of stored in state.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from compliance_copilot.critic import CriticVerdict
from compliance_copilot.guards.injection import GuardResult
from compliance_copilot.guards.output import OutputVerdict
from compliance_copilot.retriever import RetrievedChunk
from compliance_copilot.router import RouterVerdict


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


class ToolCallError(RuntimeError):
    """Raised by `retrieve_node` (graph/nodes.py) when an MCP tool call
    fails — a transport error/timeout that exhausted its bounded retries, a
    tool-reported validation failure, or a malformed result shape (ADR-0007
    Day-17 amendment). Deliberately a hard failure, never a silent fallback
    to direct retrieval: a quiet fallback here would hide a real MCP outage
    behind a confident-looking answer. Falls through api.py's `/ask` generic
    exception handler as an `internal_error` SSE event, and surfaces to the
    CLI the same way `OutputGuardError` does — an internal failure, not a
    refusal. Message is built only from the tool name and error class, never
    the call's arguments (which carry the question) — same rule
    `CitationError`'s docstring above already follows."""


@dataclass(frozen=True)
class Turn:
    """One (question, answer) pair for `GraphState.history` (ADR-0024). The
    question is whatever `guard_in_node` left in `state["question"]` — the
    already-redacted text when PII redaction fired — never the raw input,
    so a checkpointed turn never durably stores more PII than the rest of
    the pipeline already saw. Frozen + two plain `str` fields: the same
    round-trip guarantee LangGraph's checkpoint serde already gives every
    other dataclass in this state (`GuardResult`, `OutputVerdict`, ...) —
    `dataclasses.is_dataclass` reconstruction, no custom serde needed."""

    question: str
    answer: str


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
    # ADR-0007 Day-17 amendment: `search_regulation`/`get_article`
    # LangChain `BaseTool` objects (from `langchain_mcp_adapters.
    # MultiServerMCPClient.get_tools()`, built once in build.py's
    # `make_mcp_tools()`), keyed by tool name — `retrieve_node` reads
    # `tools["search_regulation"]` etc. instead of importing `retrieve()`
    # directly. `None` (the default) means "no tools loaded" — a real
    # question reaching `retrieve_node` with no tools raises `ToolCallError`
    # immediately (fail loudly, never a silent fallback to direct
    # retrieval); tests that only exercise `guard_in`'s refusal path never
    # reach `retrieve_node` at all, so they're unaffected by leaving this
    # unset, same "existing caller keeps working" contract `classifier`
    # above already established.
    tools: Mapping[str, Any] | None = None
    # ADR-0023: an object with `.invoke(messages) -> RouterVerdict`
    # (`router.make_router_llm()`'s return value), or `None` to disable the
    # router entirely (`settings.router_enabled=False`) — same "existing
    # caller keeps working" default `classifier` above already establishes.
    router: Any | None = None
    # ADR-0023: an object with `.invoke(messages) -> CriticVerdict`
    # (`critic.make_critic_llm()`'s return value), or `None` to disable the
    # critic entirely (`settings.critic_enabled=False`) — same contract.
    critic: Any | None = None


class GraphState(TypedDict):
    """Shared state threaded through `guard_in` -> (`router` -> (`retrieve` ->
    `answer` -> (`answer` -> `critic` -> `guard_out` | `fail`)) | `refuse`) |
    (`refuse` -> `guard_out`) — ADR-0023 inserts `router` between `guard_in`
    and `retrieve`/`refuse`, and `critic` between `answer`'s success branch
    and `guard_out`. Keys nodes don't fill in until they run are
    `NotRequired` so the initial `{"question": ...}` dict passed to
    `.invoke()` is still valid input.

    `guard`/`refused` back Day 11's input guard (ADR-0018): `guard_in_node`
    (nodes.py) always fills `guard`; `refused` is only set `True` by
    `refuse_node`, so its absence/`False` means the question passed the
    heuristic check (or refusal never applies before that node runs).

    `draft`/`citation_error`/`attempts` back the Day 7 retry-once loop
    (ADR-0015): a failed `answer_node` call stores its rejected `AnswerSchema`
    and the validation error here instead of raising immediately, so
    `route_after_answer` (build.py) can send the graph back to `answer` once
    with that context appended as extra message turns.

    `pii_entities` (ADR-0020): entity TYPE names only (e.g.
    ("EMAIL_ADDRESS", "PERSON")), empty when nothing was found — the per-turn
    reset (ADR-0024) writes `()` at the start of every turn and `guard_in_node`
    fills it in the SAME return dict that overwrites `question` with the
    redacted text, so a NON-EMPTY value always means "the `question` you're
    reading has already been redacted" and never describes an earlier turn.

    `output_guard` (ADR-0021): set by `guard_out_node`, the final gate that
    runs on EVERY terminal path — a passed answer, a `guard_in` refusal, and
    an exhausted-retry failure all funnel through it before END. Always
    present once that node has run; its `ok`/`reason` are safe to log (see
    guards/output.py's `OutputVerdict`).

    `router` (ADR-0023): set by `router_node`, which runs right after
    `guard_in` on a clean question — ABSENT when the router is disabled
    (`GraphContext.router=None`), never present on a `guard_in` refusal
    (that path skips `router` entirely). `retrieve_node` reads it (falling
    back to "no filter" when absent) to narrow `search_regulation`'s scope.

    `critic` (ADR-0023): set by `critic_node`, which runs only on the
    answer-success branch, right before `guard_out` — ABSENT on a refusal
    (there's no substantive claim to critique) and when the critic is
    disabled (`GraphContext.critic=None`).

    `history` (ADR-0024): a plain (not `Annotated`/reducer) key — `guard_out_
    node` is the ONLY node that ever writes it, once per run, so "last write
    wins" (LangGraph's default merge for a key with no reducer) is already
    exactly the semantics needed: it reads the prior checkpointed list,
    appends this turn, and returns the FULL replacement, already capped to
    the last 3 turns (`guard_out_node`'s docstring). A `BinaryOperatorAggregate`
    reducer (e.g. `operator.add`) would be the wrong tool here — it combines
    the OLD channel value with whatever a node returns, so returning an
    already-capped full list under a reducer would concatenate it onto
    itself every turn instead of replacing it; a reducer earns its keep only
    when more than one node/branch can write the same key in one run, which
    never happens for `history`. `answer_node` renders it as prior
    `("human", question), ("ai", answer)` pairs, after the system prompt and
    before the current question (prompt-caching order, ADR-0002/ADR-0007)."""

    question: str
    # `router`/`critic`/`output_guard` are typed `| None` (not just their
    # bare verdict type) because ADR-0024's per-turn reset (`guard_in_node`'s
    # `_PER_TURN_RESET`, nodes.py) explicitly writes `None` into these keys
    # at the start of every turn — the exact same "not set this turn" value
    # `state.get(...)` already treats an ABSENT key as, just made explicit
    # instead of relying on a stale value never being returned again.
    guard: NotRequired[GuardResult]
    refused: NotRequired[bool]
    pii_entities: NotRequired[tuple[str, ...]]
    router: NotRequired[RouterVerdict | None]
    articles: NotRequired[list[RetrievedChunk]]
    recitals: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[AnswerSchema | None]
    draft: NotRequired[AnswerSchema | None]
    citation_error: NotRequired[str | None]
    attempts: NotRequired[int]
    critic: NotRequired[CriticVerdict | None]
    output_guard: NotRequired[OutputVerdict | None]
    history: NotRequired[list[Turn]]
