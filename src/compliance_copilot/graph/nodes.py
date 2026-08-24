# src/compliance_copilot/graph/nodes.py — the two graph nodes
# (docs/ARCHITECTURE.md §4): `retrieve_node` fetches context, `answer_node`
# drafts and validates a cited response. Both take `runtime:
# Runtime[GraphContext]` as their second argument — LangGraph's DI mechanism
# (ADR-0014) — instead of constructing a DB session or LLM client
# themselves.
from __future__ import annotations

import re

from langchain_anthropic import ChatAnthropic
from langgraph.runtime import Runtime

from compliance_copilot.graph.state import (
    AnswerSchema,
    CitationError,
    GraphContext,
    GraphState,
)
from compliance_copilot.retriever import RetrievedChunk, retrieve
from compliance_copilot.settings import settings

# Module constant, not built inline in `answer()`: stable text across every
# call is what lets Day 7 add Anthropic prompt caching (`cache_control` on
# this block) without touching this string itself — ADR-0002's "how to
# reverse" note flags the system prompt as that target.
SYSTEM_PROMPT = """You are a compliance assistant for the EU AI Act and GDPR.

Answer ONLY using the excerpts provided below — never from outside knowledge.
Every factual claim in your answer needs a citation. Citations must use
exactly the `regulation` and `anchor` ids given with each excerpt — never
invent or guess one. `quote` must be a verbatim, word-for-word excerpt copied
from the cited excerpt's text, not a paraphrase or summary.

Recitals are supporting context only: never cite a recital, only articles.

If the excerpts do not answer the question, say so plainly in your answer
and return zero citations — do not guess."""


def retrieve_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    """Node 1: articles are the primary, citable context (ADR-0013's default
    `kinds=("article",)`); recitals are fetched separately as secondary,
    non-citable context for `answer_node` to read but never cite.

    Calls the module-level `retrieve` (imported from retriever.py) rather
    than embedding the lookup inline — tests monkeypatch that module-level
    name directly (`compliance_copilot.graph.nodes.retrieve`) to run the
    whole compiled graph with no DB/network, since a Python function looks
    up a global name from its module's namespace at call time, not at
    def time."""
    ctx = runtime.context
    articles = retrieve(
        state["question"],
        k=5,
        kinds=("article",),
        session=ctx.session,
        embeddings=ctx.embeddings,
    )
    recitals = retrieve(
        state["question"],
        k=3,
        kinds=("recital",),
        session=ctx.session,
        embeddings=ctx.embeddings,
    )
    return {"articles": articles, "recitals": recitals}


def _render_chunk(chunk: RetrievedChunk) -> str:
    header = f"[regulation={chunk.regulation} anchor={chunk.anchor} title={chunk.title}]"
    return f"{header}\n{chunk.text}"


# The corpus's own text uses Unicode curly quotes (U+2018/U+2019, U+201C/
# U+201D) around every defined term (chunker.py's definition-boundary regex
# matches '‘' for the same reason) — the model may reproduce a quote with
# straight ASCII quotes instead. Map both to straight so that difference
# alone never causes a false-positive CitationError (reviewer round 1).
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

# Minimum normalised-quote length: an empty or 1-2 character quote is a
# substring of almost any text, so it would pass the verbatim check with
# zero actual information content (reviewer round 1) — reject it outright
# rather than let it through as a "cited" claim.
_MIN_QUOTE_LENGTH = 20


def _normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded, curly-quotes-to-straight —
    applied to both the model's quote and the source chunk text before the
    substring check, so a quote that only differs by spacing, letter case,
    or quote-mark style still counts as verbatim (ADR-0014)."""
    return re.sub(r"\s+", " ", text).strip().casefold().translate(_QUOTE_MAP)


def _build_messages(state: GraphState) -> list[tuple[str, str]]:
    """Renders the retrieved context + question into the (role, content)
    message pairs `ChatAnthropic.invoke()` accepts (confirmed against
    `langchain_core.messages.utils._convert_to_message`'s documented
    2-tuple-of-(role, template) input form)."""
    articles_block = "\n\n".join(_render_chunk(c) for c in state["articles"])
    recitals_block = "\n\n".join(_render_chunk(c) for c in state["recitals"])
    human = (
        f"{articles_block}\n\n"
        f"Supporting context (do not cite):\n{recitals_block}\n\n"
        f"Question: {state['question']}"
    )
    return [("system", SYSTEM_PROMPT), ("human", human)]


def answer_node(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    """Node 2: calls the LLM for a structured `AnswerSchema`, then validates
    every citation against what `retrieve` actually fetched before returning
    it — an uncaught bad citation must never reach a caller (ADR-0014)."""
    messages = _build_messages(state)
    result: AnswerSchema = runtime.context.llm.invoke(messages)

    # Group by (regulation, anchor), not a plain dict keyed on that pair:
    # an oversize article (e.g. art_3) is split into multiple parts, each a
    # separate retrieved row sharing one anchor (chunker.py) — retrieve()
    # can return more than one part of the same anchor in one top-k result.
    # The model is never shown part numbers (_render_chunk only renders
    # regulation/anchor/title), so a citation can't specify which part it
    # means — accept the quote if it's verbatim in ANY retrieved part
    # (reviewer round 1: a plain dict silently dropped all but the last
    # part, wrongly rejecting a correct quote from a dropped part).
    parts_by_key: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for c in state["articles"]:
        parts_by_key.setdefault((c.regulation, c.anchor), []).append(c)
    recital_anchors = {c.anchor for c in state["recitals"]}
    allowed_anchors = sorted({anchor for _, anchor in parts_by_key})

    for citation in result.citations:
        key = (citation.regulation, citation.anchor)
        parts = parts_by_key.get(key)
        if parts is None:
            reason = (
                "a retrieved recital, not an article (recitals are context only)"
                if citation.anchor in recital_anchors
                else "not among the retrieved articles"
            )
            raise CitationError(
                f"Citation {citation.regulation}:{citation.anchor} is {reason}. "
                f"Allowed article anchors: {allowed_anchors}"
            )
        normalised_quote = _normalise(citation.quote)
        if len(normalised_quote) < _MIN_QUOTE_LENGTH:
            raise CitationError(
                f"Citation {citation.regulation}:{citation.anchor}'s quote is too short "
                f"to verify (must be at least {_MIN_QUOTE_LENGTH} characters after "
                f"normalisation). Allowed article anchors: {allowed_anchors}"
            )
        if not any(normalised_quote in _normalise(part.text) for part in parts):
            raise CitationError(
                f"Citation {citation.regulation}:{citation.anchor}'s quote was not "
                f"found verbatim in the retrieved excerpt. Allowed article anchors: "
                f"{allowed_anchors}"
            )

    return {"answer": result}


def make_llm(model: str | None = None) -> ChatAnthropic:
    """The one place `ChatAnthropic` is constructed (ADR-0002 "how to
    reverse"). `method="json_schema"` is pinned explicitly rather than left
    to default: reading `with_structured_output`'s signature in the
    installed `langchain_anthropic` package shows its own default is
    `"function_calling"`, not `"json_schema"` — pinning avoids depending on
    that default silently changing later."""
    llm = ChatAnthropic(model=model or settings.answer_model, temperature=0, max_tokens=1024)
    return llm.with_structured_output(AnswerSchema, method="json_schema")
