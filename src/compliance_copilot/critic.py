# src/compliance_copilot/critic.py — the critic: a cheap-LLM call that
# checks whether the drafted answer's prose claims actually follow from the
# text of its own cited excerpts (semantic support), not just whether the
# quote string is present verbatim — that check already exists
# (`_validate_citations`, graph/nodes.py, ADR-0014; `guard_out`'s
# `citation_not_retrieved` re-check, ADR-0021). The critic is the online
# sibling of `evals/judge.py`'s offline `JudgeVerdict.faithful` (see
# docs/decisions/ADR-0023) — a NEW, structurally similar schema, not an
# import of `evals.judge`: `evals/` is a dev/CI-only tree
# (pyproject.toml's `packages = ["src/compliance_copilot"]`), so importing
# it from shipped code would be a real layering violation.
#
# Today it only RECORDS a verdict into state and (from api.py) the trace —
# it never blocks an answer (lesson 18: "records first, blocks later").
# Day 20 is the day a low-confidence score becomes an `interrupt()`.
from __future__ import annotations

import html
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from compliance_copilot.settings import settings

logger = logging.getLogger(__name__)


class CriticVerdict(BaseModel):
    """`reasoning` capped at 300 chars (mirrors `evals.judge.JudgeVerdict`'s
    own cap) — free model text, never rendered to the end user (ADR-0023's
    leak-guard note: logs/traces/state only)."""

    faithful: bool = Field(
        description="True only if every claim in the answer is supported by the cited excerpts."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this verdict, 0-1.")
    reasoning: str = Field(
        max_length=300, description="Short justification, naming the gap if not faithful."
    )


# Reuses evals/judge.py's JUDGE_SYSTEM_PROMPT wording almost verbatim
# (deliberately — same rubric, adapted for `confidence` instead of
# `relevant`; `relevant` is a clean future extension, not Day 18/20 scope,
# see docs/decisions/ADR-0023). Only shown the CITED excerpts' own text
# (`critic_node`, graph/nodes.py), never the whole corpus, never the raw
# question/system prompt — same injection-surface mitigation as every other
# guard: structured output only, no tools, no free-text field to land an
# injected instruction in.
CRITIC_SYSTEM_PROMPT = """You are a strict grader for a compliance-question-answering \
system over the EU AI Act and GDPR. Score whether the answer is faithful to its \
own cited excerpts: true only if EVERY factual claim in the answer is actually \
supported by the provided context excerpts (the same text the answer's citations \
quoted). A claim not backed by the excerpts, or a citation that misquotes them, \
makes this false. Give a confidence (0-1) in your verdict and a short reasoning \
(at most 300 characters) naming the specific gap if not faithful."""

_DEFAULT_MODELS = {"openai": "gpt-4.1-nano", "anthropic": "claude-haiku-4-5"}


def make_critic_llm() -> Any:
    """Mirrors `make_classifier_llm()`/`router.make_router_llm()` exactly —
    same cheap-tier provider gating, same `max_retries=0` (nothing useful to
    retry under a tight budget; `critique()` below always returns a verdict,
    a pessimistic one on any failure, never raises)."""
    provider = settings.llm_provider
    model = settings.critic_model or _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["openai"])
    if provider == "anthropic":
        llm = ChatAnthropic(
            model=model, temperature=0, timeout=settings.critic_timeout_s, max_retries=0
        )
    else:
        llm = ChatOpenAI(
            model=model, temperature=0, timeout=settings.critic_timeout_s, max_retries=0
        )
    return llm.with_structured_output(CriticVerdict, method="json_schema")


def _build_messages(question: str, answer: str, contexts: list[str]) -> list[tuple[str, str]]:
    # Every value here is untrusted text (user question, model answer, retrieved
    # law) being wrapped in XML delimiters. Escape it first — same rule as the
    # answer prompt and the router (ADR-0015) — so a literal "</question>" in the
    # input cannot close the tag early and smuggle in a fake <answer>/<context>
    # block that talks the critic into a "faithful" verdict.
    no_context = "<context>(none — zero citations)</context>"
    wrapped = [f"<context>{html.escape(c, quote=False)}</context>" for c in contexts]
    contexts_block = "\n\n".join(wrapped) or no_context
    human = (
        f"<question>{html.escape(question, quote=False)}</question>\n\n"
        f"<answer>{html.escape(answer, quote=False)}</answer>\n\n{contexts_block}"
    )
    return [("system", CRITIC_SYSTEM_PROMPT), ("human", human)]


def critique(question: str, answer: str, contexts: list[str], llm: Any) -> CriticVerdict:
    """Runs the critic, returning a verdict — NEVER raises past this
    function. Fail policy is the opposite bias from the router/classifier's
    fail-open: nothing branches on the critic's output yet, so there's no
    availability risk to protect, but a silent gap or a falsely reassuring
    verdict would corrupt Day 20's threshold-tuning work (lesson 18: "on
    evidence, not intuition"). On any exception, still write a verdict, a
    PESSIMISTIC one — `faithful=False, confidence=0.0` — logged the same
    "exception class name only" way every guard already follows."""
    try:
        return llm.invoke(_build_messages(question, answer, contexts))
    except Exception as exc:
        logger.warning("critic call failed, recording pessimistic verdict: %s", type(exc).__name__)
        return CriticVerdict(
            faithful=False, confidence=0.0, reasoning=f"critic_error:{type(exc).__name__}"
        )
