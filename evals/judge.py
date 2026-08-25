# evals/judge.py — custom LLM-as-judge (ADR-0017, amending ADR-0005: Ragas
# 0.4.x deprecated the LangChain-wrapper integration point ADR-0005 assumed,
# and pulls a heavy ML-tooling dependency tree for 3 metric calls — see
# ADR-0017's "why not Ragas"). Scores one answer against two rubric
# questions: is every claim it makes actually backed by the cited/retrieved
# text (faithful), and does it actually answer the question in a way that
# agrees with a human-written reference (relevant)?
#
# Same construction pattern as graph/nodes.py's make_llm(): a plain chat
# model's `.with_structured_output(Schema, method="json_schema")` forces the
# judge's own output into a checkable shape, exactly like the answer LLM's
# output is forced into AnswerSchema.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from compliance_copilot.graph.state import AnswerSchema

JUDGE_SYSTEM_PROMPT = """You are a strict grader for a compliance-question-answering \
system over the EU AI Act and GDPR. You are told nothing about which model produced the \
answer being graded — grade only the text in front of you.

Score two things:
- faithful: true only if EVERY factual claim in the answer is actually supported by the \
provided context excerpts (the same text the answering system's citations quoted). A \
claim not backed by the excerpts, or a citation that misquotes them, makes this false.
- relevant: true only if the answer actually addresses the question asked AND agrees in \
substance with the reference answer given (a differently worded but substantively correct \
answer still counts as relevant; a partial, evasive, or substantively wrong answer does not).

Give a short `reasoning` (at most 300 characters) naming the specific gap if either is false."""


class JudgeVerdict(BaseModel):
    """The judge's structured verdict for one (question, answer) pair."""

    faithful: bool = Field(
        description="True only if every claim in the answer is supported by the context excerpts."
    )
    relevant: bool = Field(
        description="True only if the answer addresses the question and agrees with the reference."
    )
    reasoning: str = Field(
        max_length=300,
        description="Short justification, naming the specific gap if either is false.",
    )


def _build_messages(
    question: str, answer: AnswerSchema, contexts: list[str], reference: str
) -> list[tuple[str, str]]:
    no_context = "<context>(none — the answering system returned zero citations)</context>"
    contexts_block = "\n\n".join(f"<context>{c}</context>" for c in contexts) or no_context
    citations = [f"- {c.regulation}:{c.anchor} — {c.quote!r}" for c in answer.citations]
    citations_block = "\n".join(citations) or "(none)"
    human = (
        f"<question>{question}</question>\n\n"
        f"<answer>{answer.answer}</answer>\n\n"
        f"<citations>\n{citations_block}\n</citations>\n\n"
        f"<context_excerpts>\n{contexts_block}\n</context_excerpts>\n\n"
        f"<reference>{reference}</reference>"
    )
    return [("system", JUDGE_SYSTEM_PROMPT), ("human", human)]


def judge(
    question: str,
    answer: AnswerSchema,
    contexts: list[str],
    reference: str,
    llm: Any,
) -> JudgeVerdict:
    """Scores one answer. `llm` is a plain chat model (e.g. `ChatOpenAI(...,
    temperature=0)`) — this function is what binds it to `JudgeVerdict` via
    `with_structured_output`, not the caller, so a test double only needs
    `.with_structured_output(schema, method=...) -> object with .invoke(messages)`,
    the same minimal contract `graph/nodes.py`'s `answer_node` depends on."""
    structured_llm = llm.with_structured_output(JudgeVerdict, method="json_schema")
    messages = _build_messages(question, answer, contexts, reference)
    return structured_llm.invoke(messages)
