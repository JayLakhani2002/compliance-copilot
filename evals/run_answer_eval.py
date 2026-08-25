# evals/run_answer_eval.py — the answer-quality gate (ADR-0017): for each of
# the 10 golden Q&A pairs (evals/golden_answers.jsonl), runs the REAL
# retrieve->answer graph (`ask()`, ADR-0014) and scores the result with the
# custom LLM-as-judge (evals/judge.py). Faithfulness gates the merge;
# relevancy and the citation-error rate are reported but don't gate (per
# ADR-0017/lesson 10 — relevancy is more phrasing-sensitive than the
# factual-grounding property this project actually needs to guarantee).
#
# Costs real money (a real answer LLM call + a real judge LLM call per
# question) — run it locally, occasionally, or let CI's nightly/labelled/
# dispatched `answer-quality` job run it, never in a tight loop:
#     set -a; source .env; set +a
#     uv run python -m evals.run_answer_eval --faithfulness-min 0.8
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.db import Chunk, Document, get_engine
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import CitationError
from compliance_copilot.graph import ask as ask_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.graph.state import AnswerSchema
from evals.judge import JudgeVerdict, judge

GOLDEN_ANSWERS_PATH = Path(__file__).parent / "golden_answers.jsonl"

# gpt-4.1-mini: same interim-default reasoning as graph/nodes.py's
# make_llm() (ADR-0002 amendment) — cheapest current OpenAI model that
# supports with_structured_output(method="json_schema") without silently
# dropping temperature=0. The judge is deliberately a fresh ChatOpenAI
# instance here, not the answer LLM passed in from make_llm(), so a bug
# specific to one call site can't accidentally share state with the other.
_JUDGE_MODEL = "gpt-4.1-mini"


@dataclass
class GoldenAnswer:
    id: str
    question: str
    expected_anchors: list[str]
    reference: str


def load_golden_answers(path: Path = GOLDEN_ANSWERS_PATH) -> list[GoldenAnswer]:
    entries: list[GoldenAnswer] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(GoldenAnswer(**json.loads(line)))
    return entries


@dataclass
class QuestionOutcome:
    id: str
    faithful: bool
    relevant: bool
    n_citations: int
    note: str  # trimmed judge reasoning, or "CitationError: ..." on a refusal
    is_citation_error: bool


def _outcome_from_citation_error(golden: GoldenAnswer, exc: CitationError) -> QuestionOutcome:
    """A CitationError means the graph refused to answer at all (ADR-0014's
    hard-error path) — counted as a faithfulness AND relevancy failure, not
    skipped, since a refusal that a person can't work with is exactly what
    this gate exists to catch."""
    return QuestionOutcome(
        id=golden.id,
        faithful=False,
        relevant=False,
        n_citations=0,
        note=f"CitationError: {exc}"[:200],
        is_citation_error=True,
    )


def _outcome_from_verdict(
    golden: GoldenAnswer, answer: AnswerSchema, verdict: JudgeVerdict
) -> QuestionOutcome:
    return QuestionOutcome(
        id=golden.id,
        faithful=verdict.faithful,
        relevant=verdict.relevant,
        n_citations=len(answer.citations),
        note=verdict.reasoning[:200],
        is_citation_error=False,
    )


def _contexts_for(answer: AnswerSchema, session: Session) -> list[str]:
    """The cited chunks' own texts, fetched by (regulation, anchor) — the
    judge is shown exactly what the answering system was allowed to cite,
    never the whole corpus, so "faithful" measures grounding in the actual
    cited excerpts (same (regulation, anchor) pairing nodes.py's
    `_validate_citations` uses; an oversize article can have several parts
    sharing one anchor, so every part is included)."""
    keys = {(c.regulation, c.anchor) for c in answer.citations}
    texts: list[str] = []
    for regulation, anchor in keys:
        rows = session.execute(
            select(Chunk.text)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.regulation == regulation, Chunk.anchor_id == anchor)
        ).all()
        texts.extend(text for (text,) in rows)
    return texts


def aggregate(outcomes: list[QuestionOutcome]) -> dict[str, float]:
    n = len(outcomes)
    if n == 0:
        return {"faithfulness": 0.0, "relevancy": 0.0, "citation_error_rate": 0.0, "n": 0}
    return {
        "faithfulness": sum(o.faithful for o in outcomes) / n,
        "relevancy": sum(o.relevant for o in outcomes) / n,
        "citation_error_rate": sum(o.is_citation_error for o in outcomes) / n,
        "n": n,
    }


def print_report(outcomes: list[QuestionOutcome], aggregates: dict[str, float]) -> None:
    print("\n=== answer-quality eval ===\n")
    print(f"{'id':<6} {'faithful':<9} {'relevant':<9} {'n_cites':<8} note")
    for o in outcomes:
        print(f"{o.id:<6} {o.faithful!s:<9} {o.relevant!s:<9} {o.n_citations:<8} {o.note}")
    print("\n--- summary ---")
    print(
        f"faithfulness={aggregates['faithfulness']:.3f}  "
        f"relevancy={aggregates['relevancy']:.3f}  "
        f"citation_error_rate={aggregates['citation_error_rate']:.3f}  "
        f"n={aggregates['n']}"
    )


def _print_cost_note(n_questions: int) -> None:
    # Rough order-of-magnitude, not an invoice: ~2 LLM calls/question (answer
    # + judge) at gpt-4.1-mini's verified $0.40/$1.60 per MTok in/out
    # (ADR-0002), ~1,500 input + ~250 output tokens/call (rough estimate).
    calls = n_questions * 2
    input_tokens = calls * 1500
    output_tokens = calls * 250
    cost = (input_tokens / 1_000_000) * 0.40 + (output_tokens / 1_000_000) * 1.60
    print(
        f"\nrough cost: ~{calls} LLM calls, ~${cost:.3f} "
        "(gpt-4.1-mini pricing, order-of-magnitude only)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faithfulness-min", type=float, default=0.0)
    parser.add_argument(
        "--json", type=Path, default=None, help="Dump the full report to this path."
    )
    args = parser.parse_args()

    goldens = load_golden_answers()
    embeddings = get_embeddings()
    answer_llm = make_llm()
    judge_llm = ChatOpenAI(model=_JUDGE_MODEL, temperature=0)

    outcomes: list[QuestionOutcome] = []
    with Session(get_engine()) as session:
        for golden in goldens:
            config = tracing.run_config(tags=["answer-quality-eval", f"golden:{golden.id}"])
            try:
                answer = ask_graph(
                    golden.question,
                    session=session,
                    embeddings=embeddings,
                    llm=answer_llm,
                    config=config,
                )
            except CitationError as exc:
                outcomes.append(_outcome_from_citation_error(golden, exc))
                continue

            contexts = _contexts_for(answer, session)
            verdict = judge(golden.question, answer, contexts, golden.reference, judge_llm)
            outcomes.append(_outcome_from_verdict(golden, answer, verdict))

            trace_id = tracing.current_trace_id(config)
            tracing.score("judge_faithful", 1.0 if verdict.faithful else 0.0, trace_id)
            tracing.score("judge_relevant", 1.0 if verdict.relevant else 0.0, trace_id)

    aggregates = aggregate(outcomes)
    print_report(outcomes, aggregates)
    _print_cost_note(len(goldens))

    if args.json:
        args.json.write_text(
            json.dumps(
                {"outcomes": [asdict(o) for o in outcomes], "aggregates": aggregates}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    if aggregates["faithfulness"] < args.faithfulness_min:
        raise SystemExit(
            f"faithfulness={aggregates['faithfulness']:.3f} below "
            f"--faithfulness-min={args.faithfulness_min:.3f} — failing as a CI gate"
        )


if __name__ == "__main__":
    main()
