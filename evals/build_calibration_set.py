# evals/build_calibration_set.py — builds the 20-item judge-calibration set
# (ADR-0027, lesson 22): does evals/judge.py's LLM-as-judge agree with a
# careful human reading the same item?
#
# 10 items are the golden Q&A set (evals/golden_answers.jsonl) run through
# the REAL retrieve->answer graph + judge — the exact pipeline pieces
# evals/run_answer_eval.py's `_run_goldens`/`_contexts_for` already use,
# reused here rather than reimplemented. The other 10 are a deterministic
# subset (first 5 ids, sorted, from each file — no randomness, so re-running
# this script always samples the same items) of evals/benign.jsonl and
# evals/redteam.jsonl run through the REAL guard_in->retrieve->answer->
# guard_out graph (evals/run_redteam.py's pipeline pieces) — a different
# *shape* of answer (short refusals, benign one-off phrasing) the tidy
# golden-answer style never produces. The judge doesn't normally see these
# items (run_redteam.py's own scoring is deterministic, ADR-0022) — it's run
# here purely so there's something to calibrate against.
#
# Two files come out on purpose, not one: evals/calibration/items.jsonl (the
# question/answer/contexts a human reads) and evals/calibration/
# judge_verdicts.jsonl (the judge's own verdict) are SEPARATE so a human can
# label items.jsonl blind to what the judge said — see RUBRIC.md's
# blind-labelling protocol. evals/calibration/labels.template.jsonl is the
# empty shape Jay's real labels.jsonl should follow.
#
# Costs real money (an answer-model call + a judge call per item, plus a
# classifier call for the pipeline items) — same order of magnitude as one
# evals/run_answer_eval.py run, run occasionally, not in CI:
#     set -a; source .env; set +a
#     uv run python -m evals.build_calibration_set
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.db import get_engine
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import ask as ask_graph
from compliance_copilot.graph import make_mcp_tools
from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.graph.state import AnswerSchema, CitationError, GraphContext
from compliance_copilot.guards.classifier import make_classifier_llm
from evals.judge import JudgeVerdict, judge
from evals.run_answer_eval import _JUDGE_MODEL, GoldenAnswer, _contexts_for, load_golden_answers
from evals.run_redteam import Attack, BenignQuestion, load_attacks, load_benign

CALIBRATION_DIR = Path(__file__).parent / "calibration"
ITEMS_PATH = CALIBRATION_DIR / "items.jsonl"
VERDICTS_PATH = CALIBRATION_DIR / "judge_verdicts.jsonl"
LABELS_TEMPLATE_PATH = CALIBRATION_DIR / "labels.template.jsonl"

# "pick a deterministic subset by id" (lesson 22) — first 5 sorted ids from
# each fixture file. redteam/benign ids are zero-padded (rt01..rt40,
# bn01..bn20) so lexical sort == numeric sort; no randomness involved.
_N_PER_FIXTURE = 5


@dataclass
class CalibrationItem:
    """One row of items.jsonl — what a human labelling blind actually
    reads. `reference` is `None` for the benign/red-team items: they have no
    hand-written reference answer the way golden_answers.jsonl does."""

    id: str
    source: str  # "golden" | "benign" | "redteam"
    question: str
    answer: str
    contexts: list[str]
    reference: str | None


def _first_n_sorted_ids(ids: list[str], n: int) -> set[str]:
    return set(sorted(ids)[:n])


async def _build_golden_items(
    goldens: list[GoldenAnswer],
) -> tuple[list[CalibrationItem], list[JudgeVerdict]]:
    """Same real-graph-then-judge pipeline as run_answer_eval.py's
    `_run_goldens` — the only difference is this keeps the raw answer text/
    contexts/verdict instead of collapsing them into a `QuestionOutcome`,
    since items.jsonl needs the text a human reads, not just a pass/fail."""
    embeddings = get_embeddings()
    answer_llm = make_llm()
    judge_llm = ChatOpenAI(model=_JUDGE_MODEL, temperature=0)
    tools = await make_mcp_tools()

    items: list[CalibrationItem] = []
    verdicts: list[JudgeVerdict] = []
    with Session(get_engine()) as session:
        for golden in goldens:
            config = tracing.run_config(tags=["judge-calibration", f"golden:{golden.id}"])
            try:
                answer = await ask_graph(
                    golden.question,
                    session=session,
                    embeddings=embeddings,
                    llm=answer_llm,
                    tools=tools,
                    config=config,
                )
                contexts = _contexts_for(answer, session)
            except CitationError as exc:
                # Same "counted, not skipped" treatment run_answer_eval.py
                # gives a CitationError — a refusal is itself a valid
                # calibration item, not a reason to drop one of the 10.
                answer = AnswerSchema(answer=f"CitationError: {exc}", citations=[])
                contexts = []

            verdict = judge(golden.question, answer, contexts, golden.reference, judge_llm)
            items.append(
                CalibrationItem(
                    id=golden.id,
                    source="golden",
                    question=golden.question,
                    answer=answer.answer,
                    contexts=contexts,
                    reference=golden.reference,
                )
            )
            verdicts.append(verdict)
    return items, verdicts


async def _build_pipeline_items(
    attacks: list[Attack], benign: list[BenignQuestion]
) -> tuple[list[CalibrationItem], list[JudgeVerdict]]:
    """Runs the deterministic 5 red-team + 5 benign items through the SAME
    full guard_in->retrieve->answer->guard_out graph evals/run_redteam.py
    uses (reused: build_graph/GraphContext/make_mcp_tools/make_classifier_
    llm), then runs the judge on whatever answer comes out — a blocked
    attack produces `REFUSAL_TEXT` (short, zero citations); a benign
    question produces an ordinary one-off answer. Neither shape appears in
    the golden set, which is the point of including them."""
    embeddings = get_embeddings()
    answer_llm = make_llm()
    classifier_llm = make_classifier_llm()
    judge_llm = ChatOpenAI(model=_JUDGE_MODEL, temperature=0)
    tools = await make_mcp_tools()

    requests: list[tuple[str, str, str]] = [
        *[("redteam", a.id, a.attack) for a in attacks],
        *[("benign", b.id, b.question) for b in benign],
    ]

    items: list[CalibrationItem] = []
    verdicts: list[JudgeVerdict] = []
    with Session(get_engine()) as session:
        for source, item_id, question in requests:
            graph = build_graph()
            context = GraphContext(
                session=session,
                embeddings=embeddings,
                llm=answer_llm,
                classifier=classifier_llm,
                tools=tools,
            )
            config = tracing.run_config(tags=["judge-calibration", f"{source}:{item_id}"])
            try:
                state = await graph.ainvoke({"question": question}, context=context, config=config)
                answer = state["answer"]
                contexts = _contexts_for(answer, session) if answer.citations else []
            except Exception as exc:  # noqa: BLE001 — an invariant break
                # (CitationError/OutputGuardError/ToolCallError) is itself
                # worth calibrating against (the judge still has to score
                # SOME text), so it's recorded as an item, never skipped.
                answer = AnswerSchema(answer=f"{type(exc).__name__}: {exc}", citations=[])
                contexts = []

            # No hand-written reference exists for these items — the judge
            # gets an empty <reference> block (evals/judge.py's
            # _build_messages), so "relevant" here really just means
            # "addresses the question", a known limitation of calibrating
            # off-golden-set items this way (see docs/handoffs note).
            verdict = judge(question, answer, contexts, "", judge_llm)
            items.append(
                CalibrationItem(
                    id=item_id,
                    source=source,
                    question=question,
                    answer=answer.answer,
                    contexts=contexts,
                    reference=None,
                )
            )
            verdicts.append(verdict)
    return items, verdicts


async def _build_all() -> tuple[list[CalibrationItem], list[JudgeVerdict]]:
    goldens = load_golden_answers()
    all_attacks = load_attacks()
    all_benign = load_benign()
    redteam_ids = _first_n_sorted_ids([a.id for a in all_attacks], _N_PER_FIXTURE)
    benign_ids = _first_n_sorted_ids([b.id for b in all_benign], _N_PER_FIXTURE)
    attacks_subset = [a for a in all_attacks if a.id in redteam_ids]
    benign_subset = [b for b in all_benign if b.id in benign_ids]

    golden_items, golden_verdicts = await _build_golden_items(goldens)
    pipeline_items, pipeline_verdicts = await _build_pipeline_items(attacks_subset, benign_subset)
    return golden_items + pipeline_items, golden_verdicts + pipeline_verdicts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    CALIBRATION_DIR.mkdir(exist_ok=True)
    items, verdicts = asyncio.run(_build_all())

    _write_jsonl(ITEMS_PATH, [asdict(item) for item in items])
    _write_jsonl(
        VERDICTS_PATH,
        [
            {
                "id": item.id,
                "faithful": v.faithful,
                "relevant": v.relevant,
                "reasoning": v.reasoning,
            }
            for item, v in zip(items, verdicts, strict=True)
        ],
    )
    _write_jsonl(
        LABELS_TEMPLATE_PATH,
        [{"id": item.id, "faithful": None, "relevant": None, "why": ""} for item in items],
    )

    n_golden = sum(1 for i in items if i.source == "golden")
    n_pipeline = len(items) - n_golden
    print(f"wrote {len(items)} items ({n_golden} golden, {n_pipeline} benign/redteam) to:")
    print(f"  {ITEMS_PATH}")
    print(f"  {VERDICTS_PATH}  (judge's own verdicts — do not read before labelling blind)")
    print(f"  {LABELS_TEMPLATE_PATH}")
    # Rough order-of-magnitude, not an invoice — same framing
    # run_answer_eval.py's _print_cost_note uses. 10 golden items: ~2 calls
    # each (answer+judge). 10 pipeline items: ~1-3 calls each (classifier,
    # maybe answer, judge). Order-of-magnitude a few cents, per ADR-0017/
    # ADR-0022's own measured per-call pricing.
    print("rough cost: ~40-50 LLM calls total, order-of-magnitude a few cents")


if __name__ == "__main__":
    main()
