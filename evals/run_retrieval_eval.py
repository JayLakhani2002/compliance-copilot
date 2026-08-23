# evals/run_retrieval_eval.py — Day 5 retrieval eval runner (ADR-0005,
# docs/GLOSSARY.md's "golden set"/hit@k/MRR terms, lesson 05's two-variant
# experiment). Loads evals/golden_retrieval.jsonl, runs ONE of the two
# retriever variants lesson 05 designed against it, and reports hit@5 + MRR
# overall and per category, plus a per-question table for failure analysis
# — printing WHAT broke, not just THAT it broke (lesson 05's "failure
# analysis over the score" point).
#
# Real embeddings only, via get_embeddings() (ADR-0004) — costs a few cents
# per run against OPENAI_API_KEY, so run it locally, once or twice, not in
# a loop:
#     set -a && . ./.env && set +a
#     uv run python -m evals.run_retrieval_eval --variant plain
#     uv run python -m evals.run_retrieval_eval --variant articles
#
# `EVAL_HIT5_MIN` (default "0"): exits nonzero if overall hit@5 falls below
# this threshold — the "eval as CI gate" mechanism ADR-0005 calls for. 0 by
# default so a first/local run never fails before a real threshold is
# chosen (lesson 05: "wired when the numbers stabilise").
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from compliance_copilot.db import get_engine
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.retriever import retrieve

GOLDEN_PATH = Path(__file__).parent / "golden_retrieval.jsonl"

# The two retriever variants lesson 05 asks to compare, keyed by --variant:
# "plain" = variant A (all kinds compete for top-k); "articles" = variant B
# (kind-aware — recitals excluded, retriever.py's default `kinds`).
VARIANTS: dict[str, tuple[str, ...]] = {
    "plain": ("article", "recital"),
    "articles": ("article",),
}

K = 5  # hit@5 / rank-within-top-5, per lesson 05's chosen k.


@dataclass
class GoldenEntry:
    id: str
    question: str
    regulation: str  # "ai_act" | "gdpr" | "any"
    expected_anchors: list[str]
    category: str


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenEntry]:
    entries: list[GoldenEntry] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(GoldenEntry(**json.loads(line)))
    return entries


@dataclass
class QuestionResult:
    entry: GoldenEntry
    rank: int | None  # 1-indexed rank of the first expected-anchor hit within top-K; None = MISS


def run_variant(
    entries: list[GoldenEntry], kinds: tuple[str, ...], session: Session, embeddings
) -> list[QuestionResult]:
    results = []
    for entry in entries:
        # "any" (the golden set's cross-regulation entries) -> regulation=None,
        # i.e. search both regulations — retriever.py's contract for that arg.
        regulation = None if entry.regulation == "any" else entry.regulation
        hits = retrieve(
            entry.question,
            k=K,
            kinds=kinds,
            regulation=regulation,
            session=session,
            embeddings=embeddings,
        )
        rank = next(
            (i for i, chunk in enumerate(hits, start=1) if chunk.anchor in entry.expected_anchors),
            None,
        )
        results.append(QuestionResult(entry=entry, rank=rank))
    return results


def _hit_at_k(results: list[QuestionResult]) -> float:
    return sum(1 for r in results if r.rank is not None) / len(results) if results else 0.0


def _mrr(results: list[QuestionResult]) -> float:
    return sum((1 / r.rank) if r.rank else 0.0 for r in results) / len(results) if results else 0.0


def print_report(results: list[QuestionResult], variant: str) -> float:
    print(f"\n=== retrieval eval — variant={variant!r} (k={K}) ===\n")
    print(f"{'id':<6} {'category':<11} {'expected':<20} {'result':<10} question")
    for r in results:
        expected = ",".join(r.entry.expected_anchors)
        outcome = f"rank {r.rank}" if r.rank else "MISS"
        question = r.entry.question[:70]
        print(f"{r.entry.id:<6} {r.entry.category:<11} {expected:<20} {outcome:<10} {question}")

    print(f"\n--- summary (variant={variant!r}) ---")
    overall_hit5 = _hit_at_k(results)
    overall_mrr = _mrr(results)
    print(f"overall:      hit@{K}={overall_hit5:.3f}  MRR={overall_mrr:.3f}  n={len(results)}")

    for cat in sorted({r.entry.category for r in results}):
        cat_results = [r for r in results if r.entry.category == cat]
        print(
            f"  {cat:<11} hit@{K}={_hit_at_k(cat_results):.3f}  "
            f"MRR={_mrr(cat_results):.3f}  n={len(cat_results)}"
        )

    return overall_hit5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()

    entries = load_golden()
    embeddings = get_embeddings()
    with Session(get_engine()) as session:
        results = run_variant(entries, VARIANTS[args.variant], session, embeddings)

    overall_hit5 = print_report(results, args.variant)

    # str, not float, default — os.environ values are always strings; "0"
    # parses the same as 0 would, and keeps this one type all the way through.
    threshold = float(os.environ.get("EVAL_HIT5_MIN", "0"))
    if overall_hit5 < threshold:
        raise SystemExit(
            f"hit@{K}={overall_hit5:.3f} below EVAL_HIT5_MIN={threshold:.3f} — failing as a CI gate"
        )


if __name__ == "__main__":
    main()
