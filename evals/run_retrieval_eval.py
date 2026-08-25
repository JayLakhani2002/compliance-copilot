# evals/run_retrieval_eval.py — Day 5 retrieval eval runner (ADR-0013,
# docs/GLOSSARY.md's "golden set"/hit@k/MRR terms, lesson 05's two-variant
# experiment). Loads evals/golden_retrieval.jsonl, runs ONE of the two
# retriever variants lesson 05 designed against it, and reports hit@5 + MRR
# overall and per category, plus a per-question table for failure analysis
# — printing WHAT broke, not just THAT it broke (lesson 05's "failure
# analysis over the score" point).
#
# Golden schema note (reviewer round 1): `expected_anchors` entries are
# "<regulation>:<anchor>" strings (e.g. "ai_act:art_6"), not bare anchors —
# `anchor_id` collides across regulations (both AI Act and GDPR have an
# art_14, with unrelated content), so scoring must match on the pair, not
# the anchor alone. Every entry is qualified this way, including
# single-regulation ones, so the scorer never branches on entry.regulation —
# one format, one comparison, always unambiguous (see run_variant() below).
#
# Real embeddings by default, via get_embeddings() (ADR-0004) — costs a few
# cents per run against OPENAI_API_KEY, so run it locally, once or twice,
# not in a loop:
#     set -a && . ./.env && set +a
#     uv run python -m evals.run_retrieval_eval --variant plain
#     uv run python -m evals.run_retrieval_eval --variant articles
#
# `--embeddings cached` (ADR-0017): swaps in CachedEmbeddings, which reads
# the 30 golden questions' REAL vectors from evals/embeddings_cache/queries.jsonl
# instead of calling OpenAI — this is what CI's `quality-gate` job uses, so
# the retrieval gate runs on every PR with no API key and no network call.
# The corpus side needs no cache-awareness here: retrieve() only ever calls
# embed_query() on the question, never embed_documents() — the corpus's own
# vectors are already sitting in the DB from whatever ingested them.
#
# `--hit5-min`/`--mrr-min` (also settable via `EVAL_HIT5_MIN`/`EVAL_MRR_MIN`
# env vars, for backward compatibility with the original mechanism): exits
# nonzero if overall hit@5 or MRR falls below these thresholds — the "eval
# as CI gate" mechanism ADR-0013 introduced and ADR-0017 extends with an MRR
# gate. Both default to "0" so a first/local run never fails before a real
# threshold is chosen (lesson 05: "wired when the numbers stabilise").
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from compliance_copilot.cached_embeddings import CachedEmbeddings
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
    expected_anchors: list[str]  # "<regulation>:<anchor>", e.g. "ai_act:art_6"
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
        # Match on (regulation, anchor), not anchor alone — anchor_id is not
        # globally unique (art_14 exists in both ai_act and gdpr with
        # unrelated content), so an anchor-only match could score a hit
        # against the wrong regulation's chunk for regulation="any"
        # (cross-regulation) entries, where retrieve() searches both.
        rank = next(
            (
                i
                for i, chunk in enumerate(hits, start=1)
                if f"{chunk.regulation}:{chunk.anchor}" in entry.expected_anchors
            ),
            None,
        )
        results.append(QuestionResult(entry=entry, rank=rank))
    return results


def _hit_at_k(results: list[QuestionResult]) -> float:
    return sum(1 for r in results if r.rank is not None) / len(results) if results else 0.0


def _mrr(results: list[QuestionResult]) -> float:
    return sum((1 / r.rank) if r.rank else 0.0 for r in results) / len(results) if results else 0.0


def print_report(results: list[QuestionResult], variant: str) -> tuple[float, float]:
    print(f"\n=== retrieval eval — variant={variant!r} (k={K}) ===\n")
    print(f"{'id':<6} {'category':<11} {'expected':<26} {'result':<10} question")
    for r in results:
        expected = ",".join(r.entry.expected_anchors)
        outcome = f"rank {r.rank}" if r.rank else "MISS"
        question = r.entry.question[:70]
        print(f"{r.entry.id:<6} {r.entry.category:<11} {expected:<26} {outcome:<10} {question}")

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

    return overall_hit5, overall_mrr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default "articles" (not required=True, as before): ADR-0013's decision
    # is articles-first, so that's the variant the CI gate and any casual
    # local run should exercise unless a comparison run explicitly asks for
    # "plain".
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="articles")
    parser.add_argument(
        "--embeddings",
        choices=("cached", "real"),
        default="real",
        help="'cached' uses CachedEmbeddings (ADR-0017) — no API key, no network.",
    )
    # str, not float, env default — os.environ values are always strings;
    # "0" parses the same as 0 would, and keeps this one type all the way
    # through. A --hit5-min/--mrr-min flag on the command line always wins
    # over the env var (CI passes these explicitly).
    parser.add_argument(
        "--hit5-min", type=float, default=float(os.environ.get("EVAL_HIT5_MIN", "0"))
    )
    parser.add_argument("--mrr-min", type=float, default=float(os.environ.get("EVAL_MRR_MIN", "0")))
    args = parser.parse_args()

    entries = load_golden()
    embeddings = CachedEmbeddings() if args.embeddings == "cached" else get_embeddings()
    with Session(get_engine()) as session:
        results = run_variant(entries, VARIANTS[args.variant], session, embeddings)

    overall_hit5, overall_mrr = print_report(results, args.variant)

    failures = []
    if overall_hit5 < args.hit5_min:
        failures.append(f"hit@{K}={overall_hit5:.3f} below --hit5-min={args.hit5_min:.3f}")
    if overall_mrr < args.mrr_min:
        failures.append(f"MRR={overall_mrr:.3f} below --mrr-min={args.mrr_min:.3f}")
    if failures:
        raise SystemExit("; ".join(failures) + " — failing as a CI gate")


if __name__ == "__main__":
    main()
