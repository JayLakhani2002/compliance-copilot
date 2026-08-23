# ADR-0013 — Retrieval strategy: articles-first, recitals as secondary context

**Status:** accepted 2026-08-23

## Context
The corpus contains both articles (binding law) and recitals (explanatory preamble). On the 30-question golden set (`evals/golden_retrieval.jsonl`), plain cosine retrieval over all chunks let recitals crowd out the legally authoritative articles — e.g. "What is a high-risk AI system?" ranked Recital 52 first and Article 6 ~11th.

## Options considered
1. **Plain cosine over all chunks** — measured: hit@5 0.633, MRR 0.332.
2. **Articles-only retrieval** (recitals excluded from primary search, fetched separately later as supporting context for the answer node) — measured: hit@5 0.867, MRR 0.696.
3. Hybrid BM25 + vector, or score boosting by kind — not measured; more machinery for an unproven gain.

## Decision
Option 2. `retrieve()` defaults to `kinds=("article",)`. Recitals stay in the index and become supporting context in the agent graph (Week 2+), never the primary citation.

## Why not the others
Option 1 loses on both metrics by a wide margin — recitals paraphrase concepts in natural language and win cosine similarity while being non-binding. Option 3 adds infrastructure before a measured need; revisit only if the eval shows a gap articles-only can't close.

## Security & cost implications
None new; same index, one extra WHERE clause (kind filter). Slightly fewer candidate rows per query.

## How to reverse
One default parameter. The eval harness re-measures any change: `python -m evals.run_retrieval_eval --variant plain|articles`.

## Known residual gap (tracked, not hidden)
4/30 definitional questions miss under both variants (deployer, GDPR controller, GDPR consent, GPAI model): single-term definitions are diluted inside multi-definition mega-articles (AI Act Art. 3, GDPR Art. 4). Planned fix: definition-level sub-chunking of those articles, then re-measure. Numbers live in the eval output.

## References
Golden set + runner in `evals/`; measured 2026-08-23 with text-embedding-3-small on the full 576-chunk corpus.

## Amendment — 2026-08-23 (definition sub-chunking)
The residual gap is closed. Definition articles (AI Act Art. 3, GDPR Art. 4) are now split at definition boundaries (regex `(?=\(\d{1,3}\)\s+‘)`, verified 68/68 + 26/26 matches, zero false positives corpus-wide, lossless re-join). Corpus 576 → 587 chunks. Re-measured on the same golden set:
| variant | hit@5 | MRR |
|---|---|---|
| plain | 0.633 → 0.900 | 0.332 → 0.509 |
| **articles-only (default)** | 0.867 → **1.000** | 0.696 → **0.881** |
All 4 previously-missing definition questions now hit under the default retriever. Articles-first remains the decision; the golden set will grow with agent-era failures, so 1.000 is a snapshot, not a claim.
