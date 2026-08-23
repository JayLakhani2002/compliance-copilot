# ADR-0012: Corpus & licensing — EUR-Lex HTML/XML, chunked by article/recital

## Status
Accepted. 2026-08-23.

## Context
The system needs an actual, correctly-licensed source text for the EU AI Act (Regulation (EU) 2024/1689) and GDPR (Regulation (EU) 2016/679), and a chunking strategy (how the text is split into retrievable units) that matches how a legal document is actually structured and cited — a "recital" and an "article" are the natural citable units of an EU regulation, not arbitrary fixed-length text windows.

## Options considered
1. **EUR-Lex HTML/XML** — the EU's own official law portal, the canonical source for the consolidated/in-force text of both regulations, reusable under the EU's own reuse policy (EUR-Lex content is, per the EU's stated reuse policy, reusable including for commercial purposes provided the source is acknowledged, unless otherwise stated for specific documents — this project is non-commercial/portfolio use, which is comfortably within that policy, and the source (EUR-Lex, with celex numbers) should be acknowledged in the app/README regardless).
2. Any secondary/mirrored source (e.g., a third-party legal-text aggregator) — rejected outright rather than seriously evaluated: a compliance-focused RAG project citing a *non-canonical* copy of the regulations it claims to be authoritative about would undermine the project's own premise, independent of any licensing question.

## Decision
**EUR-Lex HTML/XML** as the sole source for both the AI Act and GDPR text. **Chunking strategy: by article/recital**, not by fixed token/character windows — each chunk carries **metadata** (`regulation` — "AI Act" or "GDPR"; `article` — the article or recital number; `title` — the article's heading, where the source provides one). This directly serves two other ADRs: ADR-0003's storage schema (metadata columns alongside the embedding) and ADR-0006's citation-must-exist guardrail check (a citation is only valid if it names a `regulation` + `article` pair that actually exists in the ingested metadata, not just a plausible-looking reference).

**Note on what was and wasn't independently re-verified in this research pass:** this ADR's licensing/reuse-policy framing was **not** re-fetched from a live EUR-Lex reuse-policy page during this pass (the research focus was library/API verification, per the task scope) — it restates the reuse-policy premise already given in the project brief. Before ingestion code is written, the actual EUR-Lex reuse/legal-notice page should be fetched and read directly (not summarized from memory) and that URL added to this ADR's References, since "the EU's reuse policy allows it" is exactly the kind of claim this project's own standards (`CLAUDE.md`: "Verify every library API against Context7 MCP + official docs before writing it. Never guess.") would require checking before relying on it for something the project publishes.

## Why not the others
- **Secondary/mirrored sources**: not seriously considered — using anything other than the canonical EUR-Lex text for a project whose entire value proposition is "accurately cites the actual regulation" would be self-undermining regardless of any licensing convenience a mirror might offer.

## Security & cost implications
- **Security:** the corpus itself is public legal text — no PII, no confidentiality concern on the source data. The only security-relevant part of ingestion is treating the *ingestion job's* network access to EUR-Lex as an outbound-only, offline, unauthenticated fetch (`docs/ARCHITECTURE.md` §6, boundary #7 — lowest-risk boundary in the system, explicitly).
- **Cost:** no licensing cost (public EU legal text, EUR-Lex access is free); ingestion is a one-time/rare job (re-run only if a regulation is amended/consolidated again), so it's a negligible, non-recurring cost against ADR-0004's embedding-cost model.

## How to reverse
The chunking-by-article/recital strategy and metadata schema are corpus-agnostic in structure (any legal-text source with articles/recitals could be ingested the same way) — swapping to a different regulation or source document would mean writing a different ingestion parser for that source's HTML/XML structure, while the downstream schema (chunk + `regulation`/`article`/`title` metadata), vector store (ADR-0003), and citation-check guardrail (ADR-0006) stay the same. This is explicitly named in the project brief as a portfolio-flexibility property ("domain is swappable... without changing architecture").

## References
- Project brief statement of scope and reuse premise: task brief (this ADR set's originating instructions) — "EU reuse policy allows it"
- **Open item, not yet independently verified**: the live EUR-Lex reuse/legal-notice page should be fetched and its exact URL/terms recorded here before ingestion code ships — flagged in `docs/handoffs/phase1-architecture/writer.md`
- Regulation identifiers: EU AI Act = Regulation (EU) 2024/1689; GDPR = Regulation (EU) 2016/679 (as stated in the project brief)
