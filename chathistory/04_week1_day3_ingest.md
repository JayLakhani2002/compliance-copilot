# Handoff 04 — Week 1 Day 3: EUR-Lex ingestion (2026-08-23)

## Done
- R1 licence verification: eur-lex legal notice read via browser (scripted HTTP gets empty 202). Reuse under Decision 2011/833/EU; only OJ authentic → app needs disclaimer. Recorded in ADR-0012.
- Source decision: Publications Office Cellar `GET https://publications.europa.eu/resource/celex/{CELEX}` with `Accept: application/xhtml+xml` → AI Act 113 art/180 rct, GDPR 99/173 (ids art_N, rct_N; classes oj-*, eli-*).
- Lesson 03 docs/lessons/03_corpus_and_chunking.md.
- Coder: src/compliance_copilot/ingest/eurlex.py (httpx + selectolax), ArticleChunk model, atomic cache, EurLexFetchError, count sanity (articles + recitals), CLI `ingest --dry-run`, fixture tests/fixtures/eurlex_sample.xhtml (23 KB), unit + gated network tests; db.py lazy get_engine(). Handoffs: docs/handoffs/week1-day3-ingest/{coder,reviewer}.md.
- Reviewer: APPROVE; 1 major (recital count) + 3 minors fixed in round 1. Article stats: min 149 / median 1862 / max 17079 chars.
- PR #2 squash-merged to develop; CI green (unit 15 passed, integration passed).

## Decisions
- selectolax chosen over bs4 (speed/simplicity; verified via Context7).
- Network tests gated by RUN_NETWORK_TESTS=1, not in CI (determinism).
- Oversize articles (Art. 3 = 17k chars) → Day 4 will split by paragraph keeping anchor id (lesson 03 design).

## Open issues
- Jay: install Docker (INBOX), add OPENAI_API_KEY/ANTHROPIC_API_KEY to .env (INBOX), answer pace question (INBOX).

## Exact next step (builder)
Day 4: `git checkout -b feature/embeddings-ingest` from develop → lesson 04 → coder: embeddings.py (LangChain OpenAIEmbeddings text-embedding-3-small, verify via Context7), chunk splitter for oversize articles (paragraph-level, anchor kept, size cap ~1500 tokens), ingest writes Document/Chunk rows + vectors with upsert, CLI `ingest` (non-dry-run), unit tests with a fake Embeddings class, integration test (needs DATABASE_URL + OPENAI_API_KEY; CI: skip if no key, or use fake embeddings with deterministic vectors to at least exercise the upsert path) → reviewer → PR → merge → tag v0.1 after Day 5.
Check docs/INBOX.md first.

## Commands to resume
cat CLAUDE.md docs/PROGRESS.md chathistory/04_week1_day3_ingest.md
