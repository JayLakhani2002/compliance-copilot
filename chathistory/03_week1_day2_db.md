# Handoff 03 — Week 1 Day 2: Postgres + pgvector (2026-08-23)

## Done
- Lesson docs/lessons/02_postgres_pgvector.md (builder-drafted; teacher delivers in VS Code).
- Coder: docker-compose.yml (pgvector/pgvector:pg16, loopback), settings.py (pydantic-settings), db.py (Document, Chunk, Vector(1536), HNSW cosine index, init_db), cli.py init-db, unit + integration tests, CI integration job with pgvector service container. Handoff: docs/handoffs/week1-day2-db/coder.md.
- Reviewer: APPROVE, 2 minors (loopback binding fixed; HNSW pull-forward noted in lesson). docs/handoffs/week1-day2-db/reviewer.md.
- PR #1 squash-merged to develop; CI: unit 6 passed, integration passed against real pgvector.
- Deps pinned: sqlalchemy 2.0.52, psycopg 3.3.4, pgvector 0.5.0, pydantic-settings 2.15.0.

## Decisions
- Docker not installed locally → integration tests skip without DATABASE_URL, run in CI. INBOX TASK for Jay to install Docker/OrbStack.

## Open issues
- None blocking.

## Exact next step
Day 3: `git checkout -b feature/ingest-eurlex` from develop. (1) WebFetch EUR-Lex legal notice / reuse page, record URL + quote in ADR-0012 (R1). (2) Lesson 03 (legal-text chunking by article/recital vs token windows). (3) Coder: `src/compliance_copilot/ingest/eurlex.py` — fetch AI Act (32024R1689) + GDPR (32016R0679) HTML from EUR-Lex, parse into article/recital chunks with metadata; save raw HTML under data/raw (gitignored) with a small fixture sample under tests/fixtures for unit tests; unit test asserts article counts (AI Act 113 articles, GDPR 99 articles — verify from the source) and a sample article's text. No embeddings yet (Day 4). (4) Reviewer → PR → merge.
Check docs/INBOX.md first.

## Commands to resume
cat CLAUDE.md docs/PROGRESS.md chathistory/03_week1_day2_db.md
