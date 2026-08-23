# Progress

## Status: Phase 2 — Week 1 Day 2 DONE (PR #1 merged, CI green incl. pgvector integration). Next: Day 3 ingestion.
Project: Compliance Copilot (Option 1). Defaults: 3–4 h/day · 6 weeks · EU deploy. Repo: github.com/JayLakhani2002/compliance-copilot (private for now).

## Done
- Phase 0: 50 ads → docs/research/market_research.md, Option 1 chosen.
- Phase 1: docs/ARCHITECTURE.md, 12 ADRs (ADR-0009/0010 amended → Langfuse Cloud EU for v1.0), docs/CURRICULUM.md (30 days), repo scaffold (uv, ruff, pytest, CI, pre-commit, Python 3.12 pinned), teacher/builder two-screen workflow (TEACHER.md, docs/INBOX.md, docs/SETUP_SCREENS.md).
- Day 1 of curriculum = read ARCHITECTURE + ADRs with the teacher (Jay does this in VS Code).

## Next (builder)
- Week 1 Day 3: verify EUR-Lex reuse notice → ADR-0012; `ingest/eurlex.py` fetch AI Act + GDPR, article-level chunks with metadata; unit test on article counts. Branch `feature/ingest-eurlex`. Lesson 03 first.

## Done this week
- Day 2: compose, settings, db models, HNSW index, CI integration job (PR #1).

## Open questions
- Jay: install Docker (INBOX TASK) to run integration tests locally; CI covers it meanwhile.
