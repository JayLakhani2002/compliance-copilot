# Progress

## Status: Phase 2 — Week 1 Day 3 DONE (PR #2 merged). Next: Day 4 embeddings + ingest-to-DB.
Project: Compliance Copilot (Option 1). Defaults: 3–4 h/day · 6 weeks · EU deploy. Repo: github.com/JayLakhani2002/compliance-copilot (private for now).

## Done
- Phase 0: 50 ads → docs/research/market_research.md, Option 1 chosen.
- Phase 1: docs/ARCHITECTURE.md, 12 ADRs (ADR-0009/0010 amended → Langfuse Cloud EU for v1.0), docs/CURRICULUM.md (30 days), repo scaffold (uv, ruff, pytest, CI, pre-commit, Python 3.12 pinned), teacher/builder two-screen workflow (TEACHER.md, docs/INBOX.md, docs/SETUP_SCREENS.md).
- Day 1 of curriculum = read ARCHITECTURE + ADRs with the teacher (Jay does this in VS Code).

## Next (builder)
- Week 1 Day 4: lesson 04 (embeddings, cosine, HNSW params, multilingual); `embeddings.py` (LangChain OpenAIEmbeddings text-embedding-3-small, 1536-d, batch), `ingest` CLI writes Document+Chunk rows + vectors (upsert by regulation+anchor); oversize-article split by paragraph keeping anchor; integration test: top-1 for "What is a high-risk AI system?" → AI Act Art. 6. Needs OPENAI_API_KEY in .env (Jay) — CI: mock embeddings for unit tests. Branch `feature/embeddings-ingest`.

## Done this week
- Day 2: compose, settings, db models, HNSW index, CI integration job (PR #1).
- Day 3: Cellar fetch + article/recital parser, fixtures, lazy engine, ADR-0012 licence verified (PR #2).

## Open questions
- Jay: install Docker (INBOX TASK) to run integration tests locally; CI covers it meanwhile.
