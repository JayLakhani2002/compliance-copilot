# Handoff 02 — Phase 1 architecture (2026-08-23)

## Done
- Jay picked Option 1 (Compliance Copilot). Two-screen workflow set up: TEACHER.md (VS Code panel), builder (terminal), channel docs/INBOX.md, lessons in docs/lessons/.
- Writer agent: docs/ARCHITECTURE.md (4 mermaid diagrams) + docs/decisions/ADR-0001..0012, versions verified via Context7 (handoff: docs/handoffs/phase1-architecture/writer.md). Flags from verification: mcp SDK v2 renamed FastMCP→MCPServer; Ragas 0.4 deprecates evaluate() (use experiments API); Bedrock EU inference-profile IDs = deploy-time check; EUR-Lex reuse notice to verify Day 3.
- Planner amendments: ADR-0009 → Langfuse Cloud EU region (eu-west-1) for v1.0, self-host = Week-6 stretch; ADR-0010 → 4 GB VPS.
- Coder agent: repo scaffold (uv, ruff 0.16.4, pytest 9.1.1, pre-commit, GH Actions setup-uv, Makefile, README stub) commit 21e7fd4; Python 3.12 pinned 25f24a7; Phase 1 docs 6cd359f; tag v0.0-phase1. Jay created remote (private) and pushed main/develop.
- docs/CURRICULUM.md: 6 weeks × 5 days, tags v0.1..v1.0, 6 LinkedIn posts.

## Decisions
- Python 3.12 pinned (.python-version). Repo private for now.
- Langfuse Cloud EU instead of self-host for v1.0.

## Open issues
- None blocking. Jay should paste TEACHER.md into VS Code Claude panel and do Day 1 (read ARCHITECTURE + ADRs with teacher).

## Exact next step (builder)
Phase 2, Week 1 Day 2: `git checkout -b feature/db-pgvector` from develop → teacher lesson docs/lessons/02_postgres_pgvector.md → coder: docker-compose.yml (pgvector/pgvector:pg16), src/compliance_copilot/db.py (SQLAlchemy engine from DATABASE_URL), migration for documents/chunks(embedding vector(1536)), tests/test_db.py (integration marker) → reviewer → PR → merge develop.
Check docs/INBOX.md before starting.

## Commands to resume
cat CLAUDE.md docs/PROGRESS.md chathistory/02_phase1_architecture.md docs/CURRICULUM.md
