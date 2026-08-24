# Compliance Copilot

Agentic RAG over the EU AI Act and GDPR: an LLM agent that answers compliance
questions grounded in the actual regulatory text, with retrieval/answer
quality measured by evals and guardrails enforced against prompt injection
and hallucination — all gated by CI so a broken change can't merge.

**Status:** Phase 1 scaffolding — repo structure, tooling, and CI are in
place; no application code yet. See `docs/ARCHITECTURE.md` for the system
design and `docs/research/market_research.md` for why this project was chosen.

## How to run

```bash
make setup   # uv sync + install the pre-commit git hook (one-time)
cp .env.example .env   # fill in real secrets locally; DATABASE_URL default matches docker-compose.yml
make db-up   # start Postgres+pgvector via Docker Compose (needs Docker installed)
make db-init # create the vector extension, tables, and HNSW index
make test    # uv run pytest -m "not integration"
```

DB integration tests (`tests/test_db_integration.py`) need a real Postgres —
they're skipped automatically unless `DATABASE_URL` is set, and run in
GitHub CI's `integration` job against a `pgvector/pgvector:pg16` service
container regardless of what's set up locally.

## Run the API

```bash
# set API_KEY in .env first (see .env.example)
make api  # uvicorn on http://127.0.0.1:8000, auto-reload
curl -sN -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" -d '{"question":"When is an AI system high-risk?"}'
```

Set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` to enable tracing (Langfuse Cloud, EU) — unset by default, so the app runs with zero tracing until you do.
