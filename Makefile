# Makefile — shortcuts for the commands you'll run most while developing.
# Run `make <target>` from the repo root.

.PHONY: setup lint test db-up db-init api eval-cache quality-gate

setup:  ## Install deps (incl. dev tools) and enable the pre-commit git hook.
	uv sync
	uv run pre-commit install

lint:  ## Lint + check formatting with ruff.
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the non-integration test suite.
	uv run pytest -m "not integration"

db-up:  ## Start local Postgres+pgvector (needs Docker).
	docker compose up -d postgres

db-init:  ## Create the vector extension, tables, and HNSW index.
	uv run python -m compliance_copilot.cli init-db

api:  ## Run the FastAPI app locally (needs API_KEY set in .env).
	uv run uvicorn compliance_copilot.api:app --host 127.0.0.1 --port 8000 --reload

eval-cache:  ## Refresh evals/embeddings_cache/*.jsonl with real vectors (needs OPENAI_API_KEY).
	uv run python -m evals.cache_embeddings

quality-gate:  ## Retrieval quality gate against the LOCAL DB, cached query embeddings (ADR-0017).
	uv run python -m evals.run_retrieval_eval --embeddings cached --hit5-min 0.93 --mrr-min 0.80
