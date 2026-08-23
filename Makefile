# Makefile — shortcuts for the commands you'll run most while developing.
# Run `make <target>` from the repo root.

.PHONY: setup lint test db-up db-init

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
