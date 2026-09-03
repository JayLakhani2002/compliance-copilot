# Makefile — shortcuts for the commands you'll run most while developing.
# Run `make <target>` from the repo root.

.PHONY: setup lint test db-up db-init api mcp eval-cache quality-gate redteam redteam-fast calibration calibration-build cost-report backup-now deploy-validate

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
	# --no-server-header (ADR-0030, Day 25 security review): don't advertise
	# "uvicorn" + its version in every response's Server header — free
	# reconnaissance for an attacker fingerprinting this service for a
	# known CVE, for zero functional benefit.
	uv run uvicorn compliance_copilot.api:app --host 127.0.0.1 --port 8000 --reload --no-server-header

mcp:  ## Run the MCP server (stdio by default — MCP_TRANSPORT=streamable-http to switch).
	uv run python -m compliance_copilot.mcp_server

eval-cache:  ## Refresh evals/embeddings_cache/*.jsonl with real vectors (needs OPENAI_API_KEY).
	uv run python -m evals.cache_embeddings

quality-gate:  ## Retrieval quality gate against the LOCAL DB, cached query embeddings (ADR-0017).
	uv run python -m evals.run_retrieval_eval --embeddings cached --hit5-min 0.93 --mrr-min 0.80

redteam:  ## Full red-team ASR/FPR gate (needs OPENAI_API_KEY, real DB, costs cents — ADR-0022).
	uv run python -m evals.run_redteam --subset all --asr-max 0.05 --fpr-max 0.10

redteam-fast:  ## No-key heuristics-subset red-team check (zero cost, zero network — ADR-0022).
	uv run python -m evals.run_redteam --subset heuristics --asr-max 0.05

calibration-build:  ## Rebuild evals/calibration/*.jsonl from the real pipeline (needs OPENAI_API_KEY, DB, costs cents — ADR-0027).
	uv run python -m evals.build_calibration_set

calibration:  ## Judge-vs-human agreement report — reads existing evals/calibration/*.jsonl, zero cost (ADR-0027).
	uv run python -m evals.run_judge_calibration

cost-report:  ## Measured €/question over the golden set — full call shape, real spend (ADR-0029).
	uv run python -m evals.run_cost_report

backup-now:  ## Run an on-demand pg_dump into the prod `backups` volume (ADR-0032).
	docker compose -f docker-compose.prod.yml exec -T backup sh -c \
		'pg_dump -h postgres -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -F c -f "/backups/$$(date +%F-%H%M).dump"'

deploy-validate:  ## Dry-validate deploy.sh + the Terraform stub (no server/credentials needed, ADR-0033).
	bash -n deploy/deploy.sh
	docker run --rm -v "$(CURDIR)/infra/terraform":/w -w /w hashicorp/terraform:latest fmt -check
	docker run --rm -v "$(CURDIR)/infra/terraform":/w -w /w hashicorp/terraform:latest init -backend=false
	docker run --rm -v "$(CURDIR)/infra/terraform":/w -w /w hashicorp/terraform:latest validate
