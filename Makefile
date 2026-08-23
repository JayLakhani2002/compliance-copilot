# Makefile — shortcuts for the commands you'll run most while developing.
# Run `make <target>` from the repo root.

.PHONY: setup lint test

setup:  ## Install deps (incl. dev tools) and enable the pre-commit git hook.
	uv sync
	uv run pre-commit install

lint:  ## Lint + check formatting with ruff.
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the non-integration test suite.
	uv run pytest -m "not integration"
