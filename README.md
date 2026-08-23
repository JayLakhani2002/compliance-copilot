# Compliance Copilot

Agentic RAG over the EU AI Act and GDPR: an LLM agent that answers compliance
questions grounded in the actual regulatory text, with retrieval/answer
quality measured by evals and guardrails enforced against prompt injection
and hallucination — all gated by CI so a broken change can't merge.

**Status:** Phase 1 scaffolding — repo structure, tooling, and CI are in
place; no application code yet. See `docs/ARCHITECTURE.md` for the system
design and `docs/research/market_research.md` for why this project was chosen.

## Running tests

```bash
make setup   # uv sync + install the pre-commit git hook (one-time)
make test    # uv run pytest -m "not integration"
```
