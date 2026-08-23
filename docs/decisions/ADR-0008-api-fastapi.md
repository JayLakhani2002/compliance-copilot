# ADR-0008: API — FastAPI + Pydantic v2, SSE streaming, API-key auth, rate limiting

## Status
Accepted. 2026-08-23.

## Context
The graph (ADR-0001) needs an HTTP surface: a way for a user to POST a question and receive a streamed answer, plus operator endpoints to resume a paused (HITL-interrupted) run. FastAPI is named in 14% of sampled ads — table-stakes rather than a differentiator, but the framework choice still needs to support streaming (SSE — Server-Sent Events, a one-way push protocol over plain HTTP, simpler than WebSockets for the "stream tokens to the client" use case) and structured request/response validation cleanly.

## Options considered
1. **FastAPI + Pydantic v2**, with SSE for streaming, an API-key auth dependency, and **slowapi** for rate limiting (ADR referenced, not restated — see below), plus structured JSON logging.
2. **LangServe** (LangChain's own FastAPI-based serving layer for LangChain runnables/chains) — would have been a natural fit for a LangChain-heavy stack, but LangChain's own project direction has moved away from LangServe as the recommended deployment path (superseded by LangGraph Platform / self-managed FastAPI patterns) — building new code on it risks building on a deprecated direction.
3. **Flask / Django** — both viable general-purpose Python web frameworks, but neither has FastAPI's native async support (needed for streaming LLM responses without blocking) or its automatic OpenAPI schema generation from Pydantic models, which this project uses for both documentation and structured-output validation (ADR-0006's `guard_out` schema check reuses the same Pydantic-model discipline as the API layer).

## Decision
**FastAPI**, request/response models in **Pydantic v2**, **SSE** for streaming the answer (and status updates, e.g. "under review" when a run hits the HITL interrupt path), an **API-key auth** dependency (checked on every request before a graph run starts — cheapest possible rejection point for unauthenticated traffic), **slowapi** for rate limiting, and **structured JSON logging** (so logs are machine-parseable for the "no PII in logs" hard rule — structured fields make it possible to assert in tests that a given field, e.g. the raw question, is never logged in cleartext).

**Verified detail:** FastAPI's current PyPI release is 0.141.1 (verified 2026-08-23) — FastAPI has not had a 1.0 release; its versioning stays in the 0.x range by design while remaining stable/production-used, which is expected and not a signal of instability. `slowapi` (PyPI, version 0.1.10, verified 2026-08-23) is a rate limiter adapted from `flask-limiter` for Starlette/FastAPI, confirmed via Context7 (`laurents/slowapi`) as actively documenting async-endpoint support, which this project needs since the graph run itself is async.

## Why not the others
- **LangServe**: rejected specifically because it represents a deprecated *direction* within the LangChain ecosystem, not because it lacks features — building a new project's serving layer on a path the maintainers have moved away from is the kind of choice that ages badly fast, and this project's brief explicitly names "deprecated direction" as the rejection reason to record.
- **Flask/Django**: rejected on async-streaming and schema-validation grounds specifically relevant to this project's needs (streaming LLM output, reusing Pydantic models between the API layer and the guardrail schema checks) — not a general claim that Flask/Django are worse frameworks.

## Security & cost implications
- **Security:** the API-key check and slowapi's rate limiting are both applied **before** a graph run starts (`docs/ARCHITECTURE.md` §6, boundary #1) — this is deliberate: rejecting unauthenticated or rate-limited traffic before it can trigger an LLM call (ADR-0002) is both a security control and a cost control. Structured JSON logging is the mechanism that makes the "no PII in logs" hard rule testable rather than just asserted.
- **Cost:** rate limiting directly caps the worst-case LLM spend from a single client (ADR-0002's per-question cost × requests/minute × rate-limit window) — without it, a misbehaving or malicious client could drive unbounded Anthropic/embedding API spend.

## How to reverse
The API layer is a thin wrapper around `graph.astream(...)` calls — the graph itself (ADR-0001) has no FastAPI dependency, so swapping the web framework means rewriting the HTTP-handling layer (routes, auth dependency, SSE response handling) without touching the graph, guardrails, or tool code underneath it.

## References
- FastAPI, PyPI: 0.141.1 — https://pypi.org/project/fastapi/ (verified 2026-08-23)
- `slowapi`, PyPI: 0.1.10 — https://pypi.org/project/slowapi/ (verified 2026-08-23); docs confirming async support: Context7 `/laurents/slowapi`
- Market data on FastAPI naming frequency: `docs/research/market_research.md`
