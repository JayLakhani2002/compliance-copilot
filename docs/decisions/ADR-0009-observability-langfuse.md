# ADR-0009: Observability — Langfuse, self-hosted via Docker Compose (EU)

## Status
Accepted. 2026-08-23.

## Context
Every LLM/tool call in the graph needs to be traceable — for debugging (why did this answer come out wrong), for cost tracking (ADR-0002/0004's per-call spend), and for the eval pipeline's score history (ADR-0005 logs results to "Langfuse datasets/scores"). Market research shows "observability" named in 16% of ads and Langfuse specifically in 2% (LangSmith also at 4%) — a smaller signal than agents/RAG/evals, but the project's EU-residency narrative (`docs/ARCHITECTURE.md` §8) makes the *hosting model* of the observability tool as important as the tool choice itself.

## Options considered
1. **Langfuse, self-hosted via Docker Compose**, integrated through LangChain's callback mechanism (LangChain/LangGraph support attaching a callback handler that automatically emits trace events for every LLM/tool call, without manual instrumentation at each call site) — EU-hosted (on the same Hetzner VPS, ADR-0010), open source.
2. **LangSmith** (LangChain's own hosted observability/eval product) — the most natural single-vendor fit given the stack is LangChain/LangGraph, but it is a **US-hosted SaaS**, which is the same conflict already identified in ADR-0005 for LangSmith evals — using it here would mean *every* LLM call's trace (including user questions and model answers) leaves the EU by default, directly undermining the residency story this project is built around.
3. **Raw OpenTelemetry only** (instrument the app with generic OTel spans, no dedicated LLM-observability product) — vendor-neutral and always an option to add later, but rejected as the *primary* mechanism for this build specifically because it would mean building cost/latency dashboards, trace search, and score correlation from scratch rather than getting them from a purpose-built LLM-observability tool — reasonable as a future addition (e.g., for infra-level metrics alongside Langfuse's LLM-specific ones), not as a replacement.

## Decision
**Langfuse, self-hosted**, deployed via **Docker Compose** on the same Hetzner VPS as the rest of the stack (ADR-0010), integrated via **LangChain's callback integration** so traces are emitted automatically from graph/node execution without manual per-call instrumentation. Cost + latency dashboards come from Langfuse's own UI; eval results (ADR-0005) are pushed to **Langfuse datasets/scores**.

**Verified detail — this is the single most important correction this research pass produced, and it directly qualifies ADR-0003's "one database" framing:** Langfuse's official self-hosting documentation (`langfuse.com/self-hosting/...`, checked via Context7 against `/websites/langfuse_self-hosting`) shows the self-host Docker Compose stack requires, **in addition to** a Postgres instance for Langfuse's own application data:
- **ClickHouse** — Langfuse's analytics/trace storage backend (a column-store database purpose-built for the high-volume, append-heavy write pattern of LLM trace data — not a fit for Postgres at Langfuse's expected trace volume, which is presumably why Langfuse itself made this choice).
- **Redis** — used as Langfuse's cache/queue layer (confirmed via a documented `REDIS_HOST`/`REDIS_PORT`/`REDIS_AUTH` config and a `docker run redis --requirepass ... --maxmemory-policy noeviction` example).
- **An S3-compatible blob store** (MinIO in the self-host/local example) — for large trace payloads that don't belong in either Postgres or ClickHouse directly.

This means the deployed stack has (at minimum) **five** stateful services once Langfuse is added: this project's own `postgres` (ADR-0003), plus Langfuse's `clickhouse`, `redis`, and `minio`, plus Langfuse's own web/worker containers. `langfuse.com/self-hosting/deployment/docker-compose` confirms the standard self-host flow is `docker compose up` against a compose file that includes these services. **ADR-0003's "the one database" claim is scoped to this project's own application data (documents, chunks/embeddings, checkpoints, eval results) — it does not describe the full deployed system once observability is included.** This has real consequences: VPS sizing (`docs/ARCHITECTURE.md` §9) needs enough RAM for ClickHouse specifically (it is not a lightweight service), and the "one database" framing in any portfolio write-up should be phrased carefully (e.g., "the application's own data lives in one Postgres instance; the observability stack brings its own storage, as any self-hosted LLM-observability product does") rather than implied to cover the whole system.

## Why not the others
- **LangSmith**: rejected on the EU-residency conflict stated above — this is the clearest case in the whole ADR set of "a technically convenient choice that would undercut the project's own stated thesis."
- **Raw OpenTelemetry only**: not rejected as *wrong*, deferred as *not sufficient alone* for this build's timeline — Langfuse gives LLM-specific dashboards (cost per model tier, latency per node, score trends) out of the box; OTel would require building that layer from scratch, which isn't a good use of a 6-week solo build's time when a purpose-built tool exists and self-hosts cleanly.

## Security & cost implications
- **Security:** trace payloads can contain the user's question and the model's answer (`docs/ARCHITECTURE.md` §6, boundary #6) — `guard_in`'s PII redaction must run before the traced content is generated, not just before the LLM call, so redacted (not raw) text is what ends up in ClickHouse. Langfuse's own credentials (Postgres/ClickHouse/Redis/MinIO passwords) belong in `.env`, same as every other secret in this project, and should not default to the example passwords shown in Langfuse's own quick-start docs (`clickhouse`/`myredissecret`/`miniosecret` are documentation placeholders, not production values).
- **Cost:** self-hosting Langfuse avoids per-seat/per-trace SaaS pricing, but the true cost is the **extra VPS resources** the ClickHouse/Redis/MinIO stack needs (see `docs/ARCHITECTURE.md` §9's VPS sizing note) — this is a real cost that a naive "Langfuse is free and open source" framing would miss. Trace-volume growth in ClickHouse over months is an unbounded-growth risk without a retention/TTL policy, which is not configured by default in the self-host compose file and is flagged as an operational follow-up.

## How to reverse
LangChain's callback interface is the integration point — swapping Langfuse for a different observability backend (including OpenTelemetry, or a different hosted product) means writing/attaching a different callback handler; graph/node code has no direct Langfuse dependency. The bigger practical cost of reversing this decision is operational (decommissioning four running containers — Langfuse web/worker, ClickHouse, Redis, MinIO — and migrating or discarding accumulated trace history), not architectural.

## References
- Langfuse Python SDK, PyPI: `langfuse` 4.14.4 — https://pypi.org/project/langfuse/ (verified 2026-08-23)
- Self-host service requirements (ClickHouse, Redis, blob storage) and `docker compose up` flow: Context7 `/websites/langfuse_self-hosting`, sources `deployment/infrastructure/clickhouse`, `deployment/infrastructure/cache`, `infrastructure/blobstorage`, `deployment/docker-compose` (verified 2026-08-23)
- Market data on observability/Langfuse naming frequency: `docs/research/market_research.md`

## Planner amendment — 2026-08-23
**Decision:** Weeks 2–5 use **Langfuse Cloud, EU region** (hosted on AWS `eu-west-1`, Ireland — verified at https://langfuse.com/docs/data-security-privacy, 2026-08-23). Self-hosting (ClickHouse + Redis + MinIO + web/worker) moves to a **Week-6 stretch** with its own compose file.
**Why:** one person, 3–4 h/day; the five-service self-host stack costs RAM and ops time that belong in the agent/evals work. EU-region cloud keeps the data-residency story honest ("traces stay in the EU"). The LangChain callback integration is identical, so nothing in the graph changes when we self-host later.
**Consequences:** ADR-0003 "one database" stays true for the whole deployed system until the stretch goal. ADR-0010 VPS sizing drops to a 4 GB-class box for v1.0. Redaction must still run before tracing (unchanged).
