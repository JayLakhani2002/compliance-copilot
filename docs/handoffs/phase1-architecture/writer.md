# Phase 1 — Architecture handoff (writer/researcher)

## What I verified (Context7 + PyPI + WebFetch, all 2026-08-23)

| Library | Version verified | Notes |
|---|---|---|
| langgraph | 1.2.11 | StateGraph, `interrupt()`/`Command(resume=...)` semantics confirmed against test suite |
| langgraph-checkpoint-postgres | 3.1.2 | **Separate package** from `langgraph` core — see flag below |
| langchain | 1.3.16 | |
| langchain-anthropic | 1.6.1 | |
| langchain-openai | 1.6.0 | |
| langchain-postgres | 0.0.17 | Current API is `PGEngine`+`PGVectorStore`, async-first; older `PGVector` class is legacy |
| langchain-mcp-adapters | 0.3.2 | `MultiServerMCPClient` + `get_tools()` + LangGraph `ToolNode`/`tools_condition` pattern confirmed |
| mcp (Python SDK) | 2.0.0 | **Major v2 rename** — see flag below |
| ragas | 0.4.3 | **`evaluate()` deprecated** — see flag below |
| langfuse (python sdk) | 4.14.4 | Self-host stack needs more than Postgres — see flag below |
| fastapi | 0.141.1 | |
| presidio-analyzer | 2.2.364 | High patch number is a CI build-number scheme, confirmed legitimate (name/summary checked) |
| slowapi | 0.1.10 | Async-endpoint support confirmed |
| uv | 0.12.5 | |
| Claude models | claude-haiku-4-5, claude-sonnet-5 | Via `claude-api` skill (cached 2026-06-24, cross-checked today) |

Doc URLs and exact source files are recorded per-library in each ADR's References section.

## Decisions the docs contradicted or needed correcting (in order of importance)

1. **ADR-0009/ADR-0003 — "Postgres is the one database" is only true for application data.** Langfuse's official self-hosting docs (`langfuse.com/self-hosting/...`) require ClickHouse (trace analytics), Redis (cache/queue), and an S3-compatible blob store (MinIO) as hard dependencies of self-hosted Langfuse — not something this project chose to add. I did **not** change the ADR-0003 decision (Postgres remains the single database for *this project's own* data: documents, chunks/embeddings, checkpoints, eval results) — I scoped the claim precisely in both ADR-0003 and ADR-0009, and flagged the VPS-sizing consequence in ADR-0010 (needs to be an 8GB-RAM-class box, not the cheapest tier, mainly because of ClickHouse). **This is the one finding I'd want the planner to explicitly sign off on**, since it changes the container count (5+ stateful services, not 2) and the cost model versus what a literal reading of the original brief ("Postgres... as the one database") implies.

2. **ADR-0007 — `mcp` SDK v2 renamed `FastMCP` to `MCPServer`.** The brief says "official `mcp` SDK, FastMCP." The current SDK (2.0.0, a major revision) exposes this as `mcp.server.mcpserver.MCPServer`, with source comments literally marking it "(formerly FastMCP)." I wrote ADR-0007 against the current name/import path and flagged that the v1 `mcp.server.fastmcp.FastMCP` import should be treated as unavailable unless separately confirmed otherwise. Did not change the underlying decision (still the official `mcp` SDK, still the same decorator-based ergonomics) — just corrected the concrete class/import name so Phase 2 code doesn't get written against a stale path.

3. **ADR-0005 — Ragas's `evaluate()` function is deprecated.** Current Ragas (0.4.3) has deprecated the classic `evaluate(dataset, metrics=[...])` entry point in favor of a collections-based metrics API (`ragas.metrics.collections.*` + `ragas.llms.llm_factory`) and an `@experiment` decorator. I wrote ADR-0005 to build the eval harness against the current (non-deprecated) API. Did not change the tool choice (still Ragas) — just the concrete API surface to build against.

4. **ADR-0002/ADR-0004 — Bedrock `eu-central-1` exact model/inference-profile IDs not fully confirmed.** I confirmed the general Bedrock model-id convention (`anthropic.` prefix) but could not confirm the exact cross-region inference-profile ID format for `eu-central-1` specifically for `claude-sonnet-5`/`claude-haiku-4-5`, since Bedrock's regional rollout lags Anthropic's own releases and wasn't covered by the sources I checked. Flagged as an open item in both ADRs — needs a live AWS Bedrock console/API check at actual deploy time, not before.

5. **ADR-0012 — EUR-Lex reuse-policy claim not independently re-verified.** I kept the brief's stated premise ("EU reuse policy allows it") but did not fetch the live EUR-Lex reuse/legal-notice page myself in this pass (scope was library/API verification). Flagged in ADR-0012 as something to check with a live fetch before ingestion code ships, given the project's own "never guess" rule.

## Open questions for the planner

1. **Sign off on the VPS-sizing consequence of finding #1** — does an 8GB-RAM-class Hetzner box (vs. a smaller/cheaper tier) still fit the project's cost tolerance? This wasn't specified in the brief and I made a judgment call in ADR-0010/ARCHITECTURE.md §9 rather than picking a number unilaterally without flagging it.
2. **Should Phase 2 (coding) budget time to independently fetch the EUR-Lex reuse-policy page** before ingestion code is written, per finding #5, or is the brief's stated premise sufficient to proceed on and revisit only if a real legal concern surfaces?
3. **Bedrock inference-profile IDs (finding #4)** — fine to leave as a deploy-time check, or does the planner want this resolved before Phase 2 starts (e.g., because it affects how the `ChatAnthropic`/Bedrock client construction code gets written)?

## Files written

- `docs/ARCHITECTURE.md`
- `docs/decisions/ADR-0001-agent-framework-langgraph.md` through `docs/decisions/ADR-0012-corpus-licensing-eurlex.md` (12 files)
- `docs/handoffs/phase1-architecture/writer.md` (this file)
