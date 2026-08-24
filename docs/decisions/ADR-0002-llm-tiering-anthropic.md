# ADR-0002: LLM tiering — Anthropic Claude via LangChain `ChatAnthropic`

## Status
Accepted. 2026-08-23.

## Context
The graph (ADR-0001) makes several LLM calls per request with different accuracy/latency/cost needs: cheap classification (router), a cheap pass/fail judgment (critic), and one call that actually has to write a good, well-cited answer (the final response). Using one model tier for everything either overspends (using the best model for a yes/no classification) or underspends (using a cheap model to write the answer users actually read). The market research shows Anthropic named in 14% of ads and OpenAI in 14% — roughly tied — so the choice here is a project decision, not a market-forced one; the deciding factors are EU-residency options and LangChain interface compatibility for future swaps.

## Options considered
1. **Anthropic Claude, tiered, via LangChain's `ChatAnthropic`** — Haiku (fastest/cheapest tier) for routing/classification and for the LLM-as-judge critic; Sonnet (mid tier, but the strongest per-dollar option for actually writing an answer) for the final drafted response. Accessed through LangChain's model interface so the concrete provider is swappable later.
2. **Single-provider OpenAI lock-in** — use one OpenAI model (e.g., GPT-4.1-class) for everything. Simpler to reason about, but no tiering (either overpays on cheap calls or underpays on the important one) and ties both the orchestration layer and the model layer to one vendor.
3. **Local open-source LLM** (e.g., a self-hosted Llama/Mistral-class model) — best data-residency story in theory (nothing leaves the VPS), but the compute needed to self-host something competent enough to draft legally-adjacent answers is real GPU spend and operational burden that doesn't fit a 6-week, 3–4h/day portfolio build. Rejected on ops-cost grounds specifically, not on capability grounds.

## Decision
**Anthropic Claude**, called through LangChain's `ChatAnthropic` (package `langchain-anthropic`), with two tiers:
- **Haiku** (model id `claude-haiku-4-5`) for the `router` node (classify the question) and the `critic` node (LLM-as-judge confidence scoring, ADR-0005).
- **Sonnet** (model id `claude-sonnet-5`) for the `answer` node (the response the user actually reads).

**Production path, documented but not the day-1 default:** AWS Bedrock in `eu-central-1` (Frankfurt), for EU data residency on the inference call itself (see `docs/ARCHITECTURE.md` §8 for why this distinction matters — the direct Anthropic API is not itself EU-region-pinned the way Bedrock `eu-central-1` is). Both paths go through the same LangChain `ChatAnthropic`-family interface, so switching between them is a client-construction change, not an application-logic change.

**Model id verification note:** current model ids and per-tier pricing were checked against the live Anthropic model table (via the `claude-api` skill, cached 2026-06-24, cross-checked 2026-08-23): Haiku `claude-haiku-4-5` ($1.00/$5.00 per MTok in/out), Sonnet `claude-sonnet-5` ($3.00/$15.00 per MTok, $2.00/$10.00 intro pricing through 2026-08-31). On Bedrock, model ids take an `anthropic.` prefix (e.g. `anthropic.claude-sonnet-5`) via Anthropic's Bedrock client conventions — the **exact cross-region inference-profile id for `eu-central-1`** (Bedrock sometimes requires an inference-profile id like `eu.anthropic.claude-sonnet-5-...` rather than the bare model id for cross-region routing) was **not confirmed** in this research pass and must be checked against the AWS Bedrock console/API at deploy time, since Bedrock's regional model rollout lags Anthropic's own release cadence and isn't covered by the sources checked here.

## Why not the others
- **Single-provider OpenAI lock-in**: rejected because it removes the cost/quality tiering this project explicitly wants to demonstrate (using a cheap model for cheap decisions is itself a "senior engineer" signal called out in the market research's read of the ads).
- **Local open-source LLM**: rejected purely on ops-cost/timeline grounds for a 6-week solo build — self-hosting a model good enough to draft legal-adjacent answers competitively with Sonnet needs GPU infrastructure this project's Hetzner CPU VPS (ADR-0010) doesn't have, and provisioning GPU infra is out of scope. This is explicitly *not* a capability judgment against open models — it's a "wrong project, wrong timeline" call, worth stating plainly rather than hand-waving.

## Security & cost implications
- **Security:** the router/answer/critic calls all receive user-derived text (the question, and for `answer`, retrieved regulation chunks). PII redaction (ADR-0006) must run before any of these calls, since text sent to Anthropic/Bedrock leaves the trust boundary described in `docs/ARCHITECTURE.md` §6. API keys (or AWS credentials, on the Bedrock path) belong only in environment variables / `.env`, per the project's hard rule — never in code or logs.
- **Cost:** tiering is the cost lever here — see `docs/ARCHITECTURE.md` §9 for the per-question ballpark (well under $0.02/question at current prices, dominated by the Sonnet call). Prompt caching (a LangChain/Anthropic feature that reuses a cached prefix of a prompt across calls to avoid re-paying for stable content like the system prompt or tool descriptions) is available on the Anthropic API and should be applied to the `answer` node's system prompt and tool schema, since those are stable across requests even though retrieved chunks vary per query.

## How to reverse
Because the model calls go through LangChain's chat-model interface (`ChatAnthropic`, and equivalently `ChatOpenAI` from `langchain-openai` or any other LangChain chat-model integration), swapping providers or tiers is a constructor-argument change in one place (wherever the graph builds its LLM clients), not a rewrite of node logic — node functions call `.invoke()`/`.ainvoke()` on whatever model object they're given. Moving from direct API to Bedrock is the same kind of swap: a different `ChatAnthropic` construction (or Bedrock-specific client) passed into the same node code.

## References
- `langchain-anthropic`, PyPI: 1.6.1 — https://pypi.org/project/langchain-anthropic/ (verified 2026-08-23)
- `langchain-openai` (alternative/reference for interface parity), PyPI: 1.6.0 — https://pypi.org/project/langchain-openai/ (verified 2026-08-23)
- Current Claude model ids and pricing table: `claude-api` skill (cached 2026-06-24; Haiku `claude-haiku-4-5`, Sonnet `claude-sonnet-5`), cross-checked 2026-08-23
- Bedrock model-id prefix convention (`anthropic.<model-id>`): `claude-api` skill, "Provider Clients" section (verified 2026-08-23) — exact `eu-central-1` inference-profile id **not verified**, open item
