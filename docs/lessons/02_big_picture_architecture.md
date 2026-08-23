# Lesson 02 — The Big Picture: architecture + every tech decision (Day 1)

**Sources:** `docs/ARCHITECTURE.md`, `docs/decisions/ADR-0001-agent-framework-langgraph.md`, `pyproject.toml`, `.github/workflows/ci.yml`. Only ADR-0001 exists so far; ADR-0002…0010 are referenced by ARCHITECTURE.md but the builder hasn't written them yet.

## What it is
An **agentic RAG** system: instead of one LLM call answering from memory, a **LangGraph state machine** takes a question about the EU AI Act / GDPR, guards it, retrieves the actual law text from **Postgres+pgvector** via **MCP tools**, drafts an answer with citations, has a second model **judge** the draft, and pauses for a **human** if confidence is low. Served by **FastAPI** (SSE streaming), traced in **Langfuse**, deployed on a **Hetzner** VPS in Germany.

## Why we need it here
Legal answers must be *grounded and auditable*. A plain chat model hallucinates article numbers; here every claim must cite a chunk retrieved *in this run*, and the whole path (which node ran, what it saw, what it cost) is inspectable in a trace.

## Why not the alternatives (one real one each)
| Choice | Rejected alternative | Why ours wins here |
|---|---|---|
| LangGraph | CrewAI | CrewAI trades explicit control for demo speed; we need auditable order, durable state, `interrupt()` (ADR-0001) |
| Postgres+pgvector | Pinecone | One DB for chunks *and* LangGraph checkpoints *and* eval results; EU-hosted on our box; no extra SaaS |
| MCP tools | plain LangChain tools | Retrieval decoupled from the agent framework — swappable, and MCP is named in 20% of job ads |
| FastAPI+SSE | WebSockets | One-way token streaming needs no bidirectional channel; SSE is plain HTTP, simpler to proxy and test |
| Langfuse (self-host) | LangSmith (cloud) | Traces contain user text → keep them on our EU box; self-hosting is itself a portfolio signal |
| Haiku router/judge + Sonnet answerer | Sonnet everywhere | Classification/judging is cheap-model work; the per-question cost driver is the one Sonnet call (~<$0.02/q total) |

## How a senior thinks about it
- **Failure modes first:** DB down → fail *closed* (no un-grounded answer); LLM 5xx → error state, never a partial answer; no citation → refuse (ARCHITECTURE §7).
- **Security:** everything downstream trusts `guard_in`, so `guard_in` bugs are the top-severity class (§6). PII is redacted *before* it can reach checkpoints or traces.
- **Cost:** fixed VPS ~€20–30/mo dominates at portfolio scale; per-question LLM cost is cents. Model tiering + prompt caching keep it there.

## Analogy
A law firm: reception (guard_in) screens the request, a paralegal (Haiku router) decides which books to pull, a clerk (MCP tools) fetches the exact articles, an associate (Sonnet) drafts the memo with citations, a reviewing partner (Haiku critic) scores it, and anything shaky goes to the *human* senior partner before it leaves the building (interrupt → HITL).

## One interview question
*"Your RAG system returned a confident answer citing an article that doesn't exist. Walk me through every layer that should have caught it, and which one you'd fix first."*
(Expected path: retrieval grounding → structured output schema → critic/judge → `guard_out` citation-exists check against this run's chunks → eval gate in CI catching the regression class. Fix first: `guard_out`, it's deterministic and cheap.)

## Check questions asked to Jay
1. Why does HITL (`interrupt()`) stop working if Postgres is down — what's the mechanical reason? (ADR-0001: `Command(resume=...)` requires a checkpointer.)
2. Why chunk the corpus by article instead of fixed 512-token windows?
