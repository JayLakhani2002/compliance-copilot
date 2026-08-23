# Curriculum — Compliance Copilot, 6 weeks × 5 days × 3–4 h

Each day = **Learn** (teacher lesson, `docs/lessons/`) → **Build** (feature branch, coder agent) → **Test** (pytest green) → **Post** (LinkedIn, weekly). Milestones are git tags. If a day slips, the week's milestone moves, not the scope.

## Week 1 — Foundations, ingestion, retrieval, first eval  → tag `v0.1` · Post #1
| Day | Learn | Build | Test passes when |
|---|---|---|---|
| 1 | Repo hygiene: uv, ruff, pytest, branches, CI as a gate. Why ADRs. | Scaffold (done in Phase 1). Read ARCHITECTURE + ADRs with teacher. | `make test` green in CI |
| 2 | Docker Compose; Postgres + pgvector; why one DB (ADR-0003). SQLAlchemy basics. | `docker-compose.yml` (postgres/pgvector), `db.py` connection + migration for `documents`, `chunks(embedding vector)` | integration test connects, creates tables |
| 3 | Corpus + chunking strategy for legal text (by article/recital, metadata). Why not fixed-size windows. | **First** fetch + record the EUR-Lex legal-notice/reuse URL in ADR-0012 (R1). Then `ingest/eurlex.py`: fetch AI Act + GDPR, parse to article-level chunks with metadata | unit test: known article count, sample article text |
| 4 | Embeddings: what a vector is, cosine vs L2, HNSW index, multilingual models (ADR-0004). | `embeddings.py` + `ingest` CLI writes chunks+vectors; HNSW index | integration: top-1 for "What is a high-risk AI system?" is Art. 6 AI Act |
| 5 | Evals, part 1: golden datasets, retrieval metrics (hit@k, MRR), why CI must block. | `evals/golden_retrieval.jsonl` (30 Q→article pairs), `tests/evals/test_retrieval.py` with threshold | hit@5 ≥ 0.8 in CI |
**Post #1:** "Week 1: I turned the EU AI Act + GDPR into a vector index and made CI fail if retrieval quality drops. Here's why I chunk by article, not by tokens."

## Week 2 — LangGraph single agent, streaming API, tracing, RAG evals  → tag `v0.2` · Post #2
| Day | Learn | Build | Test |
|---|---|---|---|
| 6 | LangGraph core: State, nodes, edges, compile, invoke vs stream. Why a graph not a chain. | `graph/state.py`, linear graph retrieve → answer with citations (Sonnet) | unit (mocked LLM): state flows, citations list non-empty |
| 7 | Prompting for grounded answers; structured output with Pydantic; citation format. | `prompts/`, `AnswerSchema`, answer node enforces schema | unit: invalid citation → validation error |
| 8 | FastAPI + SSE streaming; Pydantic request/response; API-key auth (ADR-0008). | `api/main.py` `/ask` streaming, `/health` | test client: 200, stream chunks, 401 without key |
| 9 | Observability: traces/spans, Langfuse self-host, cost per request (ADR-0009). | Langfuse in compose, callback wired, cost logged | integration: trace appears with token counts |
| 10 | Evals, part 2: Ragas faithfulness/relevancy; LLM-as-judge rubric; thresholds. | `evals/golden_qa.jsonl` (25 Q/A), `tests/evals/test_rag_quality.py` | faithfulness ≥ 0.8 gate in CI (integration marker, runs on develop) |
**Post #2:** "Week 2: a LangGraph agent that answers AI-Act questions with article citations, streams over SSE, is traced in Langfuse, and can't merge if faithfulness < 0.8."

## Week 3 — Guardrails and red-teaming  → tag `v0.3` · Post #3
| Day | Learn | Build | Test |
|---|---|---|---|
| 11 | Threat model for LLM apps: prompt injection (direct/indirect via retrieved docs), data exfil, PII. OWASP LLM Top 10. | `docs/THREAT_MODEL.md`; `guards/injection.py` heuristics | unit: 20 injection strings flagged, 20 benign pass |
| 12 | Classifier guard with Haiku; cost/latency trade-off of LLM guards; fail-closed vs fail-open. | `guards/classifier.py`; `guard_in` node; refusal path | unit (mocked) + 1 integration |
| 13 | PII: Presidio; redaction vs blocking; GDPR data-minimisation; no PII in logs. | `guards/pii.py`, log scrubbing | unit: emails/IBAN/names redacted |
| 14 | Output guards: schema, citation-exists check, scope check ("only AI Act/GDPR"). | `guard_out` node | unit: hallucinated article id → blocked |
| 15 | Evals, part 3: red-team dataset; attack success rate as a CI metric. | `evals/redteam.jsonl` (40 attacks), `tests/evals/test_redteam.py` | ASR ≤ 5% gate |
**Post #3:** "Week 3: I attacked my own agent 40 ways. Here's the layered guardrail design that holds, what it costs in latency, and the one attack that still gets through."

## Week 4 — MCP tools + multi-agent + durable state  → tag `v0.4` · Post #4
| Day | Learn | Build | Test |
|---|---|---|---|
| 16 | MCP: what/why, server vs client, tools/resources, transport. Why not plain tools (ADR-0007). | `mcp_server/` with `search_regulation`, `get_article`, `cite` (FastMCP) | MCP client test lists 3 tools, calls each |
| 17 | `langchain-mcp-adapters`; tool-calling loop; tool errors. | graph consumes MCP tools; retrieve node → tool calls | unit (mocked) trajectory includes tool call |
| 18 | Multi-agent patterns: router/supervisor, specialist subgraphs, critic. When NOT to go multi-agent. | `router` (Haiku: AI-Act / GDPR / both / out-of-scope), `critic` node scoring answer | unit: router labels 10 questions correctly |
| 19 | Persistence: checkpointers, threads, memory; Postgres checkpointer; resuming. | PostgresSaver wired; `/ask` takes `thread_id`; follow-up questions work | integration: 2-turn conversation keeps context |
| 20 | Human-in-the-loop: `interrupt`, Command(resume); when a critic should escalate. | low-confidence → interrupt; `/resume` endpoint | integration: interrupt raised and resumed |
**Post #4:** "Week 4: router → retriever → answerer → critic, tools over MCP, state in Postgres, and the agent pauses for a human when it isn't sure. Architecture diagram inside."

## Week 5 — Eval depth, hardening, cost  → tag `v0.5` · Post #5
| Day | Learn | Build | Test |
|---|---|---|---|
| 21 | Trajectory evals: asserting paths, tool choice, loop limits. | `tests/evals/test_trajectory.py` | router/critic path assertions |
| 22 | Judge calibration: agreement with human labels; bias of LLM judges. | 20 human-labelled items; agreement report in `docs/EVALS.md` | κ reported, threshold documented |
| 23 | Rate limiting, timeouts, retries, fallbacks (LLM timeout → retrieval-only answer). | slowapi, tenacity, fallback path | unit: timeout → degraded answer, not 500 |
| 24 | Cost engineering: token budgets, model tiering numbers, caching. | cost dashboard in Langfuse; prompt caching where supported | `docs/EVALS.md` has €/100 questions |
| 25 | Security review day: secrets, CORS, headers, dependency audit, logs. | reviewer agent pass; fixes | `pip-audit`/`uv` audit clean |
**Post #5:** "Week 5: how I measure an agent — retrieval, faithfulness, red-team ASR, trajectory, judge calibration — and what it costs per 100 questions."

## Week 6 — Ship  → tag `v1.0` · Post #6 (launch)
| Day | Learn | Build | Test |
|---|---|---|---|
| 26 | Production compose: Caddy TLS, healthchecks, restart policies, backups (ADR-0010). | `docker-compose.prod.yml`, Caddyfile | `docker compose config` valid; /health 200 locally |
| 27 | Hetzner deploy; secrets handling on a VPS; EU residency checklist. | deploy script; live URL | smoke test against live URL |
| 28 | IaC basics: Terraform stub for AWS eu-central-1 (stretch). | `infra/terraform/` plan-only | `terraform validate` |
| 29 | Writing a case-study README; what recruiters read first. | README rewrite with diagram, numbers, decisions | reviewed by teacher |
| 30 | Mock interview: 10 system-design + 10 code questions; demo script. | 3-min demo video script; final post | Jay explains the system back |
**Post #6:** "I built and deployed an EU AI-Act/GDPR compliance agent in 6 weeks — LangGraph multi-agent, MCP, guardrails, eval-gated CI, Langfuse, EU-hosted. Full write-up, repo, numbers."

## LinkedIn cadence rule
One post per tag (6 total) + optional short mid-week "TIL" posts. Every post: 1 diagram or screenshot, 1 number, 1 decision explained, link to repo.
