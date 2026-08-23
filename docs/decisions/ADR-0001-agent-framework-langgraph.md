# ADR-0001: Agent framework — LangGraph

## Status
Accepted. 2026-08-23.

## Context
The agent needs to run a multi-step pipeline (classify → retrieve → answer → check → maybe pause for a human) with **explicit control over what runs next**, state that **survives a process restart** (so a paused run can be resumed hours later), and a way to **pause mid-run for human approval** (HITL — human-in-the-loop). It also needs to be debuggable: given a bad answer, an engineer should be able to see exactly which node produced which intermediate value.

Market signal (`docs/research/market_research.md`): "agents" appear in 58% of sampled ads, "multi-agent" in 30%, and **LangGraph is the single most-named agent framework at 22%** — ahead of CrewAI, AutoGen, and the OpenAI Agents SDK in this sample. LangChain itself (the ecosystem/toolkit) is named in 28%.

## Options considered
1. **LangGraph** — a graph/state-machine framework: you define nodes (Python functions) and edges (including conditional edges) over a shared, typed state object, and compile the graph to something you can `.invoke()`/`.stream()`. Ships a "checkpointer" abstraction for durable state and an `interrupt()`/`Command(resume=...)` primitive for pausing/resuming a run.
2. **CrewAI** — a "role-based crew" framework (agents with roles/goals, tasks, a crew that orchestrates them). Popular for quick multi-agent demos; less explicit control over exact execution order; state persistence and HITL are less first-class than LangGraph's checkpointer.
3. **AutoGen** (Microsoft) — conversation-driven multi-agent framework (agents "talk" to each other in a chat loop). Good for open-ended agent conversations; harder to get a deterministic, auditable pipeline out of, which matters for a compliance/citation use case.
4. **OpenAI Agents SDK** — OpenAI's own agent framework (handoffs, guardrails, tracing built in). Ties the project's control-flow layer to one model vendor's SDK, which conflicts with ADR-0002's swappable-provider goal, and is less named in the target job market than LangGraph.
5. **Plain LangChain `AgentExecutor`** (the older LangChain agent loop, not a graph) — simplest to start with, but the loop is a black box (you don't control exactly which step runs next), no first-class durable-state/interrupt story, and LangChain itself has moved newer agent-building guidance toward LangGraph for anything beyond a single tool-calling loop.

## Decision
**LangGraph**, using `StateGraph` (the graph builder), a **Postgres-backed checkpointer** (`langgraph-checkpoint-postgres`, providing `PostgresSaver` / `AsyncPostgresSaver`) for durable state, `interrupt()` + `Command(resume=...)` for the HITL pause/resume path (ADR described in `docs/ARCHITECTURE.md` §4), and a subgraph for retrieval if it grows complex enough to warrant its own compiled graph.

**Verified detail that changes how this gets built:** the Postgres checkpointer is **not** part of the `langgraph` core package — it's a separate package, `langgraph-checkpoint-postgres` (verified on PyPI, version 3.1.2, 2026-08-23), imported as `from langgraph.checkpoint.postgres import PostgresSaver` (sync) or `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` (async). Both need `langgraph-checkpoint-postgres` installed alongside `langgraph`. Confirmed via Context7 against `langchain-ai/langgraph`, source `libs/checkpoint-postgres/README.md`.

`interrupt()` behavior verified against `libs/langgraph/tests/test_time_travel.py` and `libs/prebuilt/README.md`: calling `interrupt(value)` inside a node halts the graph and returns `{"__interrupt__": [...]}` from `.invoke()`/`.stream()`; resuming requires the **same `thread_id`** in `config["configurable"]` and calling `graph.invoke(Command(resume=<value>), config)`. A checkpointer is **required** for `Command(resume=...)` to work at all — without one, LangGraph raises `RuntimeError: Cannot use Command(resume=...) without checkpointer`. This directly informs ADR-0003 (why Postgres has to be present before HITL can work) and the "DB down" failure mode in `docs/ARCHITECTURE.md` §7.

## Why not the others
- **CrewAI / AutoGen**: both trade explicit control for faster demo-building. This project's differentiator (per the market research) is showing *durable state + an auditable graph*, not showing "agents that talk to each other" — LangGraph's explicit node/edge model is the better fit and the better teaching tool.
- **OpenAI Agents SDK**: would lock control-flow to OpenAI even though ADR-0002 puts Anthropic in the driver's seat for the LLM calls themselves; mixing "OpenAI orchestrates, Anthropic answers" adds an unnecessary second SDK surface with its own tracing/guardrail concepts to reconcile against Langfuse (ADR-0009) and the custom guardrails (ADR-0006).
- **Plain `AgentExecutor`**: too coarse — no way to express "pause after the critic node specifically, not after any tool call" without hand-rolling exactly what LangGraph already provides.

## Security & cost implications
- **Security:** the checkpointer persists full graph state (including the user's question and any retrieved regulation text) to Postgres — this is a trust-boundary concern (see `docs/ARCHITECTURE.md` §6): PII redaction in `guard_in` must run *before* state reaches a node whose output gets checkpointed, or PII ends up durably stored. Checkpointed state should be treated with the same sensitivity as application logs.
- **Cost:** LangGraph itself adds no per-call cost (it's a local Python library, not a hosted service) — cost is driven entirely by the LLM calls inside the nodes (ADR-0002) and by Postgres storage growth from checkpoints over time (checkpoints accumulate per thread; a retention/cleanup policy is an operational follow-up, not solved by LangGraph itself).

## How to reverse
The graph's nodes are plain Python functions with a defined state schema — the *business logic* (guardrail checks, retrieval calls, prompt construction) is not LangGraph-specific and can be lifted into a different orchestrator (e.g., a hand-rolled state machine, or CrewAI/AutoGen if requirements change) with moderate effort. The two LangGraph-specific things to replace would be (1) the checkpointer/`interrupt()` mechanism — its durable-pause behavior would need reimplementing — and (2) `StateGraph`'s conditional-edge routing, which would become explicit `if/else` in a manual loop. Retrieval is already isolated behind MCP tools (ADR-0007), so it is not entangled with the choice of orchestrator.

## References
- LangGraph (Python), PyPI: `langgraph` 1.2.11 — https://pypi.org/project/langgraph/ (verified 2026-08-23)
- `langgraph-checkpoint-postgres`, PyPI: 3.1.2 — https://pypi.org/project/langgraph-checkpoint-postgres/ (verified 2026-08-23)
- Checkpointer install/usage and `interrupt()`/`Command(resume=...)` semantics: Context7 `/langchain-ai/langgraph`, sources `libs/checkpoint-postgres/README.md`, `libs/prebuilt/README.md`, `libs/langgraph/tests/test_time_travel.py` (verified 2026-08-23)
- Market data on LangGraph/agents naming frequency: `docs/research/market_research.md`
