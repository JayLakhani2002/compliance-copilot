# Handoff 01 — Phase 0 research (2026-08-23)

## Done
- PROMPT.md / CLAUDE.md / GLOSSARY.md written; folders: chathistory/, docs/{decisions,research/raw,handoffs}.
- 3 Sonnet researcher agents collected 50 unique open DE job ads (full text opened) → docs/research/raw/{indeed_stepstone,linkedin_xing_careers,aggregators_south_north}.md; per-agent handoffs in docs/handoffs/phase0-research/.
- Tally script (python, in this session) → docs/research/market_research.md with % table + evidence + top-3 options + recommendation (Option 1: Compliance Copilot — agentic RAG over EU AI Act/GDPR, LangGraph multi-agent, MCP, guardrails, eval-gated CI, Langfuse, FastAPI, pgvector, EU deploy).

## Decisions
- Defaults: 3–4 h/day, 6 weeks, EU deploy (Railway EU/Hetzner). Not yet confirmed by Jay.
- Firecrawl MCP free tier rate-limits fast → use WebSearch+WebFetch / Indeed MCP for future research.

## Open issues
- WAITING ON JAY: pick project option 1/2/3 (or 1 with different domain). Confirm defaults.

## Exact next step (Phase 1, planner)
1. Read docs/research/market_research.md.
2. Write docs/ARCHITECTURE.md (mermaid C4), ADRs in docs/decisions/ (framework, DB/vector, eval tool, guardrails, LLM tiering, API, auth, hosting, observability), docs/CURRICULUM.md (6 weeks day-by-day + LinkedIn posts).
3. git init; branches main/develop; .github/workflows/ci.yml; pyproject with uv; first commit on develop.

## Commands to resume
cat CLAUDE.md docs/PROGRESS.md chathistory/01_phase0_research.md docs/research/market_research.md

## Update 2026-08-23 (later)
- Jay chose Option 1 (Compliance Copilot).
- Jay wants Teacher in VS Code Claude panel, builder+agents in terminal → added TEACHER.md, docs/SETUP_SCREENS.md, docs/INBOX.md, docs/lessons/; CLAUDE.md roles updated. Next: Phase 1 in the builder (terminal) session.
