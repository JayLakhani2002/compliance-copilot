# Project rules (auto-loaded every session)

Portfolio project for Senior AI Engineer roles in Germany. Full brief: `PROMPT.md`.

## On session start — do this before anything else
1. Read `docs/PROGRESS.md` (if exists).
2. Read the newest file in `chathistory/` (if any).
3. Resume from its "Next step". Do not redo finished work.

## Roles — two sessions, one repo (see docs/SETUP_SCREENS.md)
- **Builder session** (terminal, Fable 5): orchestrates; spawns Sonnet 5 subagents (`model: "sonnet"`) for code + web research + review; runs git/tests. Does not write large code blocks itself.
- **Teacher session** (VS Code Claude panel, Fable 5, prompt in `TEACHER.md`): teaches, walks through code, relays Jay's decisions. Never edits code.
- Channel between them: `docs/INBOX.md` (Q/A, DECISION, TASK lines). Builder reads INBOX before starting each feature and writes any question for Jay there instead of blocking. Lessons persist in `docs/lessons/`.
- If you are the terminal session, you are the builder. If you were started from TEACHER.md, you are the teacher.

## Hard rules
- Verify every library API against Context7 MCP + official docs before writing it. Never guess.
- One feature = one `feature/*` branch = one PR into `develop`. Never commit to `main` directly. Conventional commits. Tag milestones `v0.x`.
- `pytest` green before the next feature starts.
- Every non-trivial choice gets `docs/decisions/ADR-NNNN-*.md` (context / options / decision / why not others / security+cost / how to reverse).
- Every code file: header comment (purpose + place in architecture) and "why" comments for a basic-Python reader.
- Teach before coding each feature: ≤1 page, what/why/why-not/senior view/security+cost/1 interview question.
- Security: validate at trust boundaries, prompt-injection defenses, rate limit, no PII in logs, secrets only in `.env`. EU hosting.
- Agents coordinate: each gets full context + prior agents' outputs from `docs/handoffs/<feature>/<role>.md`; loop researcher → coder → reviewer → coder fix → planner verify; reviewer bounces max twice, then escalate to planner. Shared naming in `docs/GLOSSARY.md`.
- Handoff: write `chathistory/NN_<feature>.md` at every merge and immediately on any context-limit warning. Include: done, decisions, open issues, exact next step, commands to resume. Then update `docs/PROGRESS.md`.

## Defaults (change if Jay says so)
3–4 h/day · 6 weeks · EU deploy (Railway EU / Hetzner) · Python 3.12 · uv + pyproject · pytest · ruff.
