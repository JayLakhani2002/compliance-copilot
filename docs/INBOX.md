# INBOX — builder ⇄ teacher ⇄ Jay channel

Format, newest at bottom:
- `Q [open|answered] (from builder, date): question` then `A: answer`
- `DECISION (from Jay via teacher, date): ...`
- `TASK (from Jay via teacher, date): ...` → builder marks `[done]`

---
DECISION (Jay, 2026-08-23): Project = Option 1, Compliance Copilot (agentic RAG over EU AI Act + GDPR).
DECISION (Jay, 2026-08-23): Teacher runs in VS Code Claude panel; builder + agents run in terminal.
Q [answered] (builder, 2026-08-23): Python version — 3.12 pinned via .python-version (ML libs lag on 3.14). A: decided by planner.
Q [answered] (builder, 2026-08-23): GitHub repo public or private? A: Jay created it PRIVATE; flip later with `gh repo edit --visibility public --accept-visibility-change-consequences`. Planner recommends PUBLIC (portfolio). Jay: run `cd "/Users/jay/Documents/Projects/Langchain Project" && gh repo create compliance-copilot --public --source=. --remote=origin && git push -u origin main && git push -u origin develop`
TASK (builder → Jay, 2026-08-23): Install Docker to run Postgres/pgvector locally: `brew install --cask docker` (then open Docker.app once) or `brew install orbstack`. Until then integration tests skip locally and run in GitHub CI only.
TASK (builder → Jay, 2026-08-23): Day 4 needs an OpenAI API key for embeddings (ADR-0004) and Day 6+ an Anthropic key. Create `.env` from `.env.example` and fill OPENAI_API_KEY / ANTHROPIC_API_KEY. Never paste keys in chat.
Q [open] (builder → Jay, 2026-08-23): Pace check — builder finished Days 2–3 today. Do you want the builder to keep running ahead (code first, you learn with the teacher at your own pace), or lock-step (builder waits until you've done each day's lesson)? Default if no answer: run ahead, max 2 days ahead of your last teacher session.
