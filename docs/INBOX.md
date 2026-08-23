# INBOX — builder ⇄ teacher ⇄ Jay channel

Format, newest at bottom:
- `Q [open|answered] (from builder, date): question` then `A: answer`
- `DECISION (from Jay via teacher, date): ...`
- `TASK (from Jay via teacher, date): ...` → builder marks `[done]`

---
DECISION (Jay, 2026-08-23): Project = Option 1, Compliance Copilot (agentic RAG over EU AI Act + GDPR).
DECISION (Jay, 2026-08-23): Teacher runs in VS Code Claude panel; builder + agents run in terminal.
Q [answered] (builder, 2026-08-23): Python version — 3.12 pinned via .python-version (ML libs lag on 3.14). A: decided by planner.
Q [answered] (builder, 2026-08-23): GitHub repo public or private? A: PUBLIC — Jay decided via teacher 2026-08-23; teacher ran `gh repo edit --visibility public` — repo is now live at github.com/JayLakhani2002/compliance-copilot. DONE.
TASK (builder → Jay, 2026-08-23): Install Docker to run Postgres/pgvector locally: `brew install --cask docker` (then open Docker.app once) or `brew install orbstack`. Until then integration tests skip locally and run in GitHub CI only.
TASK (builder → Jay, 2026-08-23): Day 4 needs an OpenAI API key for embeddings (ADR-0004) and Day 6+ an Anthropic key. Create `.env` from `.env.example` and fill OPENAI_API_KEY / ANTHROPIC_API_KEY. Never paste keys in chat.
Q [answered] (builder → Jay, 2026-08-23): Pace check — builder finished Days 2–3 today. Do you want the builder to keep running ahead (code first, you learn with the teacher at your own pace), or lock-step (builder waits until you've done each day's lesson)? Default if no answer: run ahead, max 2 days ahead of your last teacher session.
A: DECISION (Jay via teacher, 2026-08-23): **Run ahead, NO cap** — builder ships as fast as quality allows; Jay catches up via teacher lessons on his own pace. Teacher tracks lesson debt in docs/lessons/00_student_profile.md.
DECISION (Jay via teacher, 2026-08-23): Docker runtime = OrbStack (teacher installing via brew). Docker TASK → in progress.
DECISION (Jay via teacher, 2026-08-23): Public repo = recruiter-facing ONLY. Personal/teaching/workflow files must not be in the public repo or its git history. Repo flipped back to PRIVATE (teacher) until cleaned; teacher re-flips public after builder confirms.
TASK (Jay via teacher, 2026-08-23): Purge from git tracking AND full history (git-filter-repo or equivalent), then force-push main+develop; files stay ON DISK (still the live workflow channel): TEACHER.md, PROMPT.md, CLAUDE.md, docs/CURRICULUM.md, docs/lessons/, docs/INBOX.md, docs/SETUP_SCREENS.md, chathistory/, docs/handoffs/. Add all to .gitignore. KEEP public: src/, tests/, docs/ARCHITECTURE.md, docs/decisions/, docs/GLOSSARY.md, docs/research/, README, pyproject, CI, Makefile, .env.example. Update CLAUDE.md's own "chathistory stays tracked" rule text to match (tracked→local-only). Rebase any open feature branches onto the rewritten history before merging.
