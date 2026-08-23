# Two-screen workflow

```
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ VS Code — Claude Code panel  │   │ Terminal(s) — `claude` builder   │
│ = TEACHER (Fable 5)          │   │ = ORCHESTRATOR (Fable 5)         │
│ teaches, reviews, relays     │   │ spawns Sonnet coder/researcher/  │
│ Jay's decisions              │   │ reviewer agents, git, tests      │
└──────────────┬───────────────┘   └──────────────┬───────────────────┘
               │          shared repo files       │
               └────── docs/INBOX.md  docs/PROGRESS.md  chathistory/  docs/handoffs/ ──────┘
```

## Start each day
1. **Terminal**: `cd "/Users/jay/Documents/Projects/Langchain Project" && claude` → type `continue` (CLAUDE.md makes it read PROGRESS + newest chathistory and resume Phase/feature). Optional: a second terminal running `watch -n 30 git log --oneline -5` or `tail -f docs/INBOX.md` to keep an eye on agent activity.
2. **VS Code**: open the folder, open the Claude Code panel (sidebar icon or ⌘⇧P → "Claude Code: Open"), paste the contents of `TEACHER.md` once per new chat. It gives status, surfaces builder questions, teaches.
3. Builder questions land in `docs/INBOX.md`; you answer them in the VS Code chat; the teacher writes the answer back; the builder picks it up (it checks INBOX before each feature).

## Why files, not direct agent-to-agent chat
Both sessions are separate processes. The repo is the single source of truth, survives restarts, and is git-tracked, so every decision is reviewable. SendMessage between local sessions is a bonus when both are alive, not the backbone.
