# ADR-0011: Tooling — Python 3.12, uv, ruff, pytest, pre-commit, GitHub Actions, conventional commits, branch model

## Status
Accepted. 2026-08-23.

## Context
Day-to-day developer experience tooling: dependency management, linting/formatting, test running, pre-commit checks, CI, commit conventions, and branch strategy. These choices don't show up in the market research as differentiators (no ad names "uv" or "ruff" specifically), but they're the substrate every other ADR's code gets built on, and the project's hard rules already commit to several of these choices (`pytest` green before the next feature, `feature/*` branches into `develop`, conventional commits) — this ADR is where they're formally recorded with the reasoning, per the same hard rule that requires an ADR for every non-trivial choice.

## Options considered
For the package manager specifically (the one place a real alternative exists): **uv** (a Rust-based Python package/project manager from Astral, positioned as a fast replacement for pip/pip-tools/pipx/poetry/venv) vs. **Poetry** (the more established Python-native dependency manager) vs. **plain pip + venv**. For linting/formatting: **ruff** (a Rust-based linter+formatter, replacing the combination of flake8+isort+black+more) vs. running flake8/isort/black separately.

## Decision
**Python 3.12**, **uv** for dependency/project management (`uv add`, `uv run`, `uv.lock` for reproducible installs), **ruff** for linting and formatting (one tool instead of several), **pytest** for testing (already the harness ADR-0005's eval suite is built on), **pre-commit** hooks (running ruff + basic checks before a commit lands), **GitHub Actions** for CI (running pytest, ruff, and the eval-gate suite from ADR-0005 on PRs), **conventional commits** (`feat:`, `fix:`, `docs:`, etc. — a commit-message convention that makes history/changelog generation mechanical), and a **`main`/`develop`/`feature/*`** branch model (one feature = one `feature/*` branch = one PR into `develop`, per the project's hard rule; `main` reserved for tagged milestones).

**Verified detail:** `uv`'s PyPI release is 0.12.5 (verified 2026-08-23) — like FastAPI, `uv` has not reached 1.0 while in wide production use; this is consistent with its stated release cadence (frequent, incremental releases) rather than a stability concern. `uv`'s own Context7-indexed docs include a dedicated GitHub Actions setup guide (`astral-sh/setup-uv`), confirming first-class CI support, which matters directly for this ADR's CI decision.

## Why not the others
- **Poetry**: not wrong, but `uv` is materially faster (a Rust implementation vs. Poetry's Python implementation) for the install/lock cycle a solo developer runs repeatedly during a 3–4h/day build — a real, if unglamorous, time saving over 6 weeks of daily `uv run`/`uv add` cycles.
- **Plain pip + venv**: rejected for lacking a lockfile by default (reproducibility across the two development sessions this project explicitly runs — builder and teacher — depends on a locked, reproducible dependency set) and for the extra manual steps (`requirements.txt` + `requirements-dev.txt` maintenance) `uv`/Poetry both remove.
- **Separate flake8/isort/black**: rejected in favor of `ruff` purely for tool-count reduction — three separate tools with three separate configs vs. one tool covering the same ground, with no functional gap significant enough to justify the extra moving parts for a solo build.

## Security & cost implications
- **Security:** `uv.lock` (or Poetry's `poetry.lock`, had that been chosen) pins exact dependency versions, which is a supply-chain-security-relevant property (no silent minor/patch bump introducing a vulnerability or behavior change between two `uv run` invocations on different machines/days) — worth naming explicitly since this project already has a "secrets only in `.env`" security rule that a lockfile complements (reproducible installs mean fewer surprises in what code actually runs in CI vs. locally).
- **Cost:** all of this tooling is free/open source and runs locally or in GitHub Actions' free tier for a project at this scale — no direct spend; the only cost-adjacent consideration is GitHub Actions minutes if CI runs grow heavy (already flagged in ADR-0005 re: scoping the eval suite to PRs rather than every commit).

## How to reverse
Each of these is a locally-scoped developer-tooling choice with low switching cost — `uv` → Poetry/pip is a dependency-file format change; `ruff` → flake8/black is a config-file change; neither touches application code. The branch model and commit convention are process, not code, and can change by team agreement without any migration work at all.

## References
- `uv`, PyPI: 0.12.5 — https://pypi.org/project/uv/ (verified 2026-08-23); docs: https://docs.astral.sh/uv/; GitHub Actions integration confirmed via Context7 `/astral-sh/setup-uv`
- Project hard rules this ADR formalizes (branch model, `pytest` green, conventional commits): project brief (README)
