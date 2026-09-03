# Dockerfile — production image for the `api` container (docs/ARCHITECTURE.md
# §3, ADR-0032). Multi-stage: `builder` resolves/syncs dependencies with uv
# into a venv from the committed, frozen `uv.lock` (never re-locks in the
# image — ADR-0016/ADR-0028's "no surprise version drift" habit extended to
# the build itself); `runtime` copies ONLY that venv into a slim, non-root
# image with no `uv` binary and no source tree. The MCP server (ADR-0007)
# runs as a stdio subprocess spawned BY this same process — see
# `settings.mcp_use_uv_run` (settings.py) and `MCP_USE_UV_RUN=false` below
# for why that subprocess still works without `uv` in this image.
#
# uv-in-Docker pattern verified via Context7 (`/astral-sh/uv`,
# docs/guides/integration/docker.md, 2026-09-03): copying the `uv`/`uvx`
# binaries from the official distroless image, `--mount=type=cache` for the
# package cache, `--no-install-project` then a second sync once source is
# copied (so dependency layers cache independently of source changes), and
# `--no-editable` so the runtime venv doesn't need the source tree at all.

FROM python:3.12-slim AS builder

# Official uv binary, not `pip install uv` — avoids bootstrapping uv with
# pip inside a stage that's about to manage all other deps with uv anyway.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# UV_PYTHON_DOWNLOADS=0: use the base image's own Python, don't let uv fetch
# a second interpreter. UV_LINK_MODE=copy: the cache mount and /app live on
# different filesystems in BuildKit, so uv's default hardlink would warn/
# fail — copy is the documented fix. UV_NO_DEV: skip the `dev` dependency
# group (pytest/ruff/pre-commit/detect-secrets) — belt-and-suspenders with
# `--no-dev` below, both do the same job, neither hurts.
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy UV_NO_DEV=1

WORKDIR /app

# Dependencies first, source second: a source-only change (the common case)
# doesn't invalidate this layer's cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

# Only src/ — this project's own package (hatchling, pyproject.toml's
# `[tool.hatch.build.targets.wheel]`) — nothing else needed to build/install
# the wheel into the venv.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- runtime stage: no uv, no source tree, no dev tools --------------------
FROM python:3.12-slim

# Non-root user (Day-25 deferral, ADR-0032). Fixed uid/gid so it's
# reproducible across builds/hosts, not a value the base image happened to
# assign to some other package's user.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

# The synced venv IS the deployable artifact — `--no-editable` above means
# `compliance_copilot` is installed into it like any other package, so the
# runtime image needs no copy of src/ at all to run `python -m
# compliance_copilot.mcp_server` or `uvicorn compliance_copilot.api:app`.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # ADR-0032: this image has no `uv` binary — `_mcp_connection()`
    # (graph/build.py) spawns the MCP server subprocess with a bare
    # `python -m ...` instead of `uv run --frozen python -m ...` when this
    # is false. An image-level fact (this image never has uv), so it's set
    # here, not in docker-compose.prod.yml's per-deploy env_file.
    MCP_USE_UV_RUN=false

USER app

EXPOSE 8000

# No `curl` in `python:3.12-slim` — stdlib `urllib` instead. Exits non-zero
# (crashes the request, `urlopen` raises) on any non-2xx/connection failure,
# which is exactly what HEALTHCHECK needs. `/healthz` (api.py) does no DB/
# LLM call, matching "is the process alive," not "is it ready."
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

# --host 0.0.0.0: reachable from other containers on the compose network
# (Caddy, not the host directly — docker-compose.prod.yml gives `api` no
# host port). --no-server-header: same reasoning as Makefile's `api` target
# (ADR-0030) — don't advertise "uvicorn"+version, free recon for nothing.
CMD ["uvicorn", "compliance_copilot.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
