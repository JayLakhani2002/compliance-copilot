# ADR-0032 — Production Docker Compose: Dockerfile, Caddy, backups

**Status:** accepted 2026-09-03

## Context

ADR-0010 decided the hosting model (Docker Compose on a Hetzner VPS, Caddy
for TLS) and ADR-0009's planner amendment moved Langfuse to Langfuse Cloud
EU, dropping the self-host ClickHouse/Redis/MinIO stack from the deploy
target — but no `Dockerfile` or `docker-compose.prod.yml` existed in the
repo yet; `docker-compose.yml` (root) is dev-only (one `postgres` service,
its own header comment says so). This feature builds the actual deployable
artifact: an image for `api`, and the compose file that runs it, Caddy, and
Postgres together on the VPS, plus a backup mechanism (named in ADR-0010's
security note as a gap: "docker-compose on a VPS" needs someone to actually
answer "what happens when the disk dies").

Day-25's security review (ADR-0030) explicitly deferred non-root container
images to "today" (Day 26) — this ADR closes that deferral too.

## Options considered

1. **Compose-on-VPS** (this ADR) vs. **Kubernetes** vs. a **managed PaaS**
   (Railway/Render) for actually running the built image.
2. **Backup approach**: a dedicated backup tool (pgBackRest/WAL-G, with
   offsite S3-compatible storage) vs. a **simple `pg_dump` sidecar loop**
   vs. no automated backup (manual `pg_dump` only, run by Jay).
3. **MCP server subprocess spawn in the runtime image**: keep `uv run
   --frozen python -m ...` (ADR-0007's existing dev/CI spawn command,
   requiring `uv` in the image) vs. **a bare `python -m ...` spawn**
   (requires the image's own venv to already have the package installed,
   which the multi-stage build already produces) vs. installing `uv` into
   the runtime image just to keep the dev spawn command unchanged.
4. **Non-root image user**: a fixed numeric uid/gid created in the
   Dockerfile vs. relying on a base-image default non-root user (`python:
   3.12-slim` has none) vs. staying root (status quo, the Day-25 deferral).

## Decision

1. **Compose-on-VPS**, per ADR-0010 — this ADR doesn't reopen that choice,
   it implements it. No k8s manifests, no Terraform beyond ADR-0010's
   existing stretch stub — out of scope for today (Ponytail: no
   speculative infra for a single-VPS, single-tenant deploy).
2. **A `pg_dump` sidecar loop** (`docker-compose.prod.yml`'s `backup`
   service): a `while true; do pg_dump ...; sleep 86400; done` shell loop
   in the same `pgvector/pgvector:pg16` image Postgres already uses (no
   second image to pull), writing timestamped custom-format dumps to a
   named `backups` volume and deleting anything older than 7 days. `make
   backup-now` runs the identical `pg_dump` on demand via `docker compose
   exec`. This is the smallest honest answer for a single-VPS portfolio
   deploy with no on-call — a dedicated backup tool's incremental/WAL-based
   restore points and offsite replication are real capabilities this setup
   doesn't have, and that gap is named here, not hidden.
3. **Bare `python -m compliance_copilot.mcp_server` in the runtime image**,
   gated by a new `settings.mcp_use_uv_run` flag (`True` everywhere except
   the Docker image, which sets `MCP_USE_UV_RUN=false` in its `ENV`).
   `graph/build.py`'s `_mcp_connection()` now branches on this flag to
   build its `command`/`args`. The runtime image's `.venv` already has
   `compliance-copilot` installed non-editable (uv's own multi-stage
   pattern, verified via Context7 `/astral-sh/uv`'s
   `docs/guides/integration/docker.md`) — invoking `python -m
   compliance_copilot.mcp_server` from that venv's `bin/python` (on `PATH`)
   runs the exact same module `uv run` would have resolved to, with no
   re-lock/re-resolve step to skip in the first place since there's no `uv`
   binary to invoke it with. Installing `uv` into the runtime image instead
   was rejected as the wrong direction: `uv run` in dev exists specifically
   to avoid needing a pre-activated venv on a developer's machine; a
   container already **is** a pre-built, immutable environment, so `uv`
   there would be dead weight solving a problem the image never has.
4. **A fixed non-root user, uid/gid 1000** (`groupadd --gid 1000 app &&
   useradd --uid 1000 --gid app`), `COPY --chown=app:app` for the venv,
   `USER app` before `CMD`. Fixed ids (not "whatever the base image
   assigns") make the image's identity reproducible across rebuilds/hosts,
   which matters if a bind-mounted volume's ownership is ever compared
   against it later.

## Why not the others

- **Kubernetes / managed PaaS**: already rejected in ADR-0010 on solo-dev
  ops-capacity and EU-residency-narrative grounds respectively — nothing
  in this feature changes that reasoning.
- **A dedicated backup tool (pgBackRest/WAL-G)**: rejected as more
  operational surface (a second tool's own config, its own failure modes)
  than a single-VPS portfolio deploy's actual risk profile justifies today.
  The upgrade path is named, not foreclosed: swap the `backup` service's
  entrypoint for the real tool without touching anything else in the
  compose file.
- **No automated backup at all**: rejected — "the VPS disk fails" is not a
  hypothetical worth leaving unhandled when a working sidecar is a dozen
  lines of shell.
- **Keeping `uv run --frozen` in the runtime image (installing `uv`
  there)**: rejected per Decision §3 above — solves a dev-workflow problem
  a container doesn't have, at the cost of a binary (and its own update
  surface) the image would otherwise never need.
- **Staying root** (Day-25 deferral, now closed): rejected — the app
  process has no legitimate reason to run as root inside its own
  container; a container escape or a dependency RCE gets meaningfully less
  useful against a uid-1000 process with no write access outside its own
  (read-only-in-practice) venv.

## Security & cost implications

- **Security:** only `caddy` publishes host ports (80/443) —
  `api`/`postgres`/`backup` have no `ports:` entry in
  `docker-compose.prod.yml` at all, reachable only over the compose-internal
  network (`compliance_copilot_prod`), consistent with ADR-0010's "only
  Caddy, and through it the API, should be publicly reachable." The `api`
  image runs as uid 1000, non-root, with no `uv`/source tree/dev tools in
  the runtime stage — a smaller attack surface than a single-stage build
  copying the whole repo in. Caddy adds `Strict-Transport-Security` only
  (not the app's other security headers, which `SecurityHeadersMiddleware`
  — ADR-0030 — already sets), consistent with ADR-0030's reasoning that
  HSTS belongs at the actual TLS-termination hop. `ALLOWED_HOSTS` and
  `DEPLOY_HOSTNAME` are both documented in `.env.example`; a deploy that
  forgets to set `ALLOWED_HOSTS` to its real hostname gets 400s from
  `TrustedHostMiddleware` (ADR-0030), not silent exposure. Secrets
  (`OPENAI_API_KEY`/`API_KEY`/`POSTGRES_PASSWORD`/etc.) reach every service
  via `env_file: .env` (gitignored) only — never baked into the image or
  written into `docker-compose.prod.yml` itself.
- **Cost:** unchanged fixed-cost bracket from `docs/ARCHITECTURE.md` §9 —
  the VPS itself is now sized for **three** stateful-ish services
  (`postgres`, `caddy`, `backup`'s volume) instead of the five-plus a
  self-hosted Langfuse stack would have needed (ADR-0010's amendment
  already banked this saving: a 4 GB-class Hetzner box, not 8 GB). No new
  paid service: Langfuse Cloud's free Hobby tier (ADR-0009's amendment),
  Caddy's automatic HTTPS (Let's Encrypt, free), and the `pg_dump` sidecar
  (no third-party backup service) all cost nothing beyond the VPS itself.

## How to reverse

Delete `Dockerfile`/`docker-compose.prod.yml`/`Caddyfile`/`.dockerignore` to
fully undo this feature — nothing else in the repo depends on their
existence (dev's `docker-compose.yml` is untouched). `settings.
mcp_use_uv_run` defaults to `True`, so removing the Dockerfile's
`MCP_USE_UV_RUN=false` override alone reverts `_mcp_connection()`'s spawn
command to the pre-existing `uv run --frozen` behaviour everywhere. Swapping
the `backup` service's `pg_dump` loop for a different tool touches only that
one service block.

## References

- uv-in-Docker multi-stage pattern (`--no-install-project`/`--no-editable`/
  cache-mount/`UV_LINK_MODE=copy`): Context7 `/astral-sh/uv`,
  `docs/guides/integration/docker.md`, verified 2026-09-03. `uv sync
  --help` (installed `uv==0.12.5`) confirms `--no-dev`/`--frozen`/
  `--no-install-project`/`--no-editable` are all real flags on this
  version.
- Caddy `reverse_proxy` streaming behaviour: Context7 `/caddyserver/website`,
  `caddyfile/directives/reverse_proxy.md`, verified 2026-09-03 —
  `flush_interval` is documented as **ignored** whenever the upstream
  response's `Content-Type` is `text/event-stream`, which is exactly what
  `/ask`/`/resume` (api.py) already send; no override needed for SSE to
  stream unbuffered through Caddy.
- pydantic-settings `list[str]` env parsing: verified live against the
  installed version — a JSON-array-shaped env var (`ALLOWED_HOSTS=
  ["host"]`) parses; a comma-separated string raises a parsing error. This
  is why `.env.example`'s new `ALLOWED_HOSTS` line uses JSON syntax.
- `caddy validate` (official `caddy:2` image) against this repo's
  `Caddyfile` with a dummy `DEPLOY_HOSTNAME`: `Valid configuration`.
- `docker build .` (this repo's `Dockerfile`): succeeds, image runs,
  `/healthz`/`/readyz` both return 200 from inside the compose network
  against a disposable Postgres (`docker compose -f docker-compose.prod.yml
  up -d postgres api` with dummy secrets + `EMBEDDINGS_PROVIDER=cached`,
  then `docker compose exec api python -c "urllib request to
  localhost:8000/readyz"` → 200; Docker's own HEALTHCHECK reached
  `healthy`; stack torn down cleanly afterwards).
