# ADR-0033 — Deploy runbook + Terraform stub (no server yet)

**Status:** accepted 2026-09-03

## Context

ADR-0010 chose the hosting model (Hetzner VPS, Docker Compose, Caddy) and
ADR-0032 built the deployable artifact (`Dockerfile`,
`docker-compose.prod.yml`, `Caddyfile`) — but no document walked an operator
through actually standing that up on a real box, and no automation existed
for the repetitive first-login hardening steps. No Hetzner VPS exists yet
(Jay's provisioning decision is pending) and the OpenAI key in `.env` is
currently unfunded, so today's constraint is: produce a runbook and script
that are *correct* against what already exists in this repo (compose file,
`settings.py`, `.env.example`) and against Docker's own current docs, without
being able to run either against a real server or make a real LLM call.

## Options considered

1. **Manual runbook only** — write `docs/DEPLOY.md`, no automation. Honest
   and simple, but repeats error-prone manual steps (ufw rules, the Docker
   apt-repo dance) every time a VPS is rebuilt.
2. **Runbook + `deploy/deploy.sh`, plus a Terraform stub for AWS
   `eu-central-1`** (this ADR) — automate the mechanical hardening/install
   steps; add a plan-only IaC stub as a literacy artifact, per the
   curriculum's Day-28 stretch goal and ADR-0010's original mention of it.
3. **Full IaC on Hetzner via the `hcloud` Terraform provider** — would
   actually provision the chosen hosting target, not just demonstrate a
   parallel one. Rejected for *today*: no VPS/Hetzner API token exists yet
   to develop or validate against, and ADR-0010 already treated Hetzner
   itself as a manually-created resource for a solo, 3–4h/day project.

## Decision

**Runbook + script now; AWS stub for IaC literacy; `hcloud` named as the
natural upgrade path, not built today.**

- `docs/DEPLOY.md`: a top-to-bottom runbook — provisioning, hardening, DNS,
  `.env`, first run (init-db + ingest, with its embedding cost called out),
  smoke tests, backup/restore drill, update/rollback, an EU-residency
  checklist that mirrors `docs/ARCHITECTURE.md` §8 honestly (storage/
  observability EU by construction; inference/embeddings EU only by
  provider/region choice, not yet exercised), and teardown.
- `deploy/deploy.sh`: idempotent, parameterized (`HOSTNAME`/`REPO_URL`/
  `APP_USER`), automates hardening (sudo user, ufw, unattended-upgrades) +
  Docker install (official apt repo, commands verified against
  <https://docs.docker.com/engine/install/ubuntu/>, fetched 2026-09-03) +
  clone + `compose up`. Stops itself before `compose up` if `.env` isn't
  present yet rather than running with placeholder secrets. No secrets
  inside the script itself.
- `infra/terraform/`: minimal AWS `eu-central-1` stub — one EC2 `t3.small`
  (Ubuntu 24.04 AMI via data source), a security group scoped to 22/80/443,
  `user_data` that fetches and runs the *same* `deploy/deploy.sh` (not a
  duplicated hardening script), state kept local (never applied, nothing to
  protect with a remote backend). Its own `README.md` states plainly that
  this is validated-but-never-applied literacy, not a competing hosting
  decision.

Everything above is honest about being **dry-validated only**:
`bash -n deploy/deploy.sh` (syntax) + manual review; the Terraform stub via
`terraform fmt -check` / `init -backend=false` / `validate` (all credential-
free) using the official `hashicorp/terraform` Docker image, since the
`terraform` binary itself isn't installed locally.

## Why not the others

- **Manual-only runbook**: rejected — the hardening/install sequence is
  identical every time a box is rebuilt (or a second box is ever needed);
  scripting it once is a smaller total cost than re-typing it correctly
  under time pressure the day a real VPS exists.
- **Full `hcloud` IaC now**: rejected for today specifically — there is no
  Hetzner account/API token to develop or validate against yet, and
  building untested IaC against a provider nobody has touched risks
  shipping something that's wrong in ways `fmt`/`validate` can't catch
  (only `plan`/`apply` against the real API would). The AWS stub, by
  contrast, validates fully offline (no credentials needed for
  `validate`), which is what today's environment can actually support.
  `hcloud` is the named upgrade path once a Hetzner account exists.

## Security & cost implications

- **Security:** `deploy.sh` enforces the exact same perimeter ADR-0010
  named (22/80/443 only via ufw), disables nothing that isn't already
  disabled (no password SSH is a provisioning-time choice, step 1 of
  `docs/DEPLOY.md`, not something the script can retroactively fix), and
  never creates, prints, or embeds a secret — `.env` is entirely the
  operator's responsibility, consistent with ADR-0032's "secrets via
  `env_file` only" stance. The Terraform stub's security group mirrors the
  same three ports; its `user_data` runs the identical hardening code path
  as Hetzner, not a second, divergently-maintained version.
- **Cost:** zero new recurring cost — no server was provisioned, no cloud
  API call was made (Terraform `validate` needs no credentials), and the
  only spend this feature could ever trigger (`cli ingest`'s embedding
  calls) is explicitly flagged in `docs/DEPLOY.md` step 5 as needing a
  funded key first.

## How to reverse

Delete `docs/DEPLOY.md`, `deploy/deploy.sh`, and `infra/terraform/` —
nothing else in the repo depends on their existence (ADR-0032's compose
file, Dockerfile, and Caddyfile are untouched by this feature). The
`deploy-validate` Makefile target and the README pointer to `DEPLOY.md` are
the only other footprint; both are one-line removals.

## References

- Docker's official Ubuntu apt-repo install steps: fetched from
  <https://docs.docker.com/engine/install/ubuntu/>, 2026-09-03 — mirrored
  verbatim in both `docs/DEPLOY.md` and `deploy/deploy.sh`.
- `docker run --rm -v "$PWD":/w -w /w hashicorp/terraform:latest fmt -check`
  / `init -backend=false` / `validate` against `infra/terraform/`: all
  pass (2026-09-03) — provider `hashicorp/aws ~> 5.0` resolves, `terraform
  validate` reports "Success! The configuration is valid."
- `bash -n deploy/deploy.sh`: clean.
- ADR-0010 (hosting decision this implements), ADR-0032 (the artifact this
  deploys), `docs/ARCHITECTURE.md` §8 (EU-residency claims this runbook's
  checklist mirrors).
