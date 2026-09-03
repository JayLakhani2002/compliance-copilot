# Deploy runbook — Hetzner VPS

A top-to-bottom runbook for the day a real Hetzner VPS exists (ADR-0010,
ADR-0032, ADR-0033). Nothing in this document has been run against a real
server yet — no VPS exists and the OpenAI key currently in `.env` is
unfunded — every step below is written from the compose file, `settings.py`,
and Docker's official docs, and marked honestly where that matters. Treat
each command as reviewed, not executed, until someone actually runs it here.

## 1. Provision the VPS

- **Image:** Ubuntu 24.04 LTS.
- **Region:** Falkenstein or Nuremberg (Hetzner's German locations) — keeps
  the VPS itself EU-resident, matching `docs/ARCHITECTURE.md` §8's "storage
  is EU by construction" claim.
- **Size:** a 4 GB-class box (e.g. CX22/CPX21-tier) — ADR-0010's planner
  amendment: Langfuse runs in Langfuse Cloud EU, so this VPS only hosts
  `api`, `postgres`, `caddy`, `backup` (docker-compose.prod.yml), not the
  five-plus-service self-host stack the original ADR-0010 sizing assumed.
- **Auth:** upload your SSH public key at creation time and disable password
  login — Hetzner's console offers this at provisioning; don't set a root
  password at all if the console allows skipping it.

## 2. First-login hardening

SSH in as `root` (or Hetzner's default user) once, then run
`deploy/deploy.sh` to automate the rest of this section — or do it by hand
following the same steps the script takes:

1. Create a non-root sudo user, copy `authorized_keys` to it.
2. `ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable`
   — nothing else is reachable from the internet.
3. `apt-get install unattended-upgrades` + enable it — security patches land
   without a manual `apt upgrade` cadence.
4. **fail2ban (optional):** not installed by `deploy.sh`. A single shared
   `X-API-Key` with SSH password auth already disabled has a much smaller
   brute-force surface than a typical VPS; add fail2ban later only if real
   SSH scan traffic in the logs justifies the extra moving part.
5. Docker + the compose plugin, via Docker's official apt repository —
   commands verified against <https://docs.docker.com/engine/install/ubuntu/>
   (fetched 2026-09-03; re-verify on the day if this has since changed):

   ```bash
   sudo apt update && sudo apt install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
   Types: deb
   URIs: https://download.docker.com/linux/ubuntu
   Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
   Components: stable
   Architectures: $(dpkg --print-architecture)
   Signed-By: /etc/apt/keyrings/docker.asc
   EOF
   sudo apt update
   sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

Run the whole thing with the script instead:

```bash
HOSTNAME=compliance-copilot.example.com \
REPO_URL=https://github.com/<you>/compliance-copilot.git \
APP_USER=deploy \
sudo -E bash deploy/deploy.sh
```

It's idempotent — re-run after fixing anything that failed partway; already-
done steps are detected and skipped.

## 3. DNS

Create an A record for `DEPLOY_HOSTNAME` (e.g.
`compliance-copilot.example.com`) pointing at the VPS's public IPv4. Caddy
(step 5) won't be able to issue a Let's Encrypt certificate until this
resolves.

## 4. `.env`

`deploy.sh` clones the repo but deliberately stops before `compose up` if
`.env` doesn't exist yet — it never creates or fetches one. On the VPS:

```bash
cd ~/compliance-copilot
cp .env.example .env
```

Fill in, per `.env.example`'s comments (cross-referenced against
`src/compliance_copilot/settings.py`):

| Variable | Note |
|---|---|
| `LLM_PROVIDER` | `openai` (interim default, ADR-0002 amendment) or `anthropic` |
| `OPENAI_API_KEY` | **a NEW key on a funded org** — the key used during development is exhausted; do not reuse it |
| `ANTHROPIC_API_KEY` | only if `LLM_PROVIDER=anthropic` |
| `DATABASE_URL` | must match `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` below exactly |
| `API_KEY` | generate with `python -c "import secrets;print(secrets.token_urlsafe(32))"` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | optional — Langfuse Cloud EU; leave unset for zero tracing |
| `DEPLOY_HOSTNAME` | the DNS name from step 3, e.g. `compliance-copilot.example.com` |
| `ALLOWED_HOSTS` | **JSON array syntax**, e.g. `["compliance-copilot.example.com"]` — `TrustedHostMiddleware` (ADR-0030) 400s every other Host header, and a comma-separated string does not parse (pydantic-settings) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | a real generated password, not the dev throwaway `user`/`password` |
| `CLASSIFIER_ENABLED` / `PII_REDACTION_ENABLED` / `ROUTER_ENABLED` / `CRITIC_ENABLED` | leave `true` (defaults) unless debugging a specific guard/feature outage |

## 5. Bring the stack up

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

First run only — create the schema, then ingest both regulations (embeds
~590 chunks; calls the OpenAI embeddings API, costs a few cents — needs the
funded key from step 4):

```bash
docker compose -f docker-compose.prod.yml exec api python -m compliance_copilot.cli init-db
docker compose -f docker-compose.prod.yml exec api python -m compliance_copilot.cli ingest --regulation all
```

Smoke tests — from the VPS itself (bypasses Caddy/DNS/TLS, checks the app
directly):

```bash
docker compose -f docker-compose.prod.yml exec api python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/healthz').status)"
```

From outside, once DNS has propagated and Caddy has issued its certificate:

```bash
curl -s https://compliance-copilot.example.com/healthz
curl -s https://compliance-copilot.example.com/readyz
curl -sN -X POST https://compliance-copilot.example.com/ask \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"question":"When is an AI system high-risk?"}'
```

`/healthz` and `/readyz` should both 200; `/ask` should stream `node` events
and end with a `final` event carrying a cited answer.

## 6. Backup verification + restore drill

The `backup` sidecar dumps `postgres` daily (kept 7 days, ADR-0032). Verify
a dump exists and restore drill works end-to-end on a throwaway copy:

```bash
make backup-now   # or wait for the daily dump
docker compose -f docker-compose.prod.yml exec backup ls -la /backups

# Restore drill (into the SAME db — --clean drops existing objects first;
# only run this against data you're prepared to overwrite, e.g. right after
# ingest, before real traffic exists):
docker compose -f docker-compose.prod.yml cp backup:/backups/<file>.dump ./restore.dump
docker compose -f docker-compose.prod.yml cp ./restore.dump postgres:/tmp/restore.dump
docker compose -f docker-compose.prod.yml exec postgres \
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean /tmp/restore.dump
```

Confirm the app still answers correctly after a restore before considering
the drill passed.

## 7. Update / rollback

```bash
cd ~/compliance-copilot
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

Roll back to a previous commit the same way (`git checkout <sha>` first).
Full stop:

```bash
docker compose -f docker-compose.prod.yml down
docker image prune -f   # reclaim space from superseded image layers
```

`down` alone keeps volumes (`pgvector_data`, `backups`, `caddy_data`,
`caddy_config`) intact — data survives a stop/start cycle.

## 8. EU-residency checklist

Mirrors `docs/ARCHITECTURE.md` §8 — stated honestly, not oversold:

| Component | EU-resident? | How |
|---|---|---|
| VPS (api/postgres/caddy/backup) | Yes, by construction | Hetzner Falkenstein/Nuremberg (step 1) |
| Postgres storage | Yes, by construction | Runs on the above VPS |
| Langfuse tracing (if enabled) | Yes, by configuration | Langfuse Cloud **EU** region (`LANGFUSE_BASE_URL=https://cloud.langfuse.com`, AWS `eu-west-1`) |
| LLM inference | **No, by default** | `LLM_PROVIDER=openai`/`anthropic` direct APIs are not EU-region-pinned; the documented production path is AWS Bedrock `eu-central-1`, not shipped yet (ADR-0002) |
| Embeddings | **No, by default** | OpenAI `text-embedding-3-small` (US); Bedrock/Cohere `eu-central-1` is the documented alternative (ADR-0004), not shipped yet |
| EUR-Lex source corpus | N/A | Public EU legislation, no residency concern |

**Caveat to state plainly in any write-up:** this deploy's storage and
observability are EU-resident from day one; inference and embeddings are
not, until the Bedrock `eu-central-1` path (referenced by `infra/terraform/`
as the AWS-side story) replaces the interim OpenAI/Anthropic direct-API
default.

## 9. Teardown

```bash
docker compose -f docker-compose.prod.yml down -v   # -v also deletes volumes — data is gone
```

Then delete the VPS from the Hetzner console and remove the DNS A record.
