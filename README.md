# Compliance Copilot

Agentic RAG over the EU AI Act and GDPR: an LLM agent that answers compliance
questions grounded in the actual regulatory text, with retrieval/answer
quality measured by evals and guardrails enforced against prompt injection
and hallucination — all gated by CI so a broken change can't merge.

**Status:** Phase 1 scaffolding — repo structure, tooling, and CI are in
place; no application code yet. See `docs/ARCHITECTURE.md` for the system
design and `docs/research/market_research.md` for why this project was chosen.

## How to run

```bash
make setup   # uv sync + install the pre-commit git hook (one-time)
cp .env.example .env   # fill in real secrets locally; DATABASE_URL default matches docker-compose.yml
make db-up   # start Postgres+pgvector via Docker Compose (needs Docker installed)
make db-init # create the vector extension, tables, and HNSW index
make test    # uv run pytest -m "not integration"
```

DB integration tests (`tests/test_db_integration.py`) need a real Postgres —
they're skipped automatically unless `DATABASE_URL` is set, and run in
GitHub CI's `integration` job against a `pgvector/pgvector:pg16` service
container regardless of what's set up locally.

## Run the API

```bash
# set API_KEY in .env first (see .env.example)
make api  # uvicorn on http://127.0.0.1:8000, auto-reload
curl -sN -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" -d '{"question":"When is an AI system high-risk?"}'
```

Set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` to enable tracing (Langfuse Cloud, EU) — unset by default, so the app runs with zero tracing until you do.

### Health and readiness (ADR-0028)

```bash
curl localhost:8000/healthz  # liveness: process alive, no DB/LLM work — "should this container restart?"
curl localhost:8000/readyz   # readiness: SELECT 1 against Postgres — "should traffic route here?" (200 or 503)
```

Both are unauthenticated and unrate-limited (orchestrator probes, not
product traffic). `/ask` itself degrades gracefully on an answer-model
outage (a `degraded: true` final event with the retrieved articles listed,
zero citations) rather than a bare error, and ends the stream with a typed
`{"type": "timeout"}` event if the whole request runs past
`REQUEST_TIMEOUT_S` (default 60s) or `{"type": "dependency_unavailable"}` if
Postgres is unreachable mid-request.

### Conversations (`thread_id`, ADR-0024)

Omit `thread_id` on the first call — the response's very first SSE event is
`event: thread`, `data: {"thread_id": "<uuid>"}`, the id the server just
minted. Send that same id back on the next call to continue the
conversation (up to the last 3 turns are replayed into the prompt):

```bash
curl -sN -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question":"And what about GDPR?","thread_id":"<uuid from the first response>"}'
```

A client-supplied `thread_id` must be a syntactically valid UUID4 (422
otherwise) — but note this doesn't grant per-caller privacy: this API has
one shared `X-API-Key`, so any key holder can supply any validly-shaped
`thread_id` and resume that conversation (ADR-0024's security note, an
open gap ADR-0016 already named). Erase a conversation's checkpointed state
with `python -m compliance_copilot.cli delete-thread <uuid>`.

### Human review on low confidence (`interrupt`/`/resume`, ADR-0025)

When the critic scores its confidence in a drafted answer below
`CRITIC_CONFIDENCE_MIN` (default 0.6) — and the critic itself didn't error;
a critic-tier outage fails OPEN (no pause), see ADR-0025's round 2 note —
the run pauses instead of streaming a `final` event. `/ask`'s stream ends
with:

```
event: interrupt
data: {"thread_id": "<uuid>", "interrupt_id": "...", "status": "under_review"}
```

That's deliberately all the END USER sees — no draft, confidence, or
reasoning on this channel (ADR-0025 round 2). An **operator** reviews the
full payload via the CLI, which reads it straight off the checkpointed
state and prints it before applying anything:

```bash
python -m compliance_copilot.cli resume <thread-id> --decision approve|edit|reject [--answer TEXT]
# prints: interrupt_id / question / draft answer / critic confidence / critic reasoning
# THEN applies the decision — the operator sees what they're approving.
```

Or resolve it over the API with `POST /resume` — note `interrupt_id` (from
the `interrupt` event above) is required:

```bash
curl -sN -X POST localhost:8000/resume -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"thread_id":"<uuid>","interrupt_id":"<uuid>","decision":"approve"}'
# or: {"thread_id":"<uuid>","interrupt_id":"<uuid>","decision":"edit","edited_answer":"..."}
# or: {"thread_id":"<uuid>","interrupt_id":"<uuid>","decision":"reject"}
```

`/resume` streams the same `node`/`final` events `/ask` does once resumed.
404 if `thread_id` is unknown; 409 if it isn't currently paused, or if
`interrupt_id` doesn't match the pending review (a stale reference —
someone already resumed it, or a later `/ask` re-paused the same thread on
a different question). `/ask` itself now also 409s if called again with a
`thread_id` that's currently paused — it never silently starts a new run
over a pending review. An `edit`'s replacement text still has to pass
every `guard_out` check a model's own draft does — never trusted just
because a human wrote it. Same shared-`X-API-Key` caveat as `thread_id`
above: any key holder who knows/guesses a paused `thread_id`+`interrupt_id`
pair can resume it (ADR-0016, not solved by this feature). `ask` prints
`under review (thread_id ...)` instead of an answer when it pauses, and
409s (exit 9) if you `--thread-id` back into a thread that's already
paused — run `resume` instead.

## Run in production

See `docs/DEPLOY.md` (ADR-0033) for the full Hetzner runbook — provisioning,
hardening, DNS, `.env`, first run, backup/restore drill, update/rollback,
and an EU-residency checklist — plus `deploy/deploy.sh` to automate the
hardening/install steps, and `infra/terraform/` for a validated-but-never-
applied AWS `eu-central-1` stub.

`docker-compose.prod.yml` (ADR-0032) is the deployed shape: Caddy (TLS +
reverse proxy, the only exposed container), `api` (this repo's `Dockerfile`
— the MCP server runs inside it as a stdio subprocess, ADR-0007, not a
separate container), `postgres`, and a `backup` sidecar. Langfuse tracing
(if configured) goes to Langfuse Cloud EU, not a self-hosted service — see
ADR-0009's amendment.

```bash
cp .env.example .env   # fill in real secrets + DEPLOY_HOSTNAME/ALLOWED_HOSTS/POSTGRES_* (see comments)
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml exec api python -m compliance_copilot.cli init-db
# Ingest both regulations (one-off; embeds ~590 chunks — this calls the
# OpenAI embeddings API and costs a few cents, so it needs a funded
# OPENAI_API_KEY in .env):
docker compose -f docker-compose.prod.yml exec api python -m compliance_copilot.cli ingest --regulation all
```

`ALLOWED_HOSTS` must be set to the real public hostname(s) — the compiled-in
default only accepts `localhost`/`127.0.0.1`/`testserver` and rejects
everything else (ADR-0030). Backups: the `backup` service dumps `postgres`
daily to a named volume (`backups`), keeping the last 7 days; `make
backup-now` runs one on demand. Restore (dump lives in `backup`'s
container, not `postgres`'s — copy it across first):

```bash
docker compose -f docker-compose.prod.yml cp backup:/backups/<file>.dump ./restore.dump
docker compose -f docker-compose.prod.yml cp ./restore.dump postgres:/tmp/restore.dump
docker compose -f docker-compose.prod.yml exec postgres \
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean /tmp/restore.dump
```

## MCP server

`make mcp` starts a standalone MCP server (`search_regulation`, `get_article`,
`cite`) over stdio — the same tools an MCP client can point at, e.g. Claude
Desktop's config:

```json
{
  "mcpServers": {
    "compliance-copilot": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/repo", "python", "-m", "compliance_copilot.mcp_server"]
    }
  }
}
```

Set `MCP_TRANSPORT=streamable-http` (internal network only — no auth today, see ADR-0007) to switch transports.

## Quality gates

Two CI gates: **`quality-gate`** (retrieval hit@5/MRR) runs on every PR against
committed, real, cached embeddings — no API key, no network call. **`answer-quality`**
(LLM-as-judge faithfulness) needs a real key, so it only runs nightly, on
manual dispatch, or on a PR labelled `quality-gate`. Refresh the committed
embedding cache after changing the golden sets, the corpus, or the embedding
model with `make eval-cache` (needs `OPENAI_API_KEY`); run the gates locally
with `make quality-gate` and `uv run python -m evals.run_answer_eval`.

A third gate, the **red-team ASR/FPR check** (ADR-0022), runs 40 original attacks
against the full guard stack: a no-key heuristics subset (`make redteam-fast`) is a
plain pytest test that runs on every PR for free, and the full-pipeline run
(`make redteam`, needs `OPENAI_API_KEY`) folds into the same `answer-quality` CI job.
Gate: ASR ≤ 5%, FPR ≤ 10%.
