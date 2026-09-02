# Security

A short, honest account of what's been checked, what the current posture
is, and how to report a problem — last reviewed Day 25 (`docs/decisions/
ADR-0030-security-review.md` has the full findings/reasoning; this page is
the summary someone should be able to read in two minutes).

## What was audited (Day 25)

- **Dependencies** — `uv audit --locked`, wired into CI as a blocking step.
  6 findings at review time, all `cryptography` 48.0.1 (3 unique CVEs),
  transitively pinned by `presidio-anonymizer`'s own upper bound — none
  reachable through this app's actual code paths (it never verifies an
  X.509 chain or decrypts PKCS#7 data). Ignored with three dated,
  commented `--ignore` flags in `.github/workflows/ci.yml`, not left
  unaddressed. Full reasoning: ADR-0030.
- **Static rules** — ruff's `S` (flake8-bandit) rule set, added to
  `pyproject.toml`'s lint config. Zero findings in `src/` or `evals/`;
  three test-only findings each got a per-line `# noqa: S### — reason`
  rather than a blanket suppression (a hashed-seed test PRNG, a
  hardcoded-literal `subprocess.run` call).
- **HTTP surface** — security-headers middleware (`X-Content-Type-Options:
  nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on
  `/ask`/`/resume`), a per-request `X-Request-ID` correlation id,
  `TrustedHostMiddleware`, and `uvicorn --no-server-header`. See "Auth
  model and known gaps" below for what's deliberately still open.
- **Secrets** — `detect-secrets` as a blocking pre-commit hook, committed
  `.secrets.baseline` audited by hand (every finding confirmed a false
  positive — a deliberate CI dummy key, placeholder local-dev DSNs, test
  fixtures). GitHub push protection is the second, provider-side layer
  (a repo setting, not something this repo's code can verify).
- **Logs** — every `logger.*` call site in `src/compliance_copilot/`
  re-read against the "no question text, no answer text, no API keys, no
  PII" rule. Clean: guard/redaction logs pass category names and entity
  types only, never matched text; tool-call logs pass names and exception
  classes only, never arguments; SSE error events send fixed type codes,
  never a request body or stack trace. `logging_filter.py`'s regex scrub
  (ADR-0020) remains the defence-in-depth backstop behind that rule, not
  the primary control.

## Current posture

**Auth model.** A single shared `X-API-Key` header, checked with
`secrets.compare_digest` (constant-time, no timing side-channel), fails
closed (503) if unconfigured (ADR-0016). This is the right amount of
mechanism for a single-tenant portfolio API with no user identity to
federate — not OAuth2/JWT, which would be more machinery than a shared
secret needs. Rate limiting (`slowapi`, pre-auth via real ASGI middleware)
and a request body-size cap both apply before any LLM call is possible.

**Known, accepted gaps** (named on purpose, not discovered by an
attacker):

- **Any key holder can resume or read any thread.** Because there's only
  one shared key, there's no binding between "the key that started a run"
  and "the key allowed to `/resume` its pause" or read its `thread_id`'s
  history. A valid-but-guessed `thread_id`/`interrupt_id` pair is
  resumable by anyone holding the one shared key (ADR-0016, ADR-0024,
  ADR-0025). Accepted for a single-tenant deployment; would need per-caller
  API keys (not just one shared secret) before this stops being true.
- **A paused human-review run never expires.** `hitl_node`'s `interrupt()`
  has no TTL — a paused thread nobody ever resumes stays paused (and its
  checkpoint row persists) indefinitely (ADR-0025). No cleanup job exists
  yet; add one the day real traffic makes stale pauses a real accumulation,
  not before.
- **The MCP stdio subprocess can wedge under sustained load.** Observed
  during Day 24's cost measurement: after roughly 15–20 tool calls, a
  session spawn/handshake can hang in a way `asyncio.wait_for`'s
  cooperative cancellation doesn't preempt (ADR-0029's "Open risk"
  section). The request-wide timeout (`settings.request_timeout_s`,
  ADR-0028) bounds how long any one request waits, but doesn't fix the
  underlying transport issue — backlogged as a focused follow-up (a
  persistent shared MCP session, or a hard PID-level kill on timeout)
  before this path carries real production traffic.
- **No `Strict-Transport-Security` at the app level.** TLS terminates at
  Caddy (ADR-0010); this app has no verified `X-Forwarded-Proto` trust
  boundary today, so an app-level HSTS claim would be unverifiable from
  where this code runs. Becomes a proxy-level header once that config
  exists in this repo with a confirmed contract.
- **In-memory, single-process rate limiting.** `slowapi`'s default
  storage resets on restart and doesn't share state across replicas
  (ADR-0016's own `ponytail:` note) — correct for today's single-process
  deployment, not for a horizontally-scaled one.
- **No non-root container user / Docker healthchecks wired to `/healthz`
  and `/readyz` yet.** Scheduled infra work, not done today — named here as
  ahead, not pretended finished.

**EU residency**, stated as plainly as `docs/ARCHITECTURE.md` §8 already
does: storage (Postgres) is EU by construction; inference/embeddings are
EU only by choice of provider/region, and that choice is not yet locked in
on the cheaper default (OpenAI) path.

## Reporting a problem

This is a portfolio project, not a production service handling real user
data — if you find a security issue, open a GitHub issue on this
repository describing it (no dedicated security contact/bug bounty exists
yet). Please don't include real credentials, real personal data, or a
working exploit payload in a public issue; a description of the mechanism
is enough.
