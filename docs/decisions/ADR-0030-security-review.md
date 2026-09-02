# ADR-0030 — Security review day: dependency audit, static rules, HTTP hardening, secrets, dependabot

**Status:** accepted 2026-09-02

## Context

Each guardrail ADR (ADR-0006 through ADR-0022) reasoned carefully about its
own slice of security — the input guard, the output guard, PII redaction,
the red-team gate. Nothing had yet stepped back and walked the categories a
reviewer or an interviewer would actually ask about, in order: is a secret
ever at risk of reaching a commit, does a dependency carry a known
vulnerability, does the HTTP surface still match what ADR-0016 decided now
that more routes exist, do the logs still hold to "no PII, no secrets"
now that more code paths log things. `docs/THREAT_MODEL.md` already models
the right habit for the LLM-specific slice; this ADR generalizes that same
walk to the rest of the system — dependencies, static analysis, the HTTP
surface, secrets, and CI enforcement — as a single dated pass (Day 25).

## Options considered

1. **Dependency audit tool**: `uv audit` (native to this project's package
   manager) vs. `pip-audit` (a separate tool/dependency) vs. no automated
   audit at all.
2. **Static security lint**: ruff's bundled `S` (flake8-bandit) rule set vs.
   a separate `bandit` invocation vs. nothing.
3. **Secret scanning**: `detect-secrets` (pure Python, pre-commit hook) vs.
   `gitleaks` (a separate Go binary) vs. relying solely on GitHub's
   repo-level push protection.
4. **HTTP hardening**: which headers/middleware to add now vs. defer to a
   future reverse proxy (Caddy, ADR-0010) that doesn't exist in this repo
   yet.
5. **CORS**: add a permissive policy for future browser clients vs. keep it
   off by default.
6. **Dependency hygiene going forward**: Dependabot vs. a manual recurring
   reminder.

## Decision

1. **`uv audit --locked`, blocking in CI.** Verified present in installed
   `uv==0.12.5` (`uv audit --help`) — no need for `pip-audit` as a second
   tool when the package manager already ships one. Ran against this
   project's `uv.lock`: **6 findings, all `cryptography` 48.0.1** (3 unique
   CVEs, each reported under both a GHSA and a PYSEC id —
   `GHSA-g6cj-pr64-35w5`/`PYSEC-2026-3552`, `GHSA-jwv3-5hgf-82ww`/
   `PYSEC-2026-3553`, `GHSA-m2h6-j472-rp4c`/`PYSEC-2026-3554`). Investigated
   with `uv tree --invert --package cryptography`: it's a transitive
   dependency of `presidio-anonymizer==2.2.364` (ADR-0020) and
   `mcp[crypto]`'s `pyjwt[crypto]` extra (ADR-0007) — neither is a direct
   dependency this project pins itself. Tried `uv lock --upgrade-package
   cryptography`: stayed at 48.0.1. Root cause, confirmed against PyPI's
   own metadata: `presidio-anonymizer==2.2.364` (the latest release)
   requires `cryptography<49.0.0,>=48.0.1` — an upper bound this project
   cannot lift without presidio-anonymizer itself relaxing it. All three
   CVEs are in `cryptography`'s X.509 certificate-chain-building
   (`GHSA-jwv3-5hgf-82ww`, `GHSA-m2h6-j472-rp4c`) and PKCS#7
   `EnvelopedData`-decryption (`GHSA-g6cj-pr64-35w5`) code paths — this app
   never verifies an X.509 chain or decrypts PKCS#7 data with
   `cryptography` itself (it only reaches this library via
   presidio-anonymizer's AES anonymizer operator and pyjwt's signing
   helpers, neither of which touches those paths), so none of the three are
   reachable through this app's own code. Ignored with three dated,
   commented `--ignore` flags in CI (`.github/workflows/ci.yml`) rather than
   left unaddressed or blocking every future PR on something this project
   cannot fix — re-check `uv audit --locked` (drop an `--ignore` line) the
   day `presidio-anonymizer` relaxes its pin.
2. **Ruff's `S` (flake8-bandit) rule set, not a separate `bandit`
   invocation.** Verified present in installed ruff 0.16.4 (`ruff rule
   S101`/`S104` both resolve) — one linter, one config file, instead of a
   second tool with its own config format. `select = [..., "S"]`
   (`pyproject.toml`) plus `[tool.ruff.lint.per-file-ignores]` `"tests/*" =
   ["S101"]` (assert-in-tests is the whole pytest idiom, not a security
   smell). Every OTHER `S` finding — three, all in `tests/`, zero in `src/`
   or `evals/` — got a per-line `# noqa: S### — reason` instead of a
   blanket suppression: `random.Random`/`random.random()` seeded from a
   hash (test-fixture embedding vectors, S311, not cryptographic use) and
   one `subprocess.run` call with a hardcoded literal script + `sys.
   executable` (S603, not untrusted input).
3. **`detect-secrets`, not `gitleaks`.** Both catch the same class of
   mistake; `detect-secrets` (PyPI 1.5.0, pure Python) installs the same
   way every other dev tool here does (`pyproject.toml`'s `dev` group,
   `uv sync`) with zero new install mechanism, where `gitleaks`'s own
   pre-commit hook needs a separate Go-binary download step — more moving
   parts for a repo this size, no material detection advantage. Baseline
   (`.secrets.baseline`, committed) generated with `detect-secrets scan
   --exclude-files 'evals/embeddings_cache/.*'` (cached embedding vectors —
   raw floats that happen to look base64/hex-shaped — produced ~1200 pure-
   noise "high entropy string" hits, excluded from the scan entirely rather
   than individually audited). The remaining 9 findings were hand-audited
   and are all `"is_secret": false`: the CI job's deliberate dummy
   `OPENAI_API_KEY: ci-dummy-key-never-used`, `.env.example`'s/
   `docker-compose.yml`'s/`settings.py`'s placeholder
   "user:password"-shaped local dev DSNs, the test suite's own fixture
   constant (`API_KEY = "test-secret-key-not-a-real-secret"`), and
   base64-encoded prompt-injection attack text used as guard/red-team
   fixtures. `pre-commit run detect-secrets --all-files` passes clean.
   GitHub's own push protection (a repo-settings toggle, not a code change)
   is the second, provider-side layer — noted here for Jay to confirm on,
   not something this ADR can verify from inside the repo.
4. **HTTP hardening, four real additions, one explicit non-addition:**
   - `SecurityHeadersMiddleware` (`api.py`, same `BaseHTTPMiddleware` shape
     as the existing `BodySizeLimitMiddleware`): `X-Content-Type-Options:
     nosniff` and `Referrer-Policy: no-referrer` on every response; on
     `/ask`/`/resume` specifically, `Cache-Control: no-store` — stronger
     than, and overriding, the SSE stream's own `Cache-Control: no-cache`
     (set by the route handlers for anti-buffering, ADR-0016) — `no-cache`
     still permits a cache to store the response as long as it revalidates
     first, the wrong default for a body that can carry a
     redacted-but-still-sensitive answer.
   - `RequestIdMiddleware` + `RequestIdFilter`: mints one `uuid.uuid4()` per
     request, echoes it as an `X-Request-ID` response header, and (via a
     `contextvars.ContextVar` + a `logging.Filter` that mutates
     `record.msg` directly, the same trick `logging_filter.py`'s
     `PiiScrubFilter` already uses) stamps it onto every log line that
     request's own graph execution emits — so a support engineer with one
     failed request's id can grep every line it touched, without threading
     a `request_id` parameter through every function in the graph.
   - `TrustedHostMiddleware` (Starlette's own, no new dependency),
     `allowed_hosts` driven by `settings.allowed_hosts` — defaults to
     `["localhost", "127.0.0.1", "testserver"]` (the last one is the fixed
     `Host` header FastAPI's/httpx's `TestClient` sends, verified against
     this project's own test suite — every existing test uses a bare
     `TestClient(app)`, which would otherwise start getting 400s from this
     middleware). A real deploy MUST override `ALLOWED_HOSTS` to its actual
     public hostname(s) — this default has no wildcard.
   - `uvicorn --no-server-header` (Makefile's `api` target): don't
     advertise "uvicorn" + version in every response — free reconnaissance
     for an attacker fingerprinting a known CVE, zero functional benefit.
     Verified flag name against installed `uvicorn --help`
     (`--server-header / --no-server-header`).
   - **No `Strict-Transport-Security`.** TLS terminates at Caddy
     (ADR-0010) — this app is upstream of that hop and has no verified
     `X-Forwarded-Proto` trust boundary today, so an app-level HSTS claim
     would be unverifiable from where this code runs. This stays a named,
     open gap (`docs/SECURITY.md`), not a silent one — the day Caddy's
     config lands in this repo with a confirmed `X-Forwarded-Proto`
     contract, HSTS becomes a proxy-level header, not an app one.
5. **No CORS, still.** Confirmed by `grep -rn CORSMiddleware src/` — nothing
   configures it, so FastAPI's own "no CORS unless asked" default already
   holds. This is now a *documented* deliberate default (`api.py`'s own
   comment, this ADR), not an unexamined absence: this API is
   key-authenticated, server-to-server (`X-API-Key`, ADR-0016) — no browser
   is ever expected to call it cross-origin, so there's no legitimate
   origin to allow, and adding a permissive policy would only ever widen
   the attack surface for zero benefit.
6. **Dependabot**, `.github/dependabot.yml`: `package-ecosystem: "uv"`
   (verified against GitHub's own docs, Context7 `/github/docs` — `uv` is a
   directly-supported ecosystem value, `v0`, not a `pip` workaround) for
   this project's own dependencies, plus `package-ecosystem:
   "github-actions"` for the pinned Action versions in `ci.yml` (an
   outdated Action is as much a supply-chain risk as an outdated library).
   Weekly, not daily — a portfolio project with no on-call, so a bump-PR
   queue that outruns review capacity is worse than a day's staleness.

## Log audit (no code change — a clean result)

Grepped every `logger.*` call in `src/compliance_copilot/` (~50 call
sites, `router.py`/`critic.py`/`checkpointer.py`/`costing.py`/
`graph/nodes.py`/`guards/*.py`/`tracing.py`/`mcp_server.py`/`cli.py`/
`embeddings.py`/`db.py`/`cached_embeddings.py`) against the "no question
text, no answer text, no API keys, no PII" rule. Every call already
follows it: `guard_in`'s flagged/classifier/PII-redaction logs pass
category names, scores, and entity TYPE names only (never the matched
text or the question itself — `guards/injection.py`'s `GuardResult.
reasons` and `guards/pii.py`'s redaction entities are built that way
already); MCP tool-call logs pass tool name, latency, and exception CLASS
name only (never `args`, which carry the already-redacted question);
outage/critic-unavailable logs pass exception class names or a fixed
`critic_error:<ClassName>` string, never free model text on the error
path; `guard_out`'s block log passes a fixed reason code
(`guards/output.py`'s `OutputVerdict.reason`), never the answer.
`logging_filter.py`'s `PiiScrubFilter` regex backstop (ADR-0020) was
re-read, not re-derived — still a defence-in-depth layer behind the
primary "never log the raw question" rule, unchanged. SSE error events
(`api.py`'s `except` clauses in `_run_graph_and_stream`) were checked one
by one: every one sends a fixed `{"type": ...}` payload built from
exception CLASS names or citation/anchor data, never a request body or a
stack trace. **No leak found — no fix needed.**

## Why not the others

- **`pip-audit`**: redundant once `uv audit` is confirmed native and
  working — a second tool doing the same job for no added coverage.
- **A separate `bandit` invocation**: ruff's bundled `S` rules already run
  in the same lint pass this project already has wired into CI and
  pre-commit — a second tool/config file for rules ruff already ships.
- **`gitleaks`**: real, well-regarded tool — rejected here purely on
  install-mechanism grounds (a Go binary vs. a `uv sync`-installed Python
  package) for a repo this size, not a detection-quality judgment.
- **Deferring all HTTP hardening to a future Caddy config**: rejected for
  the same reason ADR-0016 already rejected it for the body-size cap — no
  Caddy config exists in this repo yet, so "the proxy will handle it"
  currently means "nothing handles it." The app-level middleware is the
  floor; a proxy can add a second layer later without removing this one.
- **Permissive CORS "for future browser clients"**: rejected as
  speculative — this project has no browser client today, and adding
  cross-origin permission ahead of an actual need only expands the attack
  surface for a feature that doesn't exist yet (project rule: no
  speculative hardening, and symmetrically, no speculative *permissions*
  either).
- **`Strict-Transport-Security` at the app level**: rejected as an
  unverifiable claim from a process that doesn't independently know
  whether the original client hop was TLS (see Decision §4 above) — a
  proxy-level decision once the trust boundary is real, not an app one to
  guess at today.

## Security & cost implications

- **Security:** every new control here is either free (a lint rule, a
  header) or a one-time dev-dependency install (`detect-secrets`) — no new
  runtime dependency, no new network call per request, no latency added to
  `/ask`/`/resume` beyond a few header writes and a `uuid4()` call.
  Blocking CI on `uv audit` and `detect-secrets` means a real, fixable
  vulnerability or an accidentally committed secret fails the build the
  same way a lint error already does — visible, not silently permissive.
- **Cost:** zero — no paid service added (GitHub's push protection, OSV
  lookups for `uv audit`, and Dependabot are all free for a public/portfolio
  repo). The `--ignore`-listed `cryptography` CVEs cost nothing to carry
  forward since they're unreachable through this app's own code paths;
  revisit only when `presidio-anonymizer` ships a compatible bump.

## How to reverse

Each control is independently removable with no cascading change: drop the
`--ignore` lines (or the whole `uv audit` step) from `ci.yml` to change the
dependency-audit policy; drop `"S"` from `pyproject.toml`'s `select` list
to disable the bandit rules; delete `.pre-commit-config.yaml`'s
`detect-secrets` hook block (and `.secrets.baseline`) to remove secret
scanning; remove any of `SecurityHeadersMiddleware`/`RequestIdMiddleware`/
`TrustedHostMiddleware`'s `app.add_middleware(...)` call in `api.py` to
drop that one header/behaviour without touching the others; delete
`.github/dependabot.yml` to stop automated bump PRs. None of these touch
the graph, the guardrails, or any other feature's own ADR.

## References

- `uv audit --help`, installed `uv==0.12.5` — flags `--locked`, `--ignore`,
  `--output-format json` confirmed live.
- `uv tree --invert --package cryptography` (installed `uv.lock`) —
  confirms `cryptography` reaches this project only via
  `presidio-anonymizer` and `mcp[crypto]`'s `pyjwt[crypto]` extra.
- PyPI JSON API, `presidio-anonymizer` 2.2.364's own `requires_dist`:
  `cryptography<49.0.0,>=48.0.1` — fetched 2026-09-02.
- `ruff rule S101`/`S311`/`S603`, installed ruff 0.16.4.
- `detect-secrets scan`/`detect-secrets-hook --help`, installed
  `detect-secrets==1.5.0`; upstream repo `github.com/Yelp/detect-secrets`
  (confirmed via installed package metadata's `Home-page`).
- `uvicorn --help`, installed `uvicorn==0.52.4` —
  `--server-header`/`--no-server-header`, `--date-header`/`--no-date-header`
  flags confirmed live (the `--date-header` flag exists too but isn't used
  here — no decision needed on it, `Date` is not a sensitive header).
- Starlette `BaseHTTPMiddleware.__call__`/`TrustedHostMiddleware`,
  installed `starlette` (`.venv/lib/python3.12/site-packages/starlette/
  middleware/{base,trustedhost}.py`) — confirms the `call_next()`-spawned
  child task inherits a `contextvars.ContextVar` set before it's spawned,
  which is what makes `RequestIdFilter` see the right id inside the
  actual (concurrently running) request-handling task.
- GitHub Dependabot options reference, Context7 `/github/docs` — `uv`
  package-ecosystem support (`v0`, no beta flag), example
  `dependabot.yml` shape.
