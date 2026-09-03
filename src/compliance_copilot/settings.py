# src/compliance_copilot/settings.py — centralised app configuration.
# Sits under the "api"/"mcp-server" containers in docs/ARCHITECTURE.md: every
# module that needs a config value (DB URL today; LLM/embedding/Langfuse
# keys as later features land) imports `settings` from here instead of
# calling `os.getenv(...)` itself. Why centralise: one place to see every
# config key the app reads, one place pydantic validates types/required-ness
# at startup (fail fast on a missing DATABASE_URL, not on the first query),
# and tests can override values without touching real env vars (see
# tests/test_db_integration.py, which reads DATABASE_URL directly to decide
# whether to skip — that's an intentional exception, not a settings bypass).
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file=".env": loads local secrets from the gitignored .env (see
    # .env.example for the template). extra="ignore" so keys this feature
    # doesn't use yet (ANTHROPIC_API_KEY, LANGFUSE_*) don't fail validation —
    # they're read by their own future settings fields, not by this one.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # postgresql+psycopg:// — the "+psycopg" part tells SQLAlchemy to use the
    # psycopg 3 driver specifically (there's also a legacy psycopg2 driver
    # SQLAlchemy still supports under a different URL scheme).
    database_url: str = "postgresql+psycopg://user:password@localhost:5432/compliance_copilot"

    # ADR-0004: default embedding model + its output dimension. The two must
    # stay in sync (embeddings.py asserts this) — pgvector's Vector column
    # (db.py) is a *fixed* dimension, so a mismatched model fails loudly at
    # insert time instead of silently corrupting the nearest-neighbour index.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # OPENAI_API_KEY isn't read here directly — langchain-openai's
    # OpenAIEmbeddings reads it from the environment itself (see
    # embeddings.py). Declaring it as a field would just duplicate that and
    # risk it ending up in a log/repr of `settings`.

    # ADR-0002 amendment (2026-08-24): only an OPENAI_API_KEY exists in .env
    # today (Anthropic key not yet purchased), so "openai" is the interim
    # default — ADR-0002's actual *target* is "anthropic" with
    # answer_model="claude-sonnet-5"; make_llm() (graph/nodes.py) is the one
    # place this flag is read, so flipping it back is a one-line change.
    llm_provider: str = "openai"

    # gpt-4.1-mini: cheapest current OpenAI model that (a) supports
    # `with_structured_output(method="json_schema")` and (b) isn't a
    # reasoning-only model — gpt-5.x-class models silently drop
    # `temperature=0` (langchain_openai's own `validate_temperature`),
    # which would make answers non-deterministic. $0.40/$1.60 per MTok
    # in/out (verified platform.openai.com/docs/pricing, 2026-08-24).
    # None = "use the provider's default" (make_llm() maps openai ->
    # gpt-4.1-mini, anthropic -> claude-sonnet-5, ADR-0002), so flipping
    # LLM_PROVIDER alone never sends one vendor's model id to the other's
    # client. Set ANSWER_MODEL only to override a provider's default.
    answer_model: str | None = None

    # ANTHROPIC_API_KEY isn't read here directly — same reasoning as
    # OPENAI_API_KEY above: langchain-anthropic's ChatAnthropic reads it
    # from the environment itself (see graph/nodes.py's make_llm()).

    # ADR-0016: the API's shared secret (`X-API-Key` header, api.py). `None`
    # by default — a missing key means the API refuses EVERY request with
    # 503 "API_KEY not configured", never silently "auth disabled" (fail
    # closed, not open). Generate one with
    # `python -c "import secrets;print(secrets.token_urlsafe(32))"`.
    api_key: str | None = None

    # slowapi rate-limit string (api.py's `Limiter(default_limits=...)`,
    # applied via `SlowAPIMiddleware` — ADR-0016), e.g. "20/minute" —
    # verified against the installed `limits` package's rate-limit string
    # grammar (`<amount>/<multiplier><unit>`). Per-key (falls back to
    # per-IP pre-auth), not global: every request triggers a real LLM call
    # (ADR-0002's cost model), so this caps one caller's worst-case spend.
    rate_limit: str = "20/minute"

    # Trust-boundary cap on `AskRequest.question` (api.py) — bounds cost and
    # rejects absurd input before the graph (and any LLM call) ever runs.
    max_question_chars: int = 2000

    # ADR-0016: request body size cap (api.py's `BodySizeLimitMiddleware`),
    # checked via `Content-Length` before the body is read. 16 KiB is
    # generous headroom over `max_question_chars`'s ~2000-byte question
    # plus JSON overhead — a real request is a few hundred bytes.
    max_body_bytes: int = 16_384

    # ADR-0009 amendment: tags every Langfuse trace (tracing.py's
    # `run_config()`) so "did a deploy cause this quality drop" is a filter,
    # not a guess. No LANGFUSE_* fields here — tracing.py reads those two
    # keys straight from the environment (same reasoning as
    # OPENAI_API_KEY/ANTHROPIC_API_KEY above: a secret has no business in a
    # `Settings()` repr/log).
    env: str = "dev"

    # ADR-0018: `guard_in_node`'s flag/allow cutoff — a question flags when
    # `detect()`'s summed per-category score (guards/injection.py) is >=
    # this. 1.0 matches every category's own ceiling weight, so any ONE
    # matched category is enough to refuse; raising it would require two
    # independent categories to agree before refusing (fewer false
    # positives, more heuristics slip through) — a one-line env tweak, no
    # code change, if the false-positive/false-negative balance needs
    # retuning after real traffic.
    guard_threshold: float = 1.0

    # ADR-0019: layer 2 of `guard_in` — a cheap-LLM classifier that catches
    # paraphrased/multilingual attacks the heuristic layer's regexes can't.
    # `False` skips constructing/calling it entirely (`get_classifier_
    # dependency()` in api.py returns `None`, cli.py passes `None`) — the
    # one-line reversal a classifier-outage incident would reach for first.
    classifier_enabled: bool = True
    # `None` = "use make_classifier_llm()'s per-provider default" (openai ->
    # gpt-4.1-nano, anthropic -> claude-haiku-4-5), same "flip the provider,
    # get the right model for free" reasoning as `answer_model` above. Set
    # CLASSIFIER_MODEL only to override.
    classifier_model: str | None = None
    # Bounds one slow classifier call so it can't stall every request — a
    # timeout here is one more input to `classify()`'s fail-open path
    # (guards/classifier.py), not a hard failure.
    classifier_timeout_s: float = 3.0
    # `guard_in_node`'s cutoff for trusting a classifier "block" verdict
    # (guards/classifier.py's `Verdict.confidence`, 0-1). Below this, the
    # verdict is treated the same as "allow" — a low-confidence block isn't
    # worth refusing a real user over. One-line tuning knob if real traffic
    # shows the balance needs adjusting.
    classifier_block_confidence: float = 0.6

    # ADR-0020: `guard_in`'s layer-3 PII redaction (Presidio) — swaps
    # names/emails/phones/IBANs/credit cards/IPs in the question for a
    # `<TYPE>` placeholder before retrieval/LLM/tracing ever see it (GDPR
    # Art. 5(1)(c)/Art. 25). `False` is the one-line "how to reverse" this
    # feature's ADR names for a redaction-related incident — `guard_in_node`
    # skips straight past `redact()` entirely, same "disabled means skip
    # it" contract `classifier_enabled` above already gives layer 2.
    pii_redaction_enabled: bool = True

    # ADR-0007: transport for mcp_server.py's `FastMCP` instance. "stdio"
    # (default) is what CI/dev/`MultiServerMCPClient` all spawn — no
    # listening port, no auth surface. "streamable-http" is for the
    # Compose-internal `mcp-server` container only (docs/ARCHITECTURE.md
    # §3) — never exposed past Caddy, since there's no auth on it yet.
    mcp_transport: Literal["stdio", "streamable-http"] = "stdio"
    # Only read when mcp_transport="streamable-http". 127.0.0.1 (not
    # 0.0.0.0): even inside the Compose network this stays loopback-only
    # unless an operator deliberately widens it — narrowest default that
    # still works for the same-container/localhost case.
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001

    # ADR-0023: the router — a cheap-LLM call, before `retrieve`, that labels
    # the question ai_act/gdpr/both/out_of_scope so `retrieve_node` can
    # narrow `search_regulation`'s filter. `False` is the one-line "how to
    # reverse" (mirrors `classifier_enabled`) — `router_node` (graph/nodes.py)
    # then no-ops (`GraphContext.router=None`), `route_after_router` treats
    # the absent `state["router"]` key the same as "both" (today's
    # pre-router behaviour), never `out_of_scope`.
    router_enabled: bool = True
    # `None` = "use make_router_llm()'s per-provider default" (openai ->
    # gpt-4.1-nano, anthropic -> claude-haiku-4-5), same reasoning as
    # `classifier_model`. Set ROUTER_MODEL only to override.
    router_model: str | None = None
    # Bounds one slow router call so it can't stall every request — a
    # timeout here is one more input to `route()`'s fail-open path
    # (router.py), not a hard failure.
    router_timeout_s: float = 3.0

    # ADR-0023: the critic — a cheap-LLM call, after `answer`, that checks
    # whether the drafted answer's claims are actually supported by its own
    # cited excerpts (semantic support, not just verbatim-quote presence).
    # `False` disables it entirely (mirrors `classifier_enabled`) —
    # `critic_node` (graph/nodes.py) no-ops when `GraphContext.critic=None`.
    critic_enabled: bool = True
    # `None` = "use make_critic_llm()'s per-provider default", same
    # reasoning as `classifier_model`/`router_model`.
    critic_model: str | None = None
    # Bounds one slow critic call — same reasoning as `router_timeout_s`.
    critic_timeout_s: float = 3.0
    # ADR-0025: the confidence cutoff `hitl_node` (graph/nodes.py) reads to
    # decide whether a low-confidence critic verdict pauses an answer for
    # human review (`interrupt()`) — a verdict scoring AT or ABOVE this
    # passes straight through. Starts conservative (biased toward pausing
    # more, not less); tune from real critic-score traffic, not intuition.
    critic_confidence_min: float = 0.6

    # ADR-0007 Day-17 amendment: `retrieve_node` (graph/nodes.py) is now the
    # MCP *client* — `False` is the one-line "how to reverse" an MCP-outage
    # incident needs: `build.make_mcp_tools()` returns `None` instead of
    # spawning the server subprocess, so `GraphContext.tools` stays empty.
    # This is NOT a silent fallback to direct retrieval (the lesson's
    # fail-loud rule) — a real question still reaches `retrieve_node`, which
    # still raises `ToolCallError` the moment it finds no tools, just
    # without ever spawning a doomed subprocess first. Defaults `True` since
    # the MCP server is the only retrieval path now.
    mcp_enabled: bool = True
    # Per-tool-call ceiling (`asyncio.wait_for` in `retrieve_node`) — bounds
    # one stuck `search_regulation`/`get_article` call so it can't stall a
    # request indefinitely (lesson 17, "Check yourself" #1). 30s is
    # generous headroom over a local stdio round-trip's normal
    # millisecond-scale latency.
    mcp_tool_timeout_s: float = 30.0

    # ADR-0028: `make_llm()`'s (graph/nodes.py) answer-model client — the ONE
    # call every request actually waits on end-to-end, so it gets the most
    # generous per-call budget of any LLM client in this app (contrast the
    # guard-tier clients below, which fail open/pessimistic on their own
    # outage and so only need a tight timeout, not a retry budget). 30s is
    # headroom over a normal structured-output completion; `max_retries=1`
    # makes the SDK's own (previously implicit, `openai`'s
    # `DEFAULT_MAX_RETRIES=2`) retry an explicit, deliberate choice — one
    # bounded retry for a transient blip, not stacked with a second retry
    # layer anywhere else (lesson 23's "don't stack retries" rule).
    answer_timeout_s: float = 30.0
    answer_max_retries: int = 1

    # ADR-0028: `get_embeddings()` (embeddings.py) — a query-time embedding
    # call blocks `retrieve`'s search the same way a hung answer call blocks
    # `answer`. 10s matches the guard-tier clients' budget (a single short
    # text, not a multi-thousand-token completion, so it needs far less
    # headroom than `answer_timeout_s`); `max_retries=2` (not 1) because an
    # embedding call has no side effect worth worrying about re-running
    # (idempotent by construction — same text in, same vector out) and no
    # fallback path exists yet if it fails, so paying for a couple more
    # bounded retries before giving up is the right trade.
    embedding_timeout_s: float = 10.0
    embedding_max_retries: int = 2

    # ADR-0028: the offline judge (evals/judge.py's `judge()`, constructed in
    # evals/run_answer_eval.py + evals/build_calibration_set.py) — a
    # dev/CI-only batch call, not a request a user is waiting on, but still
    # worth an explicit budget rather than an unbounded hang stalling a CI
    # job. Matches the other nano-tier guard clients' timeout.
    judge_timeout_s: float = 10.0

    # ADR-0028: the ONE request-wide deadline `_run_graph_and_stream`
    # (api.py, shared by `/ask` and `/resume`) enforces across every
    # `graph.astream(...)` node it visits — distinct from any single LLM
    # call's own timeout above: this bounds the WHOLE request (guard_in +
    # router + retrieve + up to `MAX_ATTEMPTS` answer calls + critic,
    # build.py's `MAX_LLM_CALLS_PER_REQUEST`), not any one call in it. 60s
    # is comfortably above `answer_timeout_s` plus every other call's own
    # budget summed, so a healthy request never trips this — it's a
    # backstop against the whole chain running unusually long, not a tight
    # budget nodes are expected to race against. ADR-0028 round 2: this
    # bounds the app's OWN graph-compute time only — a slow SSE client
    # reading the response never counts against it (see the ADR for why a
    # naive wall-clock deadline gets that wrong).
    request_timeout_s: float = 60.0

    # ADR-0030 (Day 25 security review): `TrustedHostMiddleware`'s
    # (api.py) allowlist — a request whose `Host` header isn't in this list
    # gets a 400 before routing even runs. Defaults cover local dev
    # (localhost/127.0.0.1) and the fixed "testserver" host FastAPI's/
    # httpx's `TestClient` sends, so the test suite stays green with no
    # special-casing. A real deploy MUST override this to its actual public
    # hostname(s) — this default has no wildcard and rejects everything
    # else, deliberately.
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]

    # ADR-0029: `costing.estimate_cost()`'s USD->EUR multiplier — a
    # point-in-time snapshot (ECB reference rate, 2026-08-30 ≈ 0.92 EUR per
    # USD), NOT a live FX feed. Same staleness discipline as `costing.
    # PRICES`'s dated rows: this drifts as the real exchange rate moves, so
    # re-check it before quoting a euro figure anyone will act on months
    # from now — that's the honest scope for a portfolio project (no paid
    # FX API, no live feed dependency for a number this small).
    eur_usd_rate: float = 0.92

    # ADR-0031: `guards/quotes.py`'s `quote_matches()` fuzzy-fallback floor —
    # a citation whose quote misses the exact normalised-substring check
    # (ADR-0014's `_normalise`) is accepted only when its best windowed
    # `difflib.SequenceMatcher` similarity is >= this. 0.92 is the tuned
    # value from ADR-0031's measured fixture search: high enough that every
    # adversarial fixture tried (wrong-meaning same-words, spliced
    # half-sentences, cross-article quotes) scores below it, low enough to
    # accept the realistic cosmetic-drift fixtures (punctuation swap,
    # collapsed ellipsis, whitespace/newline drift, a dropped "(1)" marker).
    # One-line kill switch back to verbatim-only: 1.0 — no fuzzy score ever
    # reaches exactly 1.0 short of a match the exact substring check would
    # already have caught, so this value alone is "fuzzy fallback disabled."
    quote_similarity_min: float = 0.92

    # ADR-0032: `build.py`'s `_mcp_connection()` spawns the MCP server
    # subprocess. `True` (dev/CI, unchanged default) runs it via
    # `uv run --frozen python -m ...`. The production image
    # (Dockerfile) has no `uv` binary at all — only the synced `.venv` is
    # copied into the runtime stage — so it sets `MCP_USE_UV_RUN=false` to
    # spawn a bare `python -m compliance_copilot.mcp_server` instead, which
    # works identically since that venv already has the package installed.
    # An image-level fact, not a per-deploy tuning knob — set in the
    # Dockerfile's `ENV`, not `.env.example`.
    mcp_use_uv_run: bool = True


settings = Settings()
