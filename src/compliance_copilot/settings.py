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


settings = Settings()
