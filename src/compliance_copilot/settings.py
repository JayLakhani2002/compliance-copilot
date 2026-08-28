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


settings = Settings()
