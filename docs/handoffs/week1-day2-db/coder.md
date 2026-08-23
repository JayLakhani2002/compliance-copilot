# coder.md — Week 1 Day 2: Postgres + pgvector base schema

## Status
Done, unit-green locally. Integration test not run locally (no Docker on
this machine — see `docs/INBOX.md`); designed to run in GitHub CI instead.

## Files added/changed
- `docker-compose.yml` — new. `postgres` service, image `pgvector/pgvector:pg16`, healthcheck, named volume, port 5432.
- `src/compliance_copilot/settings.py` — new. `pydantic-settings` `Settings` (`DATABASE_URL` from `.env`, `extra="ignore"` for future keys).
- `src/compliance_copilot/db.py` — new. `Base(DeclarativeBase)`, `Document`, `Chunk` (embedding `Vector(1536)`, `chunk_metadata JSONB`), `chunk_embedding_hnsw_idx` (HNSW, `vector_cosine_ops`, m=16/ef_construction=64), `init_db(engine)`, module-level `engine`/`SessionLocal`/`get_session`. `# ponytail:` note: no Alembic yet, add on schema's 2nd change.
- `src/compliance_copilot/cli.py` — new. `python -m compliance_copilot.cli init-db` (argparse).
- `tests/test_db_models.py` — new, unit, no DB: column sets, FK, `Vector.dim == 1536`, `CreateIndex(...).compile(dialect=postgresql.dialect())` contains `hnsw` + `vector_cosine_ops`.
- `tests/test_db_integration.py` — new, `@pytest.mark.integration`, `pytest.skip(..., allow_module_level=True)` when `DATABASE_URL` unset. Inserts Document+Chunk with a random 1536-vector, queries via `Chunk.embedding.cosine_distance(vector)`, asserts round trip.
- `.github/workflows/ci.yml` — added `integration` job: `services.postgres` = `pgvector/pgvector:pg16`, `pg_isready` healthcheck options, `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres`, runs `uv run pytest -m integration`. Existing `test` job untouched.
- `.env.example`, `Makefile` (`db-up`, `db-init`), `README.md` ("How to run") updated.
- `uv add sqlalchemy "psycopg[binary]" pgvector pydantic-settings` → pinned in `uv.lock`: sqlalchemy 2.0.52, psycopg/psycopg-binary 3.3.4, pgvector 0.5.0, pydantic-settings 2.15.0 (pydantic 2.13.4).

## Doc URLs consulted (Context7 + WebFetch, 2026-08-23)
- SQLAlchemy 2.0 ORM `DeclarativeBase`/`Mapped`/`mapped_column`: Context7 `/websites/sqlalchemy_en_20_orm`, https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html, declarative_styles.html
- SQLAlchemy `Index`/`CreateIndex` compile-to-string: Context7 `/websites/sqlalchemy_en_20_core`, https://docs.sqlalchemy.org/en/20/core/ddl.html, constraints.html
- `pgvector.sqlalchemy.Vector` (alias of `VECTOR`), `postgresql_using="hnsw"` / `postgresql_with={"m":16,"ef_construction":64}` / `postgresql_ops={"embedding":"vector_cosine_ops"}`, `.cosine_distance()`: Context7 `/pgvector/pgvector-python`, https://github.com/pgvector/pgvector-python/blob/master/README.md
- `pydantic-settings` `BaseSettings`/`SettingsConfigDict(env_file=".env", extra="ignore")`: Context7 `/pydantic/pydantic-settings`, https://github.com/pydantic/pydantic-settings/blob/main/docs/index.md
- Docker Hub `pgvector/pgvector` tags — confirmed `pg16` exists (pushed 10 days prior, linux/amd64+arm64): https://hub.docker.com/r/pgvector/pgvector/tags
- GitHub Actions `services:` postgres + health-check `options:` syntax: https://docs.github.com/en/actions/using-containerized-services/creating-postgresql-service-containers

## What I couldn't verify
- `tests/test_db_integration.py` has not actually been run against a live Postgres — no Docker locally. It will get its first real run in GitHub CI's new `integration` job on this branch's next push. If it fails there, the most likely culprits are: the `vector` extension not being pre-installed in the exact `pg16` tag pulled by Actions (unlikely — same tag `docker-compose.yml` uses), or a JSONB `chunk_metadata` default not round-tripping as expected.

## Commands run + summary
- `uv add sqlalchemy "psycopg[binary]" pgvector pydantic-settings` — succeeded, `uv.lock` updated.
- `uv run ruff check .` → `All checks passed!`
- `uv run ruff format --check .` → `42 files already formatted`
- `uv run pytest -m "not integration"` → `6 passed, 1 skipped in 4.09s` (the 1 skip is `test_db_integration.py`'s module-level skip, as designed).

## Next step
Reviewer pass, then push branch + open PR into `develop` (not done — per CLAUDE.md, reviewer goes first). First real CI run will exercise the new `integration` job against the pgvector service container.
