# reviewer.md — Week 1 Day 2: Postgres + pgvector base schema

## Verdict: APPROVE (with 2 minor notes, no blockers/majors)

Read: CLAUDE.md, docs/GLOSSARY.md, ADR-0003, ADR-0004, docs/lessons/02_postgres_pgvector.md,
coder.md, docs/CURRICULUM.md (Week 1 table), and every file in
`git diff develop...feature/db-pgvector` (15 files, 772 insertions) in full.

Ran locally (no Docker on this machine, matching coder's note):
- `uv run pytest -m "not integration" -q` → **6 passed, 1 skipped** — matches coder's claim.
- `uv run ruff check .` → **All checks passed!**
- `uv run ruff format --check .` → **43 files already formatted**

## Findings

1. **[minor] Scope pull-forward — HNSW index built on Day 2, curriculum assigns it to Day 4.**
   `docs/CURRICULUM.md` line 9 (Day 2 Build): `docker-compose.yml`, `db.py` connection +
   migration for `documents`, `chunks(embedding vector)` — no index. Line 11 (Day 4 Build):
   "`embeddings.py` + ingest CLI writes chunks+vectors; **HNSW index**". `src/compliance_copilot/db.py:80-86`
   (`chunk_embedding_hnsw_idx`) plus its dedicated unit test
   (`tests/test_db_models.py:44-51`) is Day 4 curriculum content shipped on Day 2. Not a
   defect — the DDL is correct (verified below) and defining an index alongside its column
   is normal schema practice — but it front-runs the pedagogical sequencing CLAUDE.md cares
   about ("Teach before coding each feature"): a reader following the curriculum day-by-day
   hits HNSW/cosine_distance code and a "verified against Context7's pgvector-python docs"
   comment (`db.py:78`) before Day 4's lesson on what any of that means. Lesson 02
   (Day 2's own lesson doc) doesn't cover HNSW in depth either.
   **Fix (optional, non-blocking):** either move `chunk_embedding_hnsw_idx` + its test into
   the Day 4 branch/commit, or add one sentence to lesson 02 flagging "the index is built now
   but explained on Day 4" so the sequencing gap is intentional and visible rather than silent.
   Given the code is correct and doesn't block anything, leaving it as-is and noting it in
   lesson 02 is the lower-effort fix.

2. **[minor] docker-compose Postgres port binds to all interfaces, not just localhost.**
   `docker-compose.yml:22-25` — `ports: ["5432:5432"]` binds Postgres to `0.0.0.0:5432` on
   the host, not `127.0.0.1:5432`. Combined with the hardcoded dev credentials (`user`/`password`,
   correctly flagged in-file as "Local dev only, not a real secret" per `docker-compose.yml:14-17`,
   which satisfies CLAUDE.md's "say so explicitly" bar for default passwords), this means on
   any machine/network where the host's firewall doesn't block inbound 5432, the DB is reachable
   from the LAN, not just the developer's own machine — ADR-0003's stated trust boundary
   ("internal Docker network only, no public exposure") is written for the *deployed* stack
   (`docs/ARCHITECTURE.md` §6), so this isn't a contradiction, but the dev compose file doesn't
   carry the same guarantee and doesn't say so.
   **Fix:** change to `"127.0.0.1:5432:5432"` (one-line change, no behavior loss for local dev
   since `DATABASE_URL` already targets `localhost`), or add a comment explicitly stating why
   binding to all interfaces is acceptable if that's a deliberate choice.

## Verified against Context7 / official docs (spot-checks, all confirmed correct)

- `Index(..., postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding": "vector_cosine_ops"})` — matches `pgvector-python` README's SQLAlchemy HNSW example verbatim (`/pgvector/pgvector-python`). `m=16`/`ef_construction=64` confirmed as pgvector's own documented defaults (`/pgvector/pgvector` configuration docs).
- `Vector(1536)` — `pgvector.sqlalchemy.Vector`/`VECTOR(dim)` constructor confirmed; 1536 matches ADR-0004's default embedding model, `text-embedding-3-small`.
- `.cosine_distance()` — confirmed as a real `pgvector.sqlalchemy.VECTOR` comparator method (`<=>` operator, returns `Float`).
- `SettingsConfigDict(env_file=".env", extra="ignore")` — matches `pydantic-settings` docs' own canonical dotenv-compatibility example verbatim.
- GitHub Actions `services.postgres` + `options: --health-cmd ... --health-interval ... --health-timeout ... --health-retries ...` (`.github/workflows/ci.yml:63-67`) — matches GitHub's own documented postgres-service-container syntax.
- Checked one thing I suspected might be a real bug and it wasn't: `pgvector-python`'s README instructs registering `pgvector.psycopg.register_vector` via an `event.listens_for(engine, "connect")` hook for SQLAlchemy+psycopg3 — but confirmed (by reading the README's actual section headers, not just a snippet) that this registration is only required for the `ARRAY(VECTOR(n))` array-of-vectors case. A plain `Vector(1536)` column's SQLAlchemy `UserDefinedType` (`pgvector/sqlalchemy/vector.py`) does its own `bind_processor`/`result_processor` string conversion and needs no driver-level registration — `db.py`'s plain column usage is correct without it.

## Correctness (dimension 1) — will `init_db` work on a fresh DB?

Yes. `db.py:105-112`: `CREATE EXTENSION IF NOT EXISTS vector` runs and commits (via `engine.begin()`)
before `Base.metadata.create_all(engine)` — correct order, since the `vector` column type doesn't
exist until the extension is installed. `chunk_embedding_hnsw_idx` is constructed against
`Chunk.embedding` (a real `Column`), which auto-registers it on `chunk`'s `Table.indexes` —
standard SQLAlchemy behavior — so `create_all` emits its `CREATE INDEX` alongside the table DDL,
confirmed by `test_hnsw_index_ddl_compiles_with_cosine_ops` compiling real DDL text. CI's
`integration` job env (`postgresql+psycopg://postgres:postgres@localhost:5432/postgres`) matches
its own service container's `POSTGRES_USER`/`PASSWORD`/`DB`, and the `pg16` image tag matches
`docker-compose.yml`'s, so dev/CI stay in sync. GH Actions waits for the service's
`pg_isready` healthcheck before running steps, so `uv run pytest -m integration` shouldn't race
Postgres startup.

## Security (dimension 2)

No secrets logged (`cli.py`'s only output is `"Database initialised."`). No SQL injection
surface — the one raw SQL string (`CREATE EXTENSION IF NOT EXISTS vector`) is static, no
interpolation. `.env` is correctly gitignored (`.gitignore:20-23`) with only `.env.example`
(placeholder values) committed. Default dev credentials are explicitly flagged as
dev-only in-file. See finding #2 for the one gap (host binding).

## Teaching quality (dimension 3)

Every changed source file has a header comment stating purpose + place in architecture
(`db.py:1-12`, `settings.py:1-10`, `cli.py:1-5`, both test files' headers). "Why" comments
are placed at genuinely non-obvious decision points (extension-before-create_all ordering,
HNSW param meaning, why `env_file`/settings centralization, why module-level skip in the
integration test) and none are misleading — spot-checked the HNSW/cosine comment and the
`m=16`/`ef_construction=64` "pgvector's own defaults" claim against Context7; both accurate.
The `# ponytail:` note on deferring Alembic (`db.py:9-12`) correctly names the ceiling
("add on schema's 2nd change") per R9's own convention.

## Scope (dimension 5)

See finding #1 (HNSW index is Day 4 content, shipped Day 2 — non-blocking). Everything else
matches Day 2's curriculum line item almost exactly: `docker-compose.yml`, `db.py` connection +
schema for `documents`/`chunks(embedding vector)`, integration test that connects and creates
tables. The `cli.py init-db` command and `Makefile` `db-up`/`db-init` targets aren't literally
named in the curriculum row but are the minimum needed to make "integration test connects,
creates tables" operable by a human outside of pytest — reasonable, not gold-plated (27 lines,
one argparse subcommand, no framework).

## Next step

No fixes required before merge — both findings are minor and non-blocking per this review.
Coder/planner call on whether to act on finding #1 (defer HNSW to Day 4 vs. note the pull-forward
in lesson 02) and #2 (bind compose port to 127.0.0.1). Ready to push branch + open PR into
`develop` per coder.md's stated next step.
