# tests/conftest.py — shared fixtures for DB-backed integration tests
# (ADR-0003). Every integration test connects to a disposable *test*
# database, never to whatever DATABASE_URL points at in a developer's shell.
#
# WHY THIS EXISTS (read before touching it): an earlier version of this
# suite ran `init_db(engine, reset=True)` directly against DATABASE_URL —
# `uv run pytest -m integration`, the documented "free, no-OpenAI-cost"
# command — and that DROPped a real 576-chunk ingested corpus with zero
# warning. A shared dev/test DB is how you lose data. `db.py`'s `init_db`
# now refuses `reset=True` unless the target DB name ends in "_test" (or the
# caller passes `force=True`, which nothing in this file does) — this
# fixture's job is to make sure that name is always what tests actually get.
#
# Resolution order: `TEST_DATABASE_URL` if set (CI sets this explicitly,
# same server as DATABASE_URL, a "_test"-suffixed db name); otherwise
# DATABASE_URL with its database name suffixed "_test" (local dev default —
# `compliance_copilot` -> `compliance_copilot_test`).
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from compliance_copilot.db import init_db
from compliance_copilot.ingest import eurlex, pipeline

FIXTURE_XHTML = Path(__file__).parent / "fixtures" / "eurlex_sample.xhtml"


def _resolve_test_database_url() -> str | None:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = os.environ.get("DATABASE_URL")
    if not base:
        return None
    url = make_url(base)
    # render_as_string(hide_password=False), NOT str(url)/repr(url) — those
    # mask the password as "***" (safe for logging, useless as an actual
    # connection string) and this return value gets handed straight to
    # create_engine().
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The disposable test database's URL, guaranteed to exist by the time
    this returns. Skips (not errors) when no DB is configured at all — same
    "no Docker locally" escape hatch the old per-file skips had."""
    url = _resolve_test_database_url()
    if not url:
        pytest.skip("DATABASE_URL/TEST_DATABASE_URL not set — skipping DB integration tests")

    target = make_url(url)
    # CREATE DATABASE can't run inside a transaction block — AUTOCOMMIT is
    # the documented SQLAlchemy pattern for DDL like this (docs.sqlalchemy.
    # org/en/20/core/connections.html, "DBAPI-level autocommit"), verified
    # against Context7 before writing this. Connect to Postgres's own
    # "postgres" maintenance DB to issue it, since the target DB may not
    # exist yet to connect to directly.
    admin_engine = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            ).first()
            if not exists:
                # Database names can't be bind parameters in DDL; quoting
                # with double quotes is safe here because this name is
                # always our own derived "<name>_test" or an operator-set
                # TEST_DATABASE_URL, never untrusted input.
                conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin_engine.dispose()

    return url


@pytest.fixture
def test_engine(test_database_url):
    """A fresh engine against the test DB with the schema reset for this
    test. `reset=True` only succeeds because `test_database_url` always
    names a "_test"-suffixed database — `db.py`'s guard would refuse this
    against anything else."""
    engine = create_engine(test_database_url)
    init_db(engine, reset=True)
    return engine


@pytest.fixture
def fixture_regulations(monkeypatch):
    """Patches `ingest` to use the small, real, no-network fixture XHTML
    (tests/fixtures/eurlex_sample.xhtml — 3 articles, 2 recitals of the
    real AI Act) instead of hitting Cellar, and patches the expected
    article/recital counts so `ingest_regulation`'s sanity check (eurlex.py)
    doesn't fire against those smaller numbers. Shared across every
    integration test that needs a small ingested corpus in the test DB
    without spending OpenAI money or a network call — originally
    test_ingest_pipeline_integration.py's, moved here (ADR-0013) so
    tests/evals/test_retrieval_plumbing.py can reuse it via normal pytest
    fixture discovery instead of a cross-test-file import."""
    xhtml = FIXTURE_XHTML.read_text(encoding="utf-8")
    monkeypatch.setattr(pipeline, "fetch_xhtml", lambda celex, cache_dir=None: xhtml)
    monkeypatch.setitem(eurlex.REGULATIONS["ai_act"], "expected_articles", 3)
    monkeypatch.setitem(eurlex.REGULATIONS["ai_act"], "expected_recitals", 2)
