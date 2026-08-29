# src/compliance_copilot/checkpointer.py — durable graph state (ADR-0024).
# Sits beside db.py in the dependency graph but is deliberately its own
# module, not folded into db.py: db.py is the sync SQLAlchemy engine used by
# ingestion/retrieval, while this is an async psycopg connection pool +
# LangGraph's `AsyncPostgresSaver` — a different driver stack for a
# different job (checkpointing graph runs, not ORM queries), pulled in only
# by the two callers that need it (api.py's lifespan, cli.py's `ask`/
# `delete-thread`).
#
# `build_checkpointer()` is an async context manager (not a bare factory
# function) so both callers get the same "open, use, always close" shape
# with one line each — `async with build_checkpointer() as checkpointer:` —
# rather than each caller re-deriving its own open/close/error-handling.
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from compliance_copilot.settings import settings


def _checkpointer_dsn() -> str:
    """`settings.database_url` is SQLAlchemy's `postgresql+psycopg://` form
    — psycopg's own conninfo parser chokes on the `+psycopg` driver suffix
    (verified live: `psycopg.conninfo.conninfo_to_dict` raises
    `ProgrammingError` on it; the plain `postgresql://` form parses fine).
    `make_url(...).set(drivername="postgresql")` is the exact URL-surgery
    pattern `tests/conftest.py` already uses for its own DSN — reused here
    rather than a second ad-hoc `.replace("+psycopg", "", 1)`."""
    return (
        make_url(settings.database_url)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )


@asynccontextmanager
async def build_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Opens a connection pool + `AsyncPostgresSaver` for the lifetime of
    the `with` block, running `.setup()` once per open (idempotent — it
    creates the checkpoint tables/runs migrations only if they're missing,
    per the installed package's own `setup()` docstring) and always closing
    the pool on the way out, success or failure.

    `kwargs={"autocommit": True, "row_factory": dict_row}`: the saver's own
    SQL assumes both (Context7-verified against `libs/checkpoint-postgres/
    tests/test_async.py`'s fixture — not guessed). `open=False` + an
    explicit `await pool.open()`: `AsyncConnectionPool`'s implicit-open-on-
    construction path is deprecated in the installed `psycopg_pool` version,
    and constructing `AsyncPostgresSaver` itself calls `asyncio.get_running_
    loop()` (verified in the installed `aio.py`), so both have to happen
    inside a running event loop — this whole function is `async def` for
    that reason, not just to `await pool.open()`.

    One pool per open, not a process-wide singleton built here: the caller
    (api.py's `lifespan`, cli.py's one-shot commands) owns the lifetime —
    this function's only job is "build it right, close it right"."""
    # ponytail: psycopg_pool's default max_size (4) is plenty for one API
    # worker at portfolio traffic; expose it as a setting once concurrency
    # is actually measured, not before.
    pool = AsyncConnectionPool(
        _checkpointer_dsn(),
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer
    finally:
        await pool.close()


def validate_thread_id(value: str) -> str:
    """Raises `ValueError` unless `value` is a syntactically valid UUIDv4
    string. ADR-0024's security note: `thread_id` is meant to be
    server-issued (`uuid.uuid4()`) — a client that could supply an
    arbitrary/guessable/sequential id would be able to resume or read
    another caller's checkpointed conversation just by guessing it, so any
    client-supplied value has to be rejected unless it's at least the right
    *shape* the server itself would have issued (this does NOT close
    ADR-0016's separate, still-open gap: one shared API key means any key
    holder can still supply any *valid-shaped* thread_id and resume ANY
    thread — see ADR-0024's Security section).

    Shared by api.py's `AskRequest` field validator and cli.py's `ask`/
    `delete-thread` argument parsing, so both surfaces enforce the exact
    same rule from one place. Returns the canonical (lowercase, hyphenated)
    string form."""
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{value!r} is not a valid UUID4 thread_id") from exc
    if parsed.version != 4:
        raise ValueError(f"{value!r} is not a valid UUID4 thread_id")
    return str(parsed)
