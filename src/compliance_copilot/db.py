# src/compliance_copilot/db.py — the "postgres" container's schema and
# connection layer (docs/ARCHITECTURE.md §3). Defines the two tables the
# ingestion job (§1) writes into and the MCP `search_regulation`/`get_article`
# tools (§1, ADR-0007) read from: Document (one row per regulation source
# file) and Chunk (one row per article/recital, holding its embedding for
# similarity search). ADR-0003 explains why this lives in Postgres/pgvector
# rather than a dedicated vector DB.
#
# ponytail: no Alembic migrations yet — `init_db()` just does
# `Base.metadata.create_all`, which is fine for a schema that hasn't shipped
# to anyone yet. Add Alembic the *second* time this schema changes (the
# create_all point where a fresh DB is no longer everyone's DB).
#
# Why the engine is built lazily (get_engine(), not a module-level `engine`):
# a module-level `create_engine(settings.database_url)` runs at *import*
# time — so anything that imports this module (a test file that only needs
# the ORM models, a linter, `python -c "import compliance_copilot.db"`) was
# forced to have a parseable DATABASE_URL, even when it never opens a
# connection. That bit a real run: `DATABASE_URL= pytest -k eurlex` failed at
# *collection* of unrelated test files with a SQLAlchemy URL-parse error,
# before pytest's `-k` filter ever got a chance to skip them. Deferring
# engine creation to first use (functools.lru_cache, so it's still built
# once and reused) makes import side-effect-free again.
from collections.abc import Generator
from datetime import datetime
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from compliance_copilot.settings import settings


class Base(DeclarativeBase):
    """Shared declarative base — every ORM model inherits from this so
    `Base.metadata` knows about all tables at once (needed by create_all)."""


class Document(Base):
    """One row per ingested source file (e.g. the AI Act, GDPR)."""

    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    regulation: Mapped[str] = mapped_column(String(50))  # REGULATIONS key, e.g. "ai_act"
    celex: Mapped[str] = mapped_column(String(20))  # e.g. "32024R1689" (eurlex.py's CELEX id)
    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    """One row per article/recital *part* (ingest/chunker.py — most articles
    are one whole part; oversize ones like Art. 3 split into several) plus
    its embedding — the unit the vector search actually ranks
    (docs/ARCHITECTURE.md §1).

    ponytail: this replaces the old `article`/`recital` string columns with
    `kind`+`number`+`anchor_id`, which is the same identity ChunkPart already
    carries — keeping both would just be two ways to say the same thing.
    Safe to do as a plain column rename/reshape (not an Alembic migration)
    because the DB is dev-only and nothing has shipped against this schema
    yet (see init_db's `reset` flag below).
    """

    __tablename__ = "chunk"
    __table_args__ = (
        # One row per (document, anchor, part) — this is what the ingest
        # pipeline's upsert (ingest/pipeline.py) keys off of: re-running
        # ingest on unchanged text must update the same row, not insert a
        # duplicate.
        UniqueConstraint("document_id", "anchor_id", "part", name="uq_chunk_document_anchor_part"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    kind: Mapped[str] = mapped_column(String(10))  # "article" | "recital"
    number: Mapped[int] = mapped_column(Integer)  # article/recital number, e.g. 3
    anchor_id: Mapped[str] = mapped_column(String(20))  # e.g. "art_3", "rct_12"
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    # part/part_count: this chunk's position among its parent article's parts
    # (ingest/chunker.py's split_article) — 0/1 for the common whole-article
    # case, 0..N-1 of N for an oversize article split on paragraph boundaries.
    part: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    part_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    # sha256(text) of *this part's* text — lets the ingest pipeline skip
    # re-embedding a part whose text hasn't changed since the last ingest
    # (lesson 04's idempotency point).
    content_hash: Mapped[str] = mapped_column(String(64))
    # 1536 = text-embedding-3-small's output dimension (ADR-0004). Fixed (not
    # variable) dimension so a mismatched embedding model fails loudly at
    # insert time instead of silently corrupting the nearest-neighbour index.
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    # Which model produced `embedding` — ADR-0004's consistency requirement
    # (ingestion and query-time embedding must use the same model) means this
    # is worth recording per-row, not just assuming from settings: it's the
    # signal that would catch a future re-ingest-with-a-different-model bug.
    embedding_model: Mapped[str] = mapped_column(String(100))

    document: Mapped["Document"] = relationship(back_populates="chunks")


# HNSW (Hierarchical Navigable Small World) = an approximate-nearest-neighbour
# graph index — sub-linear query time instead of scanning every row's vector.
# vector_cosine_ops = the operator class pgvector needs to build the index
# for cosine distance specifically, matching OpenAI embeddings, which are
# normalised so cosine similarity is the intended metric (ADR-0003).
# m/ef_construction are pgvector's documented HNSW build parameters (higher =
# better recall, slower build/more memory); 16/64 are pgvector's own defaults.
chunk_embedding_hnsw_idx = Index(
    "chunk_embedding_hnsw_idx",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# lru_cache(maxsize=1) as a "compute once, on first call" memoiser — not a
# real cache keyed by input (get_engine takes no args), just a lazy
# module-level singleton so `settings.database_url` is only read, and the
# engine only built, the first time a connection is actually needed.
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """The app's one SQLAlchemy engine, built from `settings` on first use
    (see the "why lazy" comment at the top of this file)."""
    return create_engine(settings.database_url)


def get_session() -> Generator[Session, None, None]:
    """Yields a Session bound to get_engine(), closing it afterwards — a
    context-manager-style dependency other code (FastAPI routes, later) can
    plug into."""
    with Session(get_engine()) as session:
        yield session


def init_db(engine, *, reset: bool = False) -> None:  # engine explicit — tests pass a throwaway one
    """Creates the `vector` extension, then all tables/indexes. Idempotent —
    safe to call against a DB that already has the schema (CREATE EXTENSION
    IF NOT EXISTS / create_all both no-op on existing objects).

    `reset=True` DROPS every table this app owns first (document, chunk —
    NOT the `vector` extension itself) and recreates them empty. Only safe
    because the DB is dev-only and nothing has shipped against this schema
    (ponytail: no Alembic yet — see the module docstring above; add it the
    *next* time this schema changes, since a `reset` flag stops being
    acceptable the moment a real user's data could be sitting in these
    tables). Never call this against anything but a local/dev database.
    """
    if reset:
        # DESTRUCTIVE: drops document/chunk and everything in them. This is
        # only reachable via the CLI's `init-db --reset` flag (opt-in, never
        # the default) — see cli.py.
        Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        # Must run before create_all: the Vector column type doesn't exist
        # in Postgres until this extension is installed.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # chunk_embedding_hnsw_idx was constructed against Chunk.embedding above,
    # which auto-registers it on chunk's Table — create_all emits its
    # CREATE INDEX statement along with every table's CREATE TABLE.
    Base.metadata.create_all(engine)
