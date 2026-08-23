# tests/test_db_models.py — unit tests for the ORM schema in db.py. No real
# DB connection (Docker isn't installed on this dev machine yet, see
# docs/INBOX.md) — these only check that the models/DDL are shaped right by
# inspecting SQLAlchemy's in-memory metadata and compiling DDL to a string,
# never executing it. Real-DB behavior is covered by test_db_integration.py
# in CI (a pgvector service container), per the same "unit vs integration"
# split CLAUDE.md/ADR-0011 already use for the rest of the test suite.
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from compliance_copilot.db import Chunk, Document, chunk_embedding_hnsw_idx


def test_document_columns():
    columns = {c.name for c in Document.__table__.columns}
    assert columns == {"id", "regulation", "title", "source_url", "fetched_at"}


def test_chunk_columns():
    columns = {c.name for c in Chunk.__table__.columns}
    assert columns == {
        "id",
        "document_id",
        "article",
        "recital",
        "title",
        "text",
        "embedding",
        "chunk_metadata",
    }


def test_chunk_document_id_is_foreign_key():
    fk_targets = {fk.target_fullname for fk in Chunk.__table__.foreign_keys}
    assert fk_targets == {"document.id"}


def test_chunk_embedding_is_vector_1536():
    embedding_col = Chunk.__table__.c.embedding
    # pgvector.sqlalchemy.Vector stores its fixed dimension as .dim
    assert embedding_col.type.dim == 1536


def test_hnsw_index_ddl_compiles_with_cosine_ops():
    # Compile the CREATE INDEX statement to a string without ever connecting
    # to a database — this is what proves the Index(...) kwargs from db.py
    # actually produce valid HNSW/cosine DDL, the exact syntax verified
    # against Context7's pgvector-python docs before writing db.py.
    ddl = str(CreateIndex(chunk_embedding_hnsw_idx).compile(dialect=postgresql.dialect()))
    assert "hnsw" in ddl.lower()
    assert "vector_cosine_ops" in ddl
