# tests/test_db_integration.py — integration test against a real
# Postgres+pgvector instance. Skipped locally (no DATABASE_URL set — Docker
# may not be installed locally); runs in GitHub
# CI's `integration` job (.github/workflows/ci.yml) against a
# pgvector/pgvector:pg16 service container. Marked `integration` per
# pyproject.toml's pytest marker, so `pytest -m "not integration"` (the
# default local/unit run) skips this file's collection cost too.
import os
import random
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from compliance_copilot.db import Chunk, Document, init_db

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.integration

# Module-level skip (not just per-test) — no point building an engine at all
# when there's nothing to connect to.
if not DATABASE_URL:
    pytest.skip("DATABASE_URL not set — skipping DB integration tests", allow_module_level=True)


def test_init_db_insert_and_nearest_neighbour_query():
    engine = create_engine(DATABASE_URL)
    init_db(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        doc = Document(
            regulation="ai_act",
            celex="32024R1689",
            title="Test Regulation",
            source_url="https://example.com/ai-act",
            fetched_at=datetime.now(UTC),
        )
        session.add(doc)
        session.flush()  # assigns doc.id without committing yet

        # random.random() (0.0-1.0), not normalised — fine here since this
        # test only checks the round trip, not embedding quality.
        vector = [random.random() for _ in range(1536)]
        chunk = Chunk(
            document_id=doc.id,
            kind="article",
            number=5,
            anchor_id="art_5",
            title="Prohibited practices",
            text="Some article text.",
            content_hash="0" * 64,
            embedding=vector,
            embedding_model="text-embedding-3-small",
        )
        session.add(chunk)
        session.commit()

        # Nearest-neighbour query using the same vector: cosine_distance to
        # itself should be (near) 0, and it must be the top (only) result.
        nearest = session.scalars(
            select(Chunk).order_by(Chunk.embedding.cosine_distance(vector)).limit(1)
        ).first()

        assert nearest is not None
        assert nearest.id == chunk.id
        assert nearest.text == "Some article text."
