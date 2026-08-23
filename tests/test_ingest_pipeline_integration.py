# tests/test_ingest_pipeline_integration.py — integration test for
# ingest/pipeline.py against a real Postgres+pgvector instance (needs
# DATABASE_URL/TEST_DATABASE_URL — runs in CI's pgvector job). Uses
# FakeEmbeddings (tests/fake_embeddings.py) so this costs no OpenAI money —
# that's what makes it safe to run in CI on every PR, unlike the
# RUN_NETWORK_TESTS=1-gated real-embedding test in
# test_search_real_embeddings_integration.py. Runs against the disposable
# test DB via conftest.py's `test_engine` fixture, never DATABASE_URL
# directly — see conftest.py's module docstring for why that matters.
import pytest

# No tests/__init__.py in this project (see test_ingest_eurlex.py etc.) —
# pytest's default "prepend" import mode puts tests/ itself on sys.path, so
# sibling test modules import as top-level names, not `tests.<name>`.
from fake_embeddings import FakeEmbeddings
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_ingest_eurlex import FIXTURE

from compliance_copilot.db import Chunk
from compliance_copilot.ingest import eurlex, pipeline

pytestmark = pytest.mark.integration


@pytest.fixture
def fixture_regulations(monkeypatch):
    """Fixture XHTML has 3 articles + 2 recitals (Art. 1-3, Rct. 1-2) — patch
    the expected counts so ingest_regulation's sanity check doesn't fire, and
    patch fetch_xhtml so no network call happens."""
    xhtml = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(pipeline, "fetch_xhtml", lambda celex, cache_dir=None: xhtml)
    monkeypatch.setitem(eurlex.REGULATIONS["ai_act"], "expected_articles", 3)
    monkeypatch.setitem(eurlex.REGULATIONS["ai_act"], "expected_recitals", 2)


def test_ingest_twice_second_run_embeds_zero(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        first = pipeline.ingest("ai_act", embeddings, session)
        assert first.chunks_total == 5  # 3 articles + 2 recitals, all under the size cap
        assert first.chunks_embedded == 5
        assert first.chunks_skipped == 0

        second = pipeline.ingest("ai_act", embeddings, session)
        assert second.chunks_total == 5
        assert second.chunks_embedded == 0
        assert second.chunks_skipped == 5


def test_nearest_neighbour_for_article_1_own_text_returns_article_1_first(
    test_engine, fixture_regulations
):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        art1 = session.scalars(select(Chunk).where(Chunk.anchor_id == "art_1")).first()
        query_vector = embeddings.embed_query(art1.text)

        nearest = session.scalars(
            select(Chunk).order_by(Chunk.embedding.cosine_distance(query_vector)).limit(1)
        ).first()

        assert nearest.anchor_id == "art_1"


def test_dry_run_does_not_commit(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        stats = pipeline.ingest("ai_act", embeddings, session, dry_run=True)
        assert stats.chunks_embedded == 5

    # a fresh session/transaction sees nothing — the dry run rolled back
    with Session(test_engine) as session:
        count = len(session.scalars(select(Chunk)).all())
        assert count == 0
