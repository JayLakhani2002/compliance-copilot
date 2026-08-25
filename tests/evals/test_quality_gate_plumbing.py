# tests/evals/test_quality_gate_plumbing.py — proves CachedEmbeddings
# (ADR-0017) is a genuine drop-in for the real Embeddings interface all the
# way through ingest() -> retrieve(), not just in isolation. Builds a tiny
# in-test cache file from FakeEmbeddings' deterministic vectors (tests/
# fake_embeddings.py — no network, no OpenAI cost) for the 3-article
# fixture corpus (tests/fixtures/eurlex_sample.xhtml, same one
# fixture_regulations/tests/evals/test_retrieval_plumbing.py already use),
# then ingests and retrieves through it exactly like CI's quality-gate job
# does against the real committed cache.
import hashlib
import json
from pathlib import Path

import pytest
from fake_embeddings import FakeEmbeddings
from sqlalchemy.orm import Session

from compliance_copilot.cached_embeddings import CachedEmbeddings, encode_vector
from compliance_copilot.ingest import pipeline
from compliance_copilot.ingest.chunker import split_article
from compliance_copilot.ingest.eurlex import parse_articles
from compliance_copilot.retriever import retrieve
from compliance_copilot.settings import settings

FIXTURE_XHTML = Path(__file__).parent.parent / "fixtures" / "eurlex_sample.xhtml"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.integration
def test_cached_embeddings_round_trips_through_ingest_and_retrieve(
    test_engine, fixture_regulations, tmp_path
):
    fake = FakeEmbeddings()
    xhtml = FIXTURE_XHTML.read_text(encoding="utf-8")
    parsed = parse_articles(xhtml, "ai_act")
    parts = [p for article in parsed for p in split_article(article)]
    chunk_texts = [p.text for p in parts]
    # Recitals precede articles in the source XHTML's document order (EU
    # legal drafting convention), so parts[0] can be a recital — but
    # retrieve() below is called with kinds=("article",) only (ADR-0013's
    # default), which can never match a recital's own text. Query with an
    # ARTICLE chunk's own text specifically, so "nearest is itself" is
    # actually checkable against what's being searched.
    query_text = next(p.text for p in parts if p.kind == "article")

    # Build a tiny cache covering every chunk text ingest() will look up,
    # plus the one query text retrieve() will look up — same {sha256,
    # model, dim, vec} shape evals/cache_embeddings.py writes for real.
    cache_path = tmp_path / "cache.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for text in {*chunk_texts, query_text}:
            vector = fake.embed_query(text)
            entry = {
                "sha256": _sha256(text),
                "model": settings.embedding_model,
                "dim": len(vector),
                "vec": encode_vector(vector),
            }
            f.write(json.dumps(entry) + "\n")

    cached = CachedEmbeddings(cache_paths=(cache_path,), model=settings.embedding_model)

    with Session(test_engine) as session:
        pipeline.ingest("ai_act", cached, session)  # embed_documents() reads the cache, not OpenAI

        results = retrieve(query_text, k=1, kinds=("article",), session=session, embeddings=cached)

    assert results[0].text == query_text
