# tests/evals/test_retrieval_plumbing.py — Day 5 eval plumbing tests
# (ADR-0005). Two things checked, neither of which touches OpenAI:
# (1) evals/golden_retrieval.jsonl parses and its expected_anchors are
#     genuinely real — every anchor matches the "art_N" id pattern
#     eurlex.py produces, and (when the real cached XHTML is on disk,
#     data/raw/ — gitignored, see .gitignore) actually exists in the parsed
#     corpus for the regulation(s) the golden entry names.
# (2) retriever.py's `retrieve()` — kind filter, k, and distance ordering —
#     against the disposable test DB with FakeEmbeddings (tests/
#     fake_embeddings.py), same pattern as
#     test_ingest_pipeline_integration.py: costs no OpenAI money, so this
#     is "unit-level" in spirit despite needing a real Postgres+pgvector,
#     hence the `integration` marker (excluded from `pytest -m "not
#     integration"`, included in the free DB-backed CI job) rather than the
#     RUN_NETWORK_TESTS-gated real-embedding marker.
import re
from pathlib import Path

import pytest
from fake_embeddings import FakeEmbeddings
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk
from compliance_copilot.ingest import pipeline
from compliance_copilot.ingest.eurlex import REGULATIONS, parse_articles
from compliance_copilot.retriever import retrieve
from evals.run_retrieval_eval import GOLDEN_PATH, load_golden

pytestmark = pytest.mark.integration

# eurlex.py's own anchor format (art_N — see eurlex.py's _ARTICLE_ID_RE) —
# every golden expected_anchor must look like this, since the eval only
# ever compares against Chunk.anchor_id values the real parser produces.
_ANCHOR_RE = re.compile(r"art_\d+")

DATA_RAW = Path("data/raw")


def test_golden_file_parses_and_has_the_documented_mix():
    entries = load_golden()
    assert len(entries) == 30

    categories = [e.category for e in entries]
    # Lesson 05's mix: ~10 definitional, ~8 obligation, ~6 scope,
    # ~3 cross-regulation, ~3 German.
    assert categories.count("definition") == 10
    assert categories.count("obligation") == 8
    assert categories.count("scope") == 6
    assert categories.count("cross") == 3
    assert categories.count("german") == 3

    assert len({e.id for e in entries}) == 30  # ids unique
    assert sum(1 for e in entries if e.regulation == "gdpr") >= 8  # brief's "at least 8 GDPR"


def test_expected_anchors_match_the_real_anchor_pattern():
    for entry in load_golden():
        assert entry.expected_anchors, f"{entry.id}: no expected_anchors"
        for anchor in entry.expected_anchors:
            assert _ANCHOR_RE.fullmatch(anchor), f"{entry.id}: {anchor!r} doesn't match art_\\d+"


def test_expected_anchors_exist_in_the_real_corpus_when_cached_xhtml_is_present():
    """Only runs the existence check when data/raw's cached XHTML is on disk
    (a dev machine that's run `ingest`, not a fresh CI checkout — data/raw
    is gitignored). Skips rather than falling back to the tiny 3-article
    fixture for this check: the fixture doesn't have enough of the corpus
    for a real existence check to mean anything, so asserting against it
    would either vacuously pass or fail every real golden entry for the
    wrong reason (missing from the fixture, not wrong)."""
    available = {
        key: DATA_RAW / f"{meta['celex']}.xhtml"
        for key, meta in REGULATIONS.items()
        if (DATA_RAW / f"{meta['celex']}.xhtml").exists()
    }
    if not available:
        pytest.skip("data/raw/ not populated — run `ingest` locally to enable this check")

    anchors_by_regulation = {
        key: {c.anchor_id for c in parse_articles(path.read_text(encoding="utf-8"), key)}
        for key, path in available.items()
    }

    for entry in load_golden():
        # "any" (cross-regulation entries) may draw anchors from either
        # regulation; a specific regulation must find its anchor there.
        candidate_regs = (
            [entry.regulation]
            if entry.regulation in anchors_by_regulation
            else list(anchors_by_regulation)
        )
        for anchor in entry.expected_anchors:
            assert any(anchor in anchors_by_regulation[r] for r in candidate_regs), (
                f"{entry.id}: {anchor!r} not found in the parsed corpus for {candidate_regs} "
                f"({GOLDEN_PATH} — fix the golden entry if it's factually wrong, per this file's "
                f"header rule: golden sets are edited for correctness, not for score)"
            )


def test_kind_filter_excludes_recitals(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        results = retrieve(
            "a question about AI systems",
            k=5,
            kinds=("article",),
            session=session,
            embeddings=embeddings,
        )

        assert results  # fixture has 3 articles — something should come back
        assert all(r.kind == "article" for r in results)


def test_k_is_respected(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        # fixture has 5 chunks total (3 articles + 2 recitals) — k=2 must
        # cap the result list, not just be a hint.
        results = retrieve(
            "a question", k=2, kinds=("article", "recital"), session=session, embeddings=embeddings
        )

        assert len(results) == 2


def test_ordered_by_distance_nearest_first(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        art1_text = session.scalars(select(Chunk).where(Chunk.anchor_id == "art_1")).first().text
        results = retrieve(
            art1_text, k=3, kinds=("article",), session=session, embeddings=embeddings
        )

        assert results[0].anchor == "art_1"  # querying with its own text -> itself is nearest
        assert results[0].distance <= results[1].distance <= results[2].distance
