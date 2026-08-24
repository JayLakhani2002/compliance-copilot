# tests/test_graph_real_integration.py — the one real end-to-end test for
# the graph (ADR-0014): a real Postgres+pgvector DB (test_engine, ADR-0003)
# and a real Anthropic API call (make_llm(), ADR-0002), but still
# FakeEmbeddings (tests/fake_embeddings.py) so no OpenAI cost is spent just
# to prove the graph wiring works. With fake embeddings the retrieved
# articles are effectively a random 5-of-3 draw from the tiny fixture
# corpus, so this test can't assert *which* articles get cited — only that
# the whole pipeline runs and the hard citation-validation in answer_node
# passed (any CitationError would fail the test by raising).
#
# Skips (not errors) when ANTHROPIC_API_KEY isn't set — matches
# lesson 06's "pytest stays green for anyone without a key" promise. The DB
# side skips itself the same way via test_engine's test_database_url
# fixture (tests/conftest.py) when no DATABASE_URL/TEST_DATABASE_URL is
# configured. To run for real: `set -a; source .env; set +a; uv run pytest
# -m integration -k graph_real` (or `uv run --env-file .env pytest -m
# integration -k graph_real`) with a real ANTHROPIC_API_KEY in `.env`.
import os

import pytest
from fake_embeddings import FakeEmbeddings
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk
from compliance_copilot.graph import AnswerSchema
from compliance_copilot.graph.build import ask
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.ingest import pipeline

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real Anthropic call",
)
def test_ask_returns_validated_answer_against_real_llm_and_db(test_engine, fixture_regulations):
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        # The full set of article anchors that exist in the tiny fixture
        # corpus — every citation must fall inside this set (a superset of
        # whatever was actually retrieved, since answer_node already
        # enforces the narrower "retrieved" constraint by raising
        # CitationError otherwise; not raising is the actual proof).
        known_anchors = set(session.scalars(select(Chunk.anchor_id).where(Chunk.kind == "article")))

        result = ask(
            "What is a high-risk AI system?",
            session=session,
            embeddings=embeddings,
            llm=make_llm(),
        )

        assert isinstance(result, AnswerSchema)
        for citation in result.citations:
            assert citation.anchor in known_anchors
