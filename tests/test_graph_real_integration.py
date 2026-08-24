# tests/test_graph_real_integration.py — the one real end-to-end test for
# the graph (ADR-0014): a real Postgres+pgvector DB (test_engine, ADR-0003),
# real embeddings (`get_embeddings()`, ADR-0004), and a real LLM call
# (`make_llm()`, ADR-0002 + its 2026-08-24 amendment — OpenAI is the
# configured provider until an Anthropic key exists). Real embeddings mean
# retrieval on the tiny fixture corpus is real too, not a random draw — this
# test can assert the answer/citations are actually meaningful, not just
# that the pipeline ran without raising.
#
# Skips (not errors) when the *configured* provider's key isn't set — reads
# `settings.llm_provider` so this test tracks whichever branch `make_llm()`
# will actually take, instead of hardcoding one provider's env var. The DB
# side skips itself the same way via test_engine's test_database_url
# fixture (tests/conftest.py) when no DATABASE_URL/TEST_DATABASE_URL is
# configured. To run for real: `set -a; source .env; set +a; uv run pytest
# -m integration -k graph_real` with a real key for the configured provider
# in `.env` (OPENAI_API_KEY by default — costs cents in embeddings + one
# gpt-4.1-mini call).
import os

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import AnswerSchema
from compliance_copilot.graph.build import ask
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.ingest import pipeline
from compliance_copilot.settings import settings

pytestmark = pytest.mark.integration

_PROVIDER_KEY_VAR = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "ANTHROPIC_API_KEY"


@pytest.mark.skipif(
    not os.environ.get(_PROVIDER_KEY_VAR),
    reason=f"{_PROVIDER_KEY_VAR} not set — skipping real LLM call "
    f"(llm_provider={settings.llm_provider!r})",
)
def test_ask_returns_validated_answer_against_real_llm_and_db(test_engine, fixture_regulations):
    embeddings = get_embeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)

        # The full set of article anchors that exist in the tiny fixture
        # corpus (art_1, art_2, art_3 — no art_6; see
        # tests/fixtures/eurlex_sample.xhtml) — every citation must fall
        # inside this set (a superset of whatever was actually retrieved,
        # since answer_node already enforces the narrower "retrieved"
        # constraint by raising CitationError otherwise; not raising is the
        # actual proof).
        known_anchors = set(session.scalars(select(Chunk.anchor_id).where(Chunk.kind == "article")))

        # NOT "What is a high-risk AI system?" (the question used elsewhere,
        # e.g. tests/test_search_real_embeddings_integration.py): tried
        # first against this real LLM + real embeddings + this fixture, and
        # it correctly returned zero citations, because none of Art. 1-3
        # actually *define* "high-risk AI system" — that's Art. 6, which
        # this trimmed 3-article fixture doesn't include (this ADR-0013's
        # own docstring already flags Art. 6 as hard to retrieve for this
        # exact question even in the full corpus). That's the system
        # prompt's "say so and cite nothing" behaviour working as intended,
        # not a bug — but it can't be asserted as "returns >=1 citation".
        # "What is an AI system?" is verbatim-defined in Art. 3(1), so it
        # exercises the same real retrieve->cite->validate pipeline with an
        # answer this fixture can actually support.
        result = ask(
            "What is an AI system?",
            session=session,
            embeddings=embeddings,
            llm=make_llm(),
        )

        assert isinstance(result, AnswerSchema)
        assert result.answer.strip()
        assert len(result.citations) >= 1
        assert result.citations[0].anchor == "art_3"
        for citation in result.citations:
            assert citation.anchor in known_anchors
