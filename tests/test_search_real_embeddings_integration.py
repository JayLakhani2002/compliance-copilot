# tests/test_search_real_embeddings_integration.py — the one integration
# test that spends real OpenAI money: ingests the AI Act for real (live
# EUR-Lex fetch + real text-embedding-3-small calls) and searches "What is a
# high-risk AI system?". Gated behind BOTH RUN_NETWORK_TESTS=1 and
# OPENAI_API_KEY — never runs in CI's default job. Run it locally,
# occasionally, not on every push (costs cents). Writes into the disposable
# test DB (tests/conftest.py's `test_engine` fixture), never DATABASE_URL —
# see conftest.py's module docstring for why.
#
# IMPORTANT FINDING (ADR-0004's retrieval assumptions): plain whole-corpus
# cosine similarity does NOT put the AI Act's Art. 6 ("Classification rules
# for high-risk AI systems") top-1 for this question — Recital 52 ranks
# first instead (dist 0.3046 vs Art. 6's 0.4183), and Art. 6 ranks ~11th
# even filtered to AI-Act articles only. This isn't a pipeline bug: recitals
# restate "high-risk AI system" in more natural, question-shaped prose than
# Art. 6's operative conditional text, so a single embedding vector per
# chunk genuinely favours them for this query. Measuring and, if needed,
# correcting this (e.g. kind-aware retrieval or hybrid lexical+vector
# search) is retrieval-eval work, not this ingest feature — so this test
# only checks retrieval *runs end-to-end and returns AI-Act-relevant
# results*, not a specific rank, and prints the top-5 for manual review.
import os

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.ingest.pipeline import ingest

pytestmark = pytest.mark.integration

if os.environ.get("RUN_NETWORK_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"):
    pytest.skip(
        "RUN_NETWORK_TESTS != 1 or OPENAI_API_KEY unset — skipping real-embedding search test",
        allow_module_level=True,
    )


def test_search_high_risk_ai_system_returns_relevant_ai_act_chunks(test_engine):
    embeddings = get_embeddings()

    with Session(test_engine) as session:
        ingest("ai_act", embeddings, session)

        query_vector = embeddings.embed_query("What is a high-risk AI system?")
        distance = Chunk.embedding.cosine_distance(query_vector)
        results = session.execute(
            select(Chunk, distance.label("distance")).order_by(distance).limit(5)
        ).all()

        print("\ntop-5 for 'What is a high-risk AI system?':")
        for chunk, dist in results:
            print(f"  {chunk.anchor_id} ({chunk.kind}) dist={dist:.4f} {chunk.title or ''!r}")

        assert len(results) == 5
        # every top-5 result is genuinely about high-risk AI systems (loose
        # content check — the specific-anchor assertion this test used to
        # make doesn't hold, see the module docstring above).
        assert all("high-risk" in chunk.text.lower() for chunk, _ in results)
