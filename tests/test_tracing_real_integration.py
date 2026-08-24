# tests/test_tracing_real_integration.py — the one test in this feature that
# talks to a real Langfuse Cloud project (ADR-0009 amendment). Skipped
# unless LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and OPENAI_API_KEY are all
# set — none of which exist yet (Jay hasn't created a Langfuse Cloud account
# as of this feature's build), so this SKIPS today, locally and in CI. Same
# real-DB/real-embeddings/real-LLM setup as test_graph_real_integration.py
# (test_engine + fixture_regulations + pipeline.ingest), plus a real
# Langfuse callback in the config this time.
import os

import pytest
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph.build import ask
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.ingest import pipeline

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
            and os.environ.get("OPENAI_API_KEY")
        ),
        reason="needs a real Langfuse Cloud account + OpenAI key (LANGFUSE_PUBLIC_KEY/"
        "LANGFUSE_SECRET_KEY/OPENAI_API_KEY) — none exist yet for this project",
    ),
]


def test_ask_produces_a_real_trace_id_and_flushes(test_engine, fixture_regulations):
    """Checks the one thing assertable without polling Langfuse's async
    ingestion pipeline: a fetch-back `trace.get()` right after `flush()` can
    still 404 before ingestion catches up, so that's not a good CI
    assertion. `current_trace_id` non-empty + `flush()` raising nothing is
    the safe, low-flake bar; seeing the trace actually appear in the
    Langfuse UI is a manual check, not an automated one."""
    embeddings = get_embeddings()
    config = tracing.run_config(tags=["test"])

    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)
        ask(
            "What is an AI system?",
            session=session,
            embeddings=embeddings,
            llm=make_llm(),
            config=config,
        )

    trace_id = tracing.current_trace_id(config)
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0
    tracing.flush()  # must not raise
