# tests/test_api_real_integration.py — the one real end-to-end test for the
# HTTP surface (api.py, ADR-0016): a real Postgres+pgvector DB (read-only,
# against whatever DATABASE_URL points at — the already-ingested full
# corpus, same pattern as tests/test_graph_real_integration.py), real
# embeddings, and a real LLM call, driven through an actual `TestClient`
# request/response/SSE cycle rather than calling `ask()` directly.
#
# Skips (not errors) when the configured provider's key isn't set, or when
# DATABASE_URL's chunk table has fewer than 100 rows (not the full ingested
# corpus — a fresh clone or empty dev DB shouldn't fail this, just skip it).
# To run for real: `set -a; source .env; set +a; uv run pytest -m
# integration -k api_real` (costs cents: one embedding call + one LLM call).
import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from compliance_copilot.api import app, get_embeddings_dependency, get_llm_dependency
from compliance_copilot.db import Chunk, get_engine, get_session
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.settings import settings

pytestmark = pytest.mark.integration

_PROVIDER_KEY_VAR = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "ANTHROPIC_API_KEY"


def _prod_chunk_count() -> int:
    """Row count in whatever DATABASE_URL points at — gates this test so an
    empty/unreachable DB (CI, a fresh clone) skips quietly instead of
    erroring (same helper as test_graph_real_integration.py)."""
    try:
        with Session(get_engine()) as session:
            return session.scalar(select(func.count()).select_from(Chunk)) or 0
    except Exception:
        return 0


@pytest.mark.skipif(
    not os.environ.get(_PROVIDER_KEY_VAR) or _prod_chunk_count() < 100,
    reason=f"{_PROVIDER_KEY_VAR} not set, or DATABASE_URL's chunk table has fewer than 100 "
    "rows (not the full ingested corpus) — skipping the real /ask integration test",
)
def test_ask_high_risk_question_final_event_cites_art_6(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "integration-test-key-not-a-real-secret")

    def override_get_session():
        # Read-only: no init_db/reset, ever — this is whatever DATABASE_URL
        # points at, the same real corpus test_graph_real_integration.py's
        # second test reads from.
        with Session(get_engine()) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_embeddings_dependency] = get_embeddings
    app.dependency_overrides[get_llm_dependency] = make_llm
    try:
        client = TestClient(app)
        with client.stream(
            "POST",
            "/ask",
            json={"question": "When is an AI system classified as high-risk under the AI Act?"},
            headers={"X-API-Key": "integration-test-key-not-a-real-secret"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
    finally:
        app.dependency_overrides.clear()

    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        event_line, data_line = block.split("\n", 1)
        events.append((event_line.removeprefix("event: "), data_line.removeprefix("data: ")))

    final_payloads = [json.loads(data) for event, data in events if event == "final"]
    assert len(final_payloads) == 1, f"expected exactly one final event, got events={events}"
    anchors = [c["anchor"] for c in final_payloads[0]["citations"]]
    assert "art_6" in anchors
