# tests/test_ingest_eurlex_integration.py — real fetch against the live
# Cellar endpoint for both regulations. Gated behind RUN_NETWORK_TESTS=1
# (module-level skip below) and NOT wired into CI yet: the project rule
# "pytest green before the next feature starts" means CI's default run
# must stay deterministic and
# offline — a flaky third-party endpoint failing CI on an unrelated PR is
# exactly the kind of noise that makes people start ignoring red CI.
import os

import pytest

from compliance_copilot.ingest.eurlex import REGULATIONS, ingest_regulation

pytestmark = pytest.mark.integration

if os.environ.get("RUN_NETWORK_TESTS") != "1":
    pytest.skip("RUN_NETWORK_TESTS != 1 — skipping live EUR-Lex fetch", allow_module_level=True)


@pytest.mark.parametrize(
    ("key", "expected_recitals"),
    [("ai_act", 180), ("gdpr", 173)],
)
def test_ingest_regulation_real_counts(key, expected_recitals):
    chunks = ingest_regulation(key)
    articles = [c for c in chunks if c.kind == "article"]
    recitals = [c for c in chunks if c.kind == "recital"]
    assert len(articles) == REGULATIONS[key]["expected_articles"]
    assert len(recitals) == expected_recitals
