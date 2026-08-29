# tests/test_router_real_integration.py — the one real, network-hitting
# test for ADR-0023's router (src/compliance_copilot/router.py). Needs only
# an OPENAI_API_KEY (or ANTHROPIC_API_KEY, whichever `settings.llm_provider`
# points at) — no DB, mirrors tests/test_guards_classifier_real_integration.py's
# gating pattern exactly.
#
# Three unambiguous, hand-labelled questions (one per non-cross-regulation
# label plus one clearly out-of-scope question) — a smoke test on the real
# `gpt-4.1-nano`-tier model, not a statistical threshold (the larger
# ten-question fixture is exercised with a fake structured-output LLM in
# tests/graph_helpers.py's `ROUTER_FIXTURE`, at zero network cost — used by
# both tests/test_graph.py and tests/evals/test_trajectory.py).
#
# Run: `set -a; source .env; set +a; uv run pytest -m integration -q -s -k
# router_real` (the `-s` flag surfaces the per-question report below).
import os
import time

import pytest
from graph_helpers import ROUTER_FIXTURE

from compliance_copilot.router import make_router_llm, route
from compliance_copilot.settings import settings

pytestmark = pytest.mark.integration

_PROVIDER_KEY_VAR = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "ANTHROPIC_API_KEY"

QUESTIONS = [
    (
        "What obligations does a provider have when placing a high-risk AI system on the market?",
        "ai_act",
    ),
    (
        "What legal basis is required to process special category data under GDPR?",
        "gdpr",
    ),
    ("What is the best recipe for a German sauerbraten?", "out_of_scope"),
]


@pytest.mark.skipif(
    not os.environ.get(_PROVIDER_KEY_VAR),
    reason=f"{_PROVIDER_KEY_VAR} not set — skipping real router call "
    f"(llm_provider={settings.llm_provider!r})",
)
def test_router_labels_three_unambiguous_questions_correctly():
    llm = make_router_llm()
    rows = []
    for question, expected in QUESTIONS:
        started = time.monotonic()
        verdict = route(question, llm)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        rows.append((question, expected, verdict, elapsed_ms))

    print("\n--- router (real model) ---")
    for question, expected, verdict, elapsed_ms in rows:
        got = verdict.regulation if verdict is not None else "timeout"
        print(f"  want={expected:12s} got={got:12s} {elapsed_ms:5d}ms  {question[:60]!r}")

    for question, expected, verdict, _ in rows:
        assert verdict is not None, f"router outage on {question!r}"
        assert verdict.regulation == expected, (
            f"expected {expected!r}, got {verdict.regulation!r} for {question!r}"
        )


@pytest.mark.skipif(
    not os.environ.get(_PROVIDER_KEY_VAR),
    reason=f"{_PROVIDER_KEY_VAR} not set — skipping real router call "
    f"(llm_provider={settings.llm_provider!r})",
)
def test_router_labels_ten_question_fixture_with_at_least_nine_correct():
    """Lesson 21's router-accuracy trajectory eval (ADR-0026), gated the
    SAME way as the three-question smoke test above — extends this file
    rather than duplicating it, per that lesson's "pick the smaller diff"
    note. Reuses `graph_helpers.ROUTER_FIXTURE` (the same ten hand-labelled
    questions `tests/test_graph.py`'s fake-router mechanism tests and
    `tests/evals/test_trajectory.py`'s structural accuracy eval already
    exercise at zero cost) against the REAL `gpt-4.1-nano`-tier model — a
    statistical threshold (>=9/10), not a hard 10/10, since a real model's
    judgment call on the two deliberately ambiguous `"both"` questions can
    reasonably differ run to run; costs a few cents (ten short completions,
    same cheap tier as the three-question test above)."""
    llm = make_router_llm()
    rows = []
    for question, expected in ROUTER_FIXTURE:
        started = time.monotonic()
        verdict = route(question, llm)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        rows.append((question, expected, verdict, elapsed_ms))

    print("\n--- router (real model, 10-question fixture) ---")
    correct = 0
    for question, expected, verdict, elapsed_ms in rows:
        got = verdict.regulation if verdict is not None else "timeout"
        ok = got == expected
        correct += ok
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] want={expected:12s} got={got:12s} {elapsed_ms:5d}ms  {question[:60]!r}")

    assert correct >= 9, f"router labelled only {correct}/10 correctly (want >= 9)"
