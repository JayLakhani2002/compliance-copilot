# tests/test_guards_quotes.py — unit tests for guards/quotes.py's
# `quote_matches` (ADR-0031). No network, no DB: pure string/similarity
# logic, tested directly rather than through the full graph (tests/
# test_graph.py covers the `_validate_citations` wiring + the retry loop;
# this file covers the matching function's own decision boundary).
from __future__ import annotations

import pytest

from compliance_copilot.guards.quotes import quote_matches
from compliance_copilot.settings import settings

# A realistic, article-shaped excerpt (numbered sub-clause, an "(1)" marker,
# a comma-heavy compound sentence) — long enough that comparing a short
# quote against the WHOLE excerpt in one difflib.ratio() call would always
# score low (see quotes.py's `_best_fuzzy_window` docstring), which is
# exactly the failure mode windowing exists to avoid.
EXCERPT = (
    "Article 9(1) requires that a risk management system shall be established, "
    "implemented, documented and maintained in relation to high-risk AI systems. "
    "The risk management system shall consist of a continuous iterative process "
    "run throughout the entire lifecycle of a high-risk AI system, requiring "
    "regular systematic updating (1)."
)


def test_exact_match_takes_the_fast_path_with_no_score():
    """An unchanged, pre-ADR-0031 verbatim quote (modulo the existing
    whitespace/case normalisation) must still hit the exact substring check
    — `score=None` proves the fuzzy fallback never even ran."""
    match = quote_matches("a risk management system shall be established", EXCERPT)
    assert match.ok is True
    assert match.score is None


@pytest.mark.parametrize(
    "quote",
    [
        # punctuation swap: semicolon for comma
        "a continuous iterative process run throughout the entire lifecycle "
        "of a high-risk AI system; requiring",
        # collapsed ellipsis standing in for elided words
        "a risk management system shall be established, implemented, "
        "documented and... maintained in relation to high-risk AI systems",
        # spaced-out hyphen ("high - risk" vs. the excerpt's "high-risk")
        "a continuous iterative process run throughout the entire lifecycle "
        "of a high - risk AI system",
        # dropped trailing "(1)" marker
        "regular systematic updating.",
    ],
    ids=["punctuation_swap", "ellipsis_collapse", "hyphen_spacing", "dropped_marker"],
)
def test_cosmetic_drift_passes_only_via_fuzzy_fallback(quote):
    match = quote_matches(quote, EXCERPT)
    assert match.ok is True
    assert match.score is not None  # proves it did NOT take the exact fast path
    assert match.score >= settings.quote_similarity_min


@pytest.mark.parametrize(
    "quote",
    [
        # right words, inverted meaning (negation inserted)
        "a continuous iterative process is not required for high-risk AI "
        "systems, requiring no updating at all",
        # spliced: half from this excerpt, half paraphrased from elsewhere
        "An AI system shall be considered high-risk where it is a safety "
        "component, requiring regular systematic updating",
        # genuinely fabricated, no basis in the excerpt at all
        "providers must delete all training data within 24 hours of "
        "deployment as required by this article",
    ],
    ids=["wrong_meaning", "spliced_half_sentences", "fabricated"],
)
def test_hallucinated_quotes_stay_below_the_floor(quote):
    match = quote_matches(quote, EXCERPT)
    assert match.ok is False
    assert match.score < settings.quote_similarity_min


def test_quote_from_a_different_article_stays_below_the_floor():
    other_article = "An AI system shall be considered high-risk where it is a safety component."
    match = quote_matches(other_article, EXCERPT)
    assert match.ok is False
    assert match.score < settings.quote_similarity_min


def test_boundary_at_exact_threshold_is_inclusive(monkeypatch):
    """`ok = score >= floor` — pin the floor to a drift quote's own measured
    score, so the pass/fail boundary is exercised exactly at the threshold
    (>=), not just comfortably above/below it."""
    quote = "regular systematic updating."
    measured = quote_matches(quote, EXCERPT).score
    assert measured is not None

    monkeypatch.setattr(settings, "quote_similarity_min", measured)
    assert quote_matches(quote, EXCERPT).ok is True

    monkeypatch.setattr(settings, "quote_similarity_min", measured + 0.0001)
    assert quote_matches(quote, EXCERPT).ok is False


def test_short_excerpt_shorter_than_quote_does_not_crash():
    """`_best_fuzzy_window`'s window-range math must not raise when the
    excerpt is shorter than the quote (a too-short retrieved part) — the
    `max(len(excerpt) - window, 0)` clamp exists for exactly this. Also
    proves the round-2 subsequence check still runs on that degenerate
    single-window case, not just the ratio (reviewer BLOCKER 2's root
    cause) — "a much longer quote" adds real words ("much", "longer",
    "than", "this", "tiny") the short excerpt never had."""
    match = quote_matches("a much longer quote than this tiny excerpt has", "short excerpt")
    assert match.ok is False


def test_min_quote_length_is_a_caller_concern_not_quote_matches():
    """`quote_matches` itself has no length floor — `_MIN_QUOTE_LENGTH`
    screening happens in `_validate_citations`/`cite` BEFORE calling this
    (ADR-0014), so a trivially short quote that happens to score high here
    is still rejected upstream, not by this function."""
    match = quote_matches("is", "this is a test")
    assert match.ok is True  # "is" is a real substring — proves no length gate lives here


# --- ADR-0031 round 2: reviewer-found floor-integrity probes, now permanent
# regressions. Round 1 shipped a ratio-only floor; these three all passed
# it (0.92-0.97) despite adding words the excerpt never had — the
# order-preserving-subsequence check (`_is_ordered_subsequence` in
# quotes.py) exists specifically to catch what a character-level ratio
# structurally cannot.


def test_negation_flip_is_rejected_despite_high_ratio():
    """Reviewer round-1 probe: inserting one "not" into an otherwise-genuine,
    80+ character quote scored 0.957-0.968 (comfortably above the 0.92
    floor) under the ratio-only check — the round-2 subsequence condition
    must reject it regardless of that score."""
    quote = (
        "The risk management system shall not consist of a continuous "
        "iterative process run throughout the entire lifecycle of a "
        "high-risk AI system"
    )
    match = quote_matches(quote, EXCERPT)
    assert match.ok is False


def test_appended_fabricated_clause_is_rejected():
    """Reviewer round-1 probe: a genuine, article-length quote with a short
    fabricated clause appended (", within 90 days") scored 0.961-0.963 —
    the root cause was the too-short-excerpt window collapsing to one
    whole-string comparison that rewards the genuine majority regardless of
    what's tacked onto the end."""
    short_excerpt = (
        "Article 26(1) requires that deployers of high-risk AI systems "
        "shall take appropriate technical and organisational measures "
        "to ensure they use such systems in accordance with the "
        "instructions of use accompanying the systems."
    )
    quote = short_excerpt.rstrip(".") + ", within 90 days"
    match = quote_matches(quote, short_excerpt)
    assert match.ok is False


def test_prepended_fabricated_clause_is_rejected():
    """Reviewer round-1 probe: a short fabricated clause prepended
    ("Notwithstanding any other provision, ") to an otherwise-genuine quote
    scored 0.921 — same root cause and fix as the appended case above."""
    short_excerpt = (
        "Article 26(1) requires that deployers of high-risk AI systems "
        "shall take appropriate technical and organisational measures "
        "to ensure they use such systems in accordance with the "
        "instructions of use accompanying the systems."
    )
    quote = "Notwithstanding any other provision, " + short_excerpt
    match = quote_matches(quote, short_excerpt)
    assert match.ok is False


@pytest.mark.parametrize("pad", range(8))
def test_genuine_drift_quote_accepts_at_every_offset(pad):
    """Reviewer round-1 probe (BLOCKER 3): the same genuine drift quote
    (the `dropped_marker` fixture above), shifted 0-7 filler characters
    earlier in the excerpt — nothing about the quote's fidelity changes,
    only where it happens to sit. Round 1's step-8 window grid flipped 4 of
    these 8 offsets from accept to reject purely as a sampling artifact;
    the round-2 step-1 (full) scan must accept at every offset."""
    quote = "regular systematic updating."
    padded_excerpt = EXCERPT[:20] + (" " * pad) + EXCERPT[20:]
    match = quote_matches(quote, padded_excerpt)
    assert match.ok is True
