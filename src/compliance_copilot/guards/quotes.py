# src/compliance_copilot/guards/quotes.py — verbatim-quote normalisation and
# fuzzy fallback, shared by the graph's own citation guard (graph/nodes.py's
# `_validate_citations`, ADR-0014) and the MCP server's `cite` tool
# (mcp_server.py, ADR-0007). Moved out of graph/nodes.py (Day-16 review nit,
# ADR-0007's Day-17 amendment): `mcp_server.py` used to import
# `_normalise`/`_MIN_QUOTE_LENGTH` straight from `graph.nodes`, which drags
# `langchain_anthropic`/`langchain_openai` into the MCP server process at
# import time (a working-but-wasteful side effect, not a bug) — this module
# has no LLM-client imports at all, so either caller can import it cheaply.
#
# ADR-0031: exact-substring matching alone (`_normalise` below, unchanged)
# rejects ~6/20 benign legal questions and golden item c01 — gpt-4.1-mini
# reproduces a genuine quote with cosmetic drift (punctuation swap, a
# collapsed "...", reordered whitespace, a dropped "(1)") that the substring
# check can't tolerate. `quote_matches()` keeps that exact check as the fast
# path (unchanged behaviour on a clean match) and only falls back to a
# stdlib `difflib` similarity score, against a HIGH floor
# (`settings.quote_similarity_min`), when the exact check misses — a
# hallucinated or spliced quote must still fail (see ADR-0031's adversarial
# fixture search for why the floor sits where it does).
from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass

from compliance_copilot.settings import settings

# The corpus's own text uses Unicode curly quotes (U+2018/U+2019, U+201C/
# U+201D) around every defined term (chunker.py's definition-boundary regex
# matches '‘' for the same reason) — the model may reproduce a quote with
# straight ASCII quotes instead. Map both to straight so that difference
# alone never causes a false-positive citation rejection (reviewer round 1).
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

# Minimum normalised-quote length: an empty or 1-2 character quote is a
# substring of almost any text, so it would pass the verbatim check with
# zero actual information content (reviewer round 1) — reject it outright
# rather than let it through as a "cited" claim.
_MIN_QUOTE_LENGTH = 20


def _normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded, curly-quotes-to-straight —
    applied to both a model's quote and the source chunk text before the
    substring check, so a quote that only differs by spacing, letter case,
    or quote-mark style still counts as verbatim (ADR-0014).

    `html.unescape` runs first: `graph/nodes.py`'s `_render_chunk`
    HTML-escapes chunk text before showing it to the model (`&`/`<`/`>` ->
    `&amp;`/`&lt;`/`&gt;`), and the system prompt tells the model to copy
    its citation quote "verbatim... from the cited excerpt's text" — i.e.
    from what it was shown, escaped form included. Unescaping both sides
    here before comparing against the raw DB text is what makes a
    genuinely-verbatim quote containing one of those characters still pass
    (round-1 review finding: previously only the escape direction was
    implemented, not the reverse, so this was a live false-positive-
    `CitationError` bug for any chunk containing `&`, `<`, or `>`)."""
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip().casefold().translate(_QUOTE_MAP)


@dataclass(frozen=True)
class QuoteMatch:
    """`ok`: whether `quote` counts as present in the excerpt, either way.
    `score`: `None` on the exact-substring fast path (nothing to log — this
    is the unchanged, pre-ADR-0031 behaviour); the best difflib similarity
    ratio found when the fast path missed and the fuzzy fallback ran,
    whether or not it cleared the floor. Callers use `score is not None` to
    know "this citation only passed because of the fuzzy fallback" — the
    signal `_validate_citations` logs as the `quote_fuzzy_match` guardrail
    event (score only, never the quote text)."""

    ok: bool
    score: float | None


# ADR-0031 round 2 (reviewer BLOCKER 3): this used to slide the window in
# steps of 8, an unmeasured "performance guard" that could skip the ONE
# offset a genuine short quote actually sits at, silently flipping a real
# accept to a reject depending on nothing but where the quote's text
# happens to fall in the excerpt. Measured cost of scanning every offset
# instead (step=1), a ~300-char quote (the system prompt's stated max):
# well under 1ms for a genuine match (the `quick_ratio()` prefilter below
# prunes almost everything once a good match is found) at both 5KB and
# 10KB excerpts; up to ~400-900ms worst case for a fully-fabricated quote
# sharing almost no vocabulary with the excerpt (the adversarial case that
# most defeats that same prefilter) — only paid on a citation that was
# going to be REJECTED anyway, and still a fraction of `request_timeout_s`
# (ADR-0028). See ADR-0031 for the full measurement. Nowhere near a real
# cost at this project's actual scale (article-sized excerpts, 1-5
# citations), so there is no step to tune: scan every offset.
def _best_fuzzy_window(quote: str, excerpt: str) -> tuple[float, str]:
    """Slides a quote-length window across every offset of `excerpt` and
    returns `(best_ratio, best_window_text)` — both inputs already
    `_normalise`d by the caller. Comparing `quote` against the WHOLE excerpt
    in one `ratio()` call would always score low no matter how good the
    match is: `SequenceMatcher` scores by `2 * matches / (len(a) + len(b))`,
    so a ~200-character quote inside a several-thousand-character excerpt is
    penalised by the excerpt's own length before similarity even enters the
    picture. Windowing at quote-length keeps the comparison apples-to-apples.
    The winning window text is returned too (not just its score) — the
    order-preserving-subsequence check in `quote_matches` below runs against
    this SAME window, not the whole excerpt, for the same apples-to-apples
    reason.

    One `SequenceMatcher` is built once with `quote` fixed as sequence `a`
    and reused via `set_seq2` for every window — `difflib`'s own documented
    pattern for scanning many candidates against one fixed sequence, since
    it lets the matcher skip re-running `a`'s "junk" autodetection each
    time. `quick_ratio()` is a cheap upper bound on `ratio()` (difflib's own
    documented shortcut) — only pay for the real ratio when a window could
    plausibly beat the current best; this only skips computation, it never
    changes which window ends up winning.

    `range(0, max(len(excerpt) - window, 0) + 1)` always runs at least once
    even when `excerpt` is shorter than `quote` (`stop` would otherwise go
    negative and the range would be empty) — that one comparison is the
    whole excerpt against the whole quote. Round 1 left that degenerate
    single-window comparison's high `ratio()` as the ONLY check (reviewer
    BLOCKER 2: a genuine quote with a fabricated clause appended/prepended
    scored 0.92-0.96 there, since `ratio()` alone rewards the genuine
    majority and is blind to what a small addition says) — the window is
    still computed the same way here, but it no longer bypasses the
    subsequence check: `quote_matches` runs that check against exactly this
    returned window, genuine excerpt-length collapse included."""
    if not quote or not excerpt:
        return 0.0, ""
    window_len = len(quote)
    matcher = difflib.SequenceMatcher(None, quote)
    best_ratio = 0.0
    best_window = ""
    for start in range(0, max(len(excerpt) - window_len, 0) + 1):
        candidate = excerpt[start : start + window_len]
        matcher.set_seq2(candidate)
        if matcher.quick_ratio() >= best_ratio:
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = candidate
    return best_ratio, best_window


# Content tokens (word/number runs) for the order-preserving subsequence
# check below — deliberately punctuation-blind (a dropped "(1)" marker or a
# comma-vs-semicolon swap must never change the token sequence), but every
# WORD in the quote still has to actually be there, in order.
_CONTENT_TOKEN_RE = re.compile(r"\w+")


def _is_ordered_subsequence(quote_tokens: list[str], window_tokens: list[str]) -> bool:
    """True iff every token in `quote_tokens` appears, in the same relative
    order, somewhere in `window_tokens` — not necessarily contiguous (a
    dropped word/clause between two matched tokens is free, which is what
    keeps the drift fixtures passing), but never out of order and never
    invented (ADR-0031 round 2, reviewer BLOCKERS 1+2). `x in it` against a
    SHARED iterator is the standard stdlib subsequence idiom: each `in`
    only scans the iterator forward from wherever the previous check left
    it, so it can never "find" a token by looking backward — a quote token
    that isn't in `window_tokens` at all, or that would only be found
    behind where the iterator has already advanced, fails the whole check
    via `all`'s short-circuit.

    This is what a `difflib` ratio alone cannot see: inserting a single
    "not" into an otherwise-genuine quote, or appending/prepending a
    fabricated clause, barely moves a character-level similarity score
    (round 1 measured 0.92-0.97 on exactly these probes) but ALWAYS adds at
    least one token the window's own text never had — this check catches
    that regardless of what the ratio said. Applied to the SAME best window
    `_best_fuzzy_window` found (not the whole excerpt) — the two checks
    must agree on what "the match" is."""
    it = iter(window_tokens)
    return all(tok in it for tok in quote_tokens)


def quote_matches(quote: str, excerpt: str) -> QuoteMatch:
    """The ONE place both `_validate_citations` (graph/nodes.py) and the MCP
    `cite` tool (mcp_server.py) decide whether a citation's `quote` counts as
    present in a retrieved `excerpt` — kept as a single function specifically
    so the two callers can't drift apart on what "verbatim enough" means
    (ADR-0031).

    Exact path (unchanged from pre-ADR-0031 behaviour): `_normalise`d `quote`
    is a substring of `_normalise`d `excerpt` -> match, `score=None`, no
    fuzzy work done at all. Fuzzy fallback only runs on an exact miss, and
    BOTH of the following must hold (ADR-0031 round 2 — round 1 only had the
    first, which reviewer round 1 broke with two working exploits):
    1. the best windowed `difflib` similarity (`_best_fuzzy_window`) is
       `>= settings.quote_similarity_min` — a high floor by design, tuned
       against measured drift/adversarial fixtures (ADR-0031);
    2. every content token in the quote appears, in order, somewhere in
       that SAME winning window (`_is_ordered_subsequence`) — omissions are
       free (that's the cosmetic-drift tolerance this whole function
       exists for), but an ADDED word — a negation, a fabricated clause, a
       spliced-in fact — is not, no matter how high the ratio scores.

    Minimum-length screening (`_MIN_QUOTE_LENGTH`) is deliberately NOT
    repeated here — both callers already enforce it on the raw `quote`
    before calling this, and re-checking a normalised length here would be
    the same check against a slightly different string for no benefit."""
    normalised_quote = _normalise(quote)
    normalised_excerpt = _normalise(excerpt)
    if normalised_quote in normalised_excerpt:
        return QuoteMatch(ok=True, score=None)
    score, window = _best_fuzzy_window(normalised_quote, normalised_excerpt)
    ok = score >= settings.quote_similarity_min and _is_ordered_subsequence(
        _CONTENT_TOKEN_RE.findall(normalised_quote), _CONTENT_TOKEN_RE.findall(window)
    )
    return QuoteMatch(ok=ok, score=score)
