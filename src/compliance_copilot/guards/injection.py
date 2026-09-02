# src/compliance_copilot/guards/injection.py — the heuristic prompt-injection
# detector: layer 1 of `guard_in` (docs/ARCHITECTURE.md §4, ADR-0018). Pure
# stdlib (`re`, `unicodedata`, `dataclasses`) — no LLM call, no dependency —
# so it costs microseconds and runs before anything else in the graph
# (docs/THREAT_MODEL.md). Day 12 adds a cheap-LLM classifier as layer 2 for
# the paraphrased attacks this can't catch; this module is unchanged by that.
#
# Design in one line: normalise the text so an attacker can't dodge a
# keyword check with invisible characters or look-alike letters, then score
# it against six weighted attack categories, each capped at one ceiling so
# repeating the same trick five times can't out-score two independent
# signals. `reasons` (the flagged categories) is the only thing ever logged
# or returned alongside the question — never the matched text itself, so a
# log line can't become a second copy of the attack payload.
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Normalisation — runs before every regex check. Order matters (each step
# depends on the previous one having already run):
#   1. NFKC folds compatibility variants (fullwidth/ligature forms) into
#      their plain form.
#   2. Strip zero-width/invisible code points an attacker can hide inside a
#      keyword (e.g. "i<ZWNJ>g<ZWNJ>nore") to dodge a literal match.
#   3. Map a small set of Unicode "confusable" look-alike letters (Cyrillic/
#      Greek letters that render identically to Latin ones, unicode.org's
#      TR39) back to plain ASCII, so "Ignоre" (Cyrillic о) reads as "Ignore".
#   4. Collapse whitespace runs to one space.
#   5. Casefold, so every pattern below can be written lowercase and match
#      any input case with no `re.IGNORECASE` flag needed.
# ---------------------------------------------------------------------------
_ZERO_WIDTH = "​‌‍⁠﻿"  # ZWSP, ZWNJ, ZWJ, word joiner, BOM/ZWNBSP
_ZERO_WIDTH_STRIP = str.maketrans(dict.fromkeys(_ZERO_WIDTH))

# Kept deliberately small (project rule: teach-sized, not a full confusables
# table) — just the handful that render as ordinary Latin letters and are
# cheap for an attacker to type. Cyrillic and Greek only; add more only when
# a real attack needs it.
_CONFUSABLES = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",  # Cyrillic
        "ο": "o",
        "ρ": "p",  # Greek
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """The five-step pipeline above, in order. Every category regex below
    is written to run against this function's output, never raw input —
    fixtures #13/#14 (zero-width chars, a Cyrillic look-alike) only flag
    because this runs first."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH_STRIP)
    text = text.translate(_CONFUSABLES)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.casefold()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardResult:
    """`reasons` holds category names only (e.g. "instruction_override") —
    never the matched substring — so this object is always safe to log or
    put in an SSE event without risking an echo of attacker-controlled text
    (same rule `CitationError` already follows, graph/state.py)."""

    flagged: bool
    score: float
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Category regexes. `_spaced(word)` builds a pattern where the tolerant
# separator class `[\s\-_.]*` sits between EVERY letter of `word`, not just
# between words — this both matches the word normally (zero separators
# between contiguous letters) and defeats "i.g.n.o.r.e"-style spaced-letter
# obfuscation (fixture #17) that survives `normalise()` untouched (dots
# aren't whitespace, so step 4's collapse doesn't help there). Used only for
# the two highest-value categories (instruction_override, exfiltration) —
# role hijack/delimiter/payload markers are short, distinctive tokens
# ("DAN", "###", "PWNED") an attacker gains little from spacing out, so
# those stay as plain, more-readable literals.
# ---------------------------------------------------------------------------
def _spaced(word: str) -> str:
    return r"[\s\-_.]*".join(re.escape(ch) for ch in word)


def _mod_group(*words: str) -> str:
    """One of several optional modifier words (e.g. "all"/"previous"),
    each letter-spaced, as a non-capturing alternation."""
    return "(?:" + "|".join(_spaced(w) for w in words) + ")"


_INSTRUCTIONS = _spaced("instructions")
_ANWEISUNGEN = _spaced("anweisungen")
_EN_MOD = _mod_group("all", "previous", "prior", "above")
_DE_MOD = _mod_group("alle", "vorherigen")

# ponytail: every category below caps at ONE ceiling weight (see `_CEILING`
# in `detect()`) regardless of how many of its regexes matched — uniform
# across categories, not the researcher's sketched 1.0/0.7/0.6/0.5 tiers.
# Why: the 20-attack fixture set has at least one attack that relies on
# EXACTLY one category alone (role hijack #4 "DAN"/"developer mode",
# delimiter #9-11 "### system"/"[INST]"/"<|im_start|>", encoding #16's
# base64 blob) — any category weighted below the default threshold would
# silently under-detect that whole fixture. This was found by running the
# fixtures, not guessed; upgrade path is per-category weights again once
# Day 12's classifier gives a second signal to combine with, so a lone
# heuristic category no longer has to carry full weight by itself.
_PATTERNS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "instruction_override": [
        (
            re.compile(rf"{_spaced('ignore')}[\s\-_.]*(?:{_EN_MOD}[\s\-_.]*)*{_INSTRUCTIONS}"),
            1.0,
        ),
        # Round-1 review finding: the modifier group was entirely optional,
        # so this matched on the bare word "disregard" ANYWHERE — real legal
        # phrasing ("can an authority disregard a minor breach") flagged.
        # Requires a real target now: an override-shaped word directly, or
        # up to 2 filler words then one (same bound as the exfiltration
        # fix below, cheap enough not to matter, wide enough for "the
        # above"/"a minor technical breach" either way — the fixture set is
        # what proves which side of that gap each phrasing falls on).
        (
            re.compile(
                rf"{_spaced('disregard')}(?:[\s\-_.]+\w+){{0,2}}[\s\-_.]*"
                rf"(?:{_EN_MOD}|{_spaced('everything')}|{_spaced('this')}"
                rf"|{_spaced('rules')}|{_INSTRUCTIONS})"
            ),
            1.0,
        ),
        (
            re.compile(
                rf"{_spaced('forget')}[\s\-_.]*(?:{_mod_group('your', 'all')})?"
                rf"[\s\-_.]*{_spaced('rules')}"
            ),
            1.0,
        ),
        (
            re.compile(
                rf"{_spaced('do')}[\s\-_.]*{_spaced('not')}[\s\-_.]*"
                rf"(?:{_spaced('follow')}|{_spaced('obey')})"
            ),
            1.0,
        ),
        # German: "ignorier(e|en)? (alle|vorherigen)* anweisungen".
        (
            re.compile(
                rf"{_spaced('ignorier')}[\s\-_.]*(?:{_spaced('e')}|{_spaced('en')})?"
                rf"[\s\-_.]*(?:{_DE_MOD}[\s\-_.]*)*{_ANWEISUNGEN}"
            ),
            1.0,
        ),
        # German: "vergiss (deine|eure)? regeln" — "forget your rules".
        (
            re.compile(
                rf"{_spaced('vergiss')}[\s\-_.]*(?:{_mod_group('deine', 'eure')})?"
                rf"[\s\-_.]*{_spaced('regeln')}"
            ),
            1.0,
        ),
    ],
    "role_hijack": [
        (re.compile(r"you[\s\-_.]*are[\s\-_.]*now"), 1.0),
        # Narrower than a bare "act as" — real legal questions legitimately
        # ask "can I act as a controller and a processor" (benign fixture
        # #7); only the jailbreak-shaped phrasing counts here.
        (
            re.compile(
                r"act[\s\-_.]*as[\s\-_.]*(?:an?[\s\-_.]*)?(?:unrestricted|uncensored|jailbroken)"
            ),
            1.0,
        ),
        (re.compile(r"\bdan\b"), 1.0),
        (re.compile(r"developer[\s\-_.]*mode"), 1.0),
    ],
    "exfiltration": [
        # Round-1 review finding: the verb and target had to sit immediately
        # next to each other — ordinary filler ("print OUT THE ENTIRE
        # system prompt") slipped through untouched. `(?:\w+\s+){0,4}`
        # tolerates up to 4 filler words between the verb and the optional
        # your/the modifier, same bounded-repeat discipline as every other
        # pattern here (no unbounded `.*`, no backtracking blow-up — ReDoS
        # timing re-checked after this change, see ADR-0018).
        (
            re.compile(
                rf"(?:{_spaced('print')}|{_spaced('reveal')}|{_spaced('repeat')}|{_spaced('show')})"
                rf"(?:\s+\w+){{0,4}}[\s\-_.]*(?:{_mod_group('your', 'the')})?"
                rf"[\s\-_.]*(?:{_spaced('system')}[\s\-_.]*)?"
                # "instructed" (verb form, "you were instructed to...") added
                # alongside "instructions"/"prompt" — the filler-gap fix
                # alone doesn't bridge a different word form, only spacing.
                rf"(?:{_spaced('prompt')}|{_INSTRUCTIONS}|{_spaced('instructed')})"
            ),
            1.0,
        ),
        (
            re.compile(
                rf"{_spaced('what')}[\s\-_.]*{_spaced('are')}[\s\-_.]*"
                rf"{_spaced('your')}[\s\-_.]*{_INSTRUCTIONS}"
            ),
            1.0,
        ),
    ],
    "delimiter": [
        (re.compile(r"</?excerpt"), 1.0),
        (re.compile(r"</?question>"), 1.0),
        (re.compile(r"###\s*system"), 1.0),
        (re.compile(r"\[/?inst\]"), 1.0),
        (re.compile(r"<\|im_start\|>"), 1.0),
    ],
    "payload_marker": [
        (re.compile(r"\bpwned\b"), 1.0),
        (re.compile(r"say[\s\-_.]*only\b"), 1.0),
    ],
    # No "encoding_obfuscation" entry here — its base64-shape check needs
    # case preserved (see `_BASE64_RE` below) and every pattern in this dict
    # runs against `normalise()`'s casefolded output, so it's checked
    # separately in `detect()`, on raw text, alongside the zero-width-count
    # and mixed-script signals that need the same thing.
}

# Shape-only signal ("looks base64-encoded") — can't confirm intent from a
# regex alone; Day 12's classifier is what actually decodes and judges a
# suspicious blob (docs/THREAT_MODEL.md).
#
# Round-1 review finding: the bare 40+-char alnum run false-positived on
# long German compound words (this app's own domain —
# "Konformitaetsbewertungsverfahrensdurchfuehrungsanforderungen") and pasted
# tokens/IDs. Real base64 output over 40+ chars is near-certain to contain a
# digit; natural-language words never do — the `(?=...[0-9])` lookahead
# requires at least one, zero-cost since it only re-scans the same bounded
# span the main match already covers (no separate backtracking pass).
#
# The reviewer's digit-only fix was re-verified here against their own 5
# listed false positives and one (a 64-char lowercase hex audit ID — digits,
# but no uppercase) still matched: hex-only tokens are a distinct
# false-positive shape a bare digit check doesn't rule out. Base64's
# alphabet is upper+lower+digit+`+`/`/`; a 40+ char span with no uppercase
# at all is far more likely to be hex/a natural-language-adjacent id than
# real base64 output, so a second lookahead requires at least one uppercase
# letter too — re-verified clean against all 5 of the reviewer's false
# positives while fixture #16's real base64 payload still matches (it has
# plenty of uppercase).
#
# MUST run on case-preserved text: `normalise()` casefolds before every
# `_PATTERNS` regex runs, which would erase the uppercase this check itself
# requires — same reasoning as the zero-width/mixed-script signals below.
_BASE64_RE = re.compile(
    r"\b(?=[A-Za-z0-9+/]*[0-9])(?=[A-Za-z0-9+/]*[A-Z])[A-Za-z0-9+/]{40,}={0,2}\b"
)

# >3 zero-width characters in the raw (pre-normalise) text is itself a
# signal, independent of what they were hiding — a legitimate question has
# no reason to contain any.
_ZERO_WIDTH_FLAG_THRESHOLD = 3
_SCRIPT_PREFIXES = ("LATIN", "CYRILLIC", "GREEK")


def _count_zero_width(text: str) -> int:
    return sum(text.count(ch) for ch in _ZERO_WIDTH)


def _text_scripts(text: str) -> set[str]:
    scripts: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for prefix in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                scripts.add(prefix)
                break
    return scripts


def _has_mixed_script_word(text: str) -> bool:
    """True if the text mixes Latin with Cyrillic/Greek letters ANYWHERE —
    not just co-located in one `\\w+` token. Round-1 review finding: a
    per-token check is defeated by isolating every leftover Latin letter
    with a space ("і g  n о r е а l  l  р r еѵіо u ѕ..." — every
    substitutable letter swapped to its Cyrillic look-alike, every
    remaining bare Latin letter its own single-char "word") — no token
    ever mixes scripts even though the whole string plainly does. No
    legitimate EN/DE legal question mixes scripts at all, so widening the
    check from "one token" to "the whole text" costs no false-positive
    budget (verified against all 40 shipped fixtures + the reviewer's
    novel benign set). Must still run on the RAW text: `normalise()`'s
    confusables step maps Cyrillic look-alikes back to Latin, which is
    correct for the regex categories but would erase this signal if run
    afterwards — so `detect()` calls this before normalising."""
    return len(_text_scripts(text)) > 1


def detect(text: str, threshold: float = 1.0) -> GuardResult:
    """Scores `text` against the six categories above. Score = sum of each
    matched category's ceiling weight (one hit or five hits in the same
    category still contributes that category's weight once — see the
    `ponytail:` note above `_PATTERNS`); `flagged` is `score >= threshold`.

    Three `encoding_obfuscation` signals (zero-width count, mixed-script
    text, base64 shape) are computed on the ORIGINAL text before
    `normalise()` runs, because normalisation is what strips/casefolds away
    the very evidence each one looks for (zero-width chars, script identity,
    base64's required uppercase); every other regex category then runs
    against the normalised text."""
    category_scores: dict[str, float] = {}

    def _flag(category: str, weight: float) -> None:
        category_scores[category] = max(category_scores.get(category, 0.0), weight)

    if _count_zero_width(text) > _ZERO_WIDTH_FLAG_THRESHOLD:
        _flag("encoding_obfuscation", 1.0)
    if _has_mixed_script_word(text):
        _flag("encoding_obfuscation", 1.0)
    if _BASE64_RE.search(text):
        _flag("encoding_obfuscation", 1.0)

    normalised = normalise(text)
    for category, patterns in _PATTERNS.items():
        for pattern, weight in patterns:
            if pattern.search(normalised):
                _flag(category, weight)
                break  # ceiling: this category is already at its max

    score = sum(category_scores.values())
    reasons = tuple(sorted(category_scores))
    return GuardResult(flagged=score >= threshold, score=score, reasons=reasons)
