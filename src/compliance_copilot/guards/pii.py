# src/compliance_copilot/guards/pii.py — layer 3 of `guard_in`: detects and
# redacts PII (names, emails, phone numbers, IBANs, credit cards, IPs) in
# the user's question before it reaches retrieval, the answer LLM call, or
# a Langfuse trace (docs/decisions/ADR-0020, GDPR Art. 5(1)(c)/Art. 25).
# Runs after the injection heuristics (ADR-0018) and classifier (ADR-0019)
# have already judged the RAW text — this module never decides allow/block
# on its own, it only transforms text that was already judged safe (see
# `guard_in_node`, graph/nodes.py, for why redaction must run last).
#
# Design in one line: Presidio's `AnalyzerEngine` finds PII spans (regex/
# checksum for email/IBAN/credit-card/IP, spaCy NER for names), a fixed
# placeholder map swaps each span for a `<TYPE>` token via
# `AnonymizerEngine`, and only the entity TYPE names are ever returned to
# the caller — never the matched values (same "never echo the payload" rule
# `GuardResult.reasons` already follows, guards/injection.py).
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------------------------------------------------------------------------
# Analyzer construction — spaCy model load (~1s per language) is the
# expensive part, not the per-question analyze() call, so this is built
# once per process via lru_cache(maxsize=1) (same pattern as guards/
# classifier.py's cache, and build.py's build_graph()) instead of per
# request. `en_core_web_sm`/`de_core_news_sm` (pyproject.toml) chosen over
# the `_lg`/`_md` variants — ~26 MiB combined vs. 380MB+ — a deliberate
# trade of lower name-recall for a single-VPS deploy's image size and
# startup time (ADR-0020).
# ---------------------------------------------------------------------------
_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"},
        {"lang_code": "de", "model_name": "de_core_news_sm"},
    ],
}


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIG)
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en", "de"])


# `AnonymizerEngine` only consumes `RecognizerResult` objects (no NLP model,
# no per-call cost worth caching) — a plain module-level instance, not
# lru_cache, same "don't cache what's already cheap" reasoning as
# guards/classifier.py's `_verdict_cache` being a plain dict rather than
# something heavier.
_anonymizer = AnonymizerEngine()


# ---------------------------------------------------------------------------
# Entity set. DATE_TIME, LOCATION, NRP, URL are deliberately excluded (not
# just unconfigured) — legal text is full of dates ("Article 6 enters into
# force on..."), place names, and nationality/religious/political terms
# (NRP) that would false-positive constantly on this corpus's own subject
# matter; URL similarly flags ordinary citations. Passing `entities=` to
# `analyze()` restricts candidates to exactly this list, so those types
# never even get scored — cheaper than filtering them out afterwards, and
# it's what stops Presidio's ORGANIZATION recognizer from flagging "GDPR"/
# "EU" too (verified empirically: unrestricted `analyze()` tags "GDPR" as
# ORGANIZATION at 0.85 — restricting `entities` removes it from
# consideration entirely, no denylist needed for that case).
# ---------------------------------------------------------------------------
_ENTITIES = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD", "IP_ADDRESS")

# Score threshold: 0.4, not the commonly-suggested 0.5 (Presidio's own
# default is 0.0). Verified empirically against installed
# presidio_analyzer/predefined_recognizers/generic/phone_recognizer.py:
# `PhoneRecognizer.SCORE = 0.4` is a HARD CONSTANT — every phone match
# scores exactly 0.4 unless a context word ("phone"/"Telefon"/"cell"/...)
# sits near it in the text, which real questions rarely include ("+49 151
# 23456789" pasted bare never clears 0.5, confirmed against this project's
# own live-test question, ADR-0020). 0.4 is still comfortably below
# every other entity's achieved score here (PERSON 0.85, EMAIL/IBAN/
# CREDIT_CARD 1.0, IP 0.95), so lowering the global threshold from the
# suggested 0.5 costs nothing on their precision while fixing a real
# false-negative on phone numbers.
_SCORE_THRESHOLD = 0.4

_OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "IBAN_CODE": OperatorConfig("replace", {"new_value": "<IBAN>"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
    "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
}

# ---------------------------------------------------------------------------
# Legal-identifier false-positive guard. Presidio's NER occasionally tags a
# capitalised legal-citation token as PERSON (the class of mistake the
# brief anticipates, e.g. "Act" alone) — this project's own corpus is full
# of "Article 6", "Art. 3(1)", "Regulation (EU) 2016/679", "GDPR", "AI Act".
# Empirically (ADR-0020), restricting `entities=` above already keeps
# all of those clean on the exact fixture set tested — this denylist is
# cheap defence-in-depth for a phrasing the fixtures didn't happen to hit,
# not a fix for an observed failure.
# ---------------------------------------------------------------------------
_LEGAL_PREFIX_RE = re.compile(r"^(Article|Art\.?|Regulation|Annex|Recital|Chapter|Section)\b")
_ALL_CAPS_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")


def _is_legal_identifier(matched_text: str) -> bool:
    """Per-token, not whole-span: en_core_web_sm tags the two-token span
    "GDPR Art" as PERSON, which neither a whole-span prefix match nor a
    whole-span acronym match catches — review reproduced "<PERSON>. 22"
    from "GDPR Art. 22". Any token that is a legal keyword or an all-caps
    acronym makes the whole span a legal identifier, never a name."""
    return any(
        _LEGAL_PREFIX_RE.match(tok) or _ALL_CAPS_ACRONYM_RE.match(tok)
        for tok in matched_text.split()
    )


# ---------------------------------------------------------------------------
# Language detection — a cheap stdlib heuristic, not a language-ID model:
# this only has to pick between the two languages `get_analyzer()` actually
# has models for. Counts German stopwords + umlaut/eszett characters against
# a matching English stopword count; ties (including all-English or
# all-numeric/PII-only text with no stopwords at all, e.g. "hans@firma.de
# +49 151 23456789") default to "en" — `>` not `>=` below is what makes the
# default win ties.
# ---------------------------------------------------------------------------
_DE_STOPWORDS = frozenset("der die das und ist nicht ein eine für mit".split())
_EN_STOPWORDS = frozenset("the and is not a an for with of to".split())
_DE_CHARS = frozenset("äöüßÄÖÜ")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    words = _WORD_RE.findall(text.lower())
    de_score = sum(1 for w in words if w in _DE_STOPWORDS) + sum(
        1 for ch in text if ch in _DE_CHARS
    )
    en_score = sum(1 for w in words if w in _EN_STOPWORDS)
    return "de" if de_score > en_score else "en"


# ---------------------------------------------------------------------------
# Result type — mirrors guards/injection.py's `GuardResult`: `entities`
# holds TYPE names only ("EMAIL_ADDRESS", "PERSON", ...), never the matched
# values, so this object is always safe to log or put in an SSE event.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RedactionResult:
    text: str
    entities: tuple[str, ...]


_DE_DETERMINERS = frozenset(
    "mein meine unser unsere der die das ein eine ihr ihre sein seine dieser diese".split()
)


def _looks_like_name(span: str) -> bool:
    """Name-shaped: 2–3 tokens, every token capitalised, not starting with a
    German determiner ("Mein Mandant"). ponytail: a shape heuristic, not
    NER — good enough to keep the en model's German false positives out
    while letting "Hans Müller" / "Anna Maria Schmidt" through."""
    tokens = span.split()
    if not 2 <= len(tokens) <= 3 or tokens[0].casefold() in _DE_DETERMINERS:
        return False
    return all(t[:1].isupper() for t in tokens)


def redact(text: str, language: str | None = None) -> RedactionResult:
    """Detects and redacts PII in `text`. `language` overrides
    `detect_language()`'s guess — `guard_in_node` always lets this default,
    the parameter exists for tests/callers that already know the language.
    Returns `text` unchanged (with `entities=()`) when nothing is found —
    the common case, so this never allocates an `AnonymizerEngine` call for
    clean text."""
    lang = language or detect_language(text)
    analyzer = get_analyzer()
    results = analyzer.analyze(
        text=text, language=lang, entities=list(_ENTITIES), score_threshold=_SCORE_THRESHOLD
    )
    if lang == "de":
        # de_core_news_sm's NER missed "Hans Müller" outright in a plain
        # German sentence while en_core_web_sm caught it (measured, see
        # ADR-0020) — so PERSON recall on German text is the union of both
        # small models. Cost: the en model also tags phrases like "Mein
        # Mandant" as PERSON → over-redaction, which is the safe direction
        # (the question stays answerable; a leaked name is not reversible).
        # ...but unfiltered, the en model also tagged "zur Bewertung von
        # Bewerbern" as PERSON, which would delete the question's meaning —
        # so only spans shaped like a name are accepted (_looks_like_name).
        # The anonymizer resolves overlapping spans itself.
        results += [
            r
            for r in analyzer.analyze(
                text=text, language="en", entities=["PERSON"], score_threshold=_SCORE_THRESHOLD
            )
            if _looks_like_name(text[r.start : r.end])
        ]
    results = [r for r in results if not _is_legal_identifier(text[r.start : r.end])]
    if not results:
        return RedactionResult(text=text, entities=())

    outcome = _anonymizer.anonymize(text=text, analyzer_results=results, operators=_OPERATORS)
    entity_types = tuple(sorted({r.entity_type for r in results}))
    return RedactionResult(text=outcome.text, entities=entity_types)
