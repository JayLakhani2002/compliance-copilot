# src/compliance_copilot/logging_filter.py — a cheap regex backstop on the
# root logger (docs/decisions/ADR-0020). NOT the primary PII control: that's
# "never log the raw question" (an existing, unconditional rule every
# logger.info/exception call in this codebase already follows) plus
# `guard_in`'s Presidio redaction (guards/pii.py), which runs BEFORE the
# question reaches any node that might log something derived from it. This
# filter exists only in case a future line of code accidentally slips
# PII-shaped text into a log call — cheap insurance, not a second detection
# engine. Running the full Presidio analyzer (spaCy inference) on every log
# line would be real per-line latency spent on a control that shouldn't
# fire in the steady state; three compiled regexes cost nothing.
from __future__ import annotations

import logging
import re

# Deliberately not Presidio's own recognizers — those need the analyzer's
# spaCy models loaded (guards/pii.py's `get_analyzer()`), which is exactly
# the cost this module exists to avoid paying on every log call. A regex
# catches the same three easily-shaped entity types Presidio's own
# regex/checksum recognizers target (EMAIL_ADDRESS, IBAN_CODE,
# PHONE_NUMBER) — it intentionally does NOT try to catch names (PERSON),
# which needs NER, not a pattern.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# IBAN shape: 2 letters + 2 digits + 11-30 alphanumeric — matches
# guards/pii.py's researcher-verified IBAN_CODE pattern shape (no checksum
# validation here, unlike Presidio's real recognizer — this is a cheap
# backstop, not a second validator).
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# E.164 / DE-shaped phone number: optional +, then 8-12 digits with
# optional separators (space/dot/dash/parens) between them.
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

_SCRUB_PATTERNS = (_EMAIL_RE, _IBAN_RE, _PHONE_RE)


def _scrub(value: str) -> str:
    for pattern in _SCRUB_PATTERNS:
        value = pattern.sub("<PII>", value)
    return value


class PiiScrubFilter(logging.Filter):
    """Rewrites `record.msg` and any string `record.args` in place before a
    record is emitted. A `logging.Filter` returning `True` always lets the
    record through (this never drops a log line, only edits it) — see
    `logging.Filter.filter`'s documented contract."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _scrub(record.msg)
        if isinstance(record.args, dict):
            record.args = {
                k: _scrub(v) if isinstance(v, str) else v for k, v in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(_scrub(a) if isinstance(a, str) else a for a in record.args)
        return True


def install_pii_scrub() -> None:
    """Idempotent: adds the filter to the root logger only if it isn't
    already there (api.py's lifespan and cli.py's `main()` both call this —
    without the guard, re-importing/re-running either in the same process,
    e.g. under pytest, would stack duplicate filters)."""
    root = logging.getLogger()
    if not any(isinstance(f, PiiScrubFilter) for f in root.filters):
        root.addFilter(PiiScrubFilter())
