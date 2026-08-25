# tests/test_logging_filter.py — unit tests for the logging backstop
# (src/compliance_copilot/logging_filter.py, ADR-0020). Not the primary PII
# control (see that module's docstring) — just proves the regex scrub
# actually fires on the three shapes it targets and leaves ordinary legal
# text alone.
from __future__ import annotations

import logging

from compliance_copilot.logging_filter import PiiScrubFilter, install_pii_scrub


def _filtered_message(msg: str, args: tuple = ()) -> str:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    PiiScrubFilter().filter(record)
    return record.getMessage()


def test_email_in_log_line_is_scrubbed():
    assert "jane@example.com" not in _filtered_message("question contained jane@example.com")
    assert "<PII>" in _filtered_message("question contained jane@example.com")


def test_iban_in_log_line_is_scrubbed():
    result = _filtered_message("account DE89370400440532013000 seen")
    assert "DE89370400440532013000" not in result
    assert "<PII>" in result


def test_phone_in_log_line_is_scrubbed():
    result = _filtered_message("caller +49 151 23456789 reported an issue")
    assert "23456789" not in result
    assert "<PII>" in result


def test_legal_text_is_untouched():
    assert _filtered_message("guard_in flagged reasons=('Article 6',)") == (
        "guard_in flagged reasons=('Article 6',)"
    )
    assert _filtered_message("citing Regulation (EU) 2016/679") == "citing Regulation (EU) 2016/679"


def test_percent_style_args_are_scrubbed():
    result = _filtered_message("email seen: %s", ("jane@example.com",))
    assert "jane@example.com" not in result
    assert "<PII>" in result


def test_install_pii_scrub_is_idempotent():
    root = logging.getLogger()
    before = len(root.filters)
    install_pii_scrub()
    install_pii_scrub()
    after = len(root.filters)
    assert after == before + 1
    root.removeFilter(next(f for f in root.filters if isinstance(f, PiiScrubFilter)))
