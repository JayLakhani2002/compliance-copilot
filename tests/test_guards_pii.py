# tests/test_guards_pii.py — unit tests for ADR-0020's PII redaction
# (src/compliance_copilot/guards/pii.py) and its wiring into `guard_in_node`
# (graph/nodes.py). No network: Presidio runs fully offline once its spaCy
# models are installed (pyproject.toml's pinned wheel URLs) — the only cost
# is the ~1-2s one-time model load, shared across every test in this file
# via `get_analyzer()`'s `lru_cache(maxsize=1)`.
from __future__ import annotations

import logging

import pytest

from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.state import AnswerSchema, GraphContext
from compliance_copilot.guards.pii import detect_language, redact
from compliance_copilot.settings import settings

# --- redact(): individual entity types ------------------------------------


def test_email_is_redacted():
    result = redact("Please contact me at jane.doe@example.com about this.")
    assert "<EMAIL>" in result.text
    assert "jane.doe@example.com" not in result.text
    assert result.entities == ("EMAIL_ADDRESS",)


def test_german_phone_is_redacted():
    result = redact("Meine Nummer ist +49 151 23456789, bitte melden Sie sich.", language="de")
    assert "<PHONE>" in result.text
    assert "23456789" not in result.text
    assert "PHONE_NUMBER" in result.entities


def test_intl_phone_is_redacted():
    result = redact("Call me at (202) 555-0143 tomorrow.", language="en")
    assert "<PHONE>" in result.text
    assert "PHONE_NUMBER" in result.entities


def test_valid_iban_is_redacted_spaced_and_unspaced():
    spaced = redact("IBAN DE89 3704 0044 0532 0130 00 gehört ihm.", language="de")
    assert "<IBAN>" in spaced.text
    assert "IBAN_CODE" in spaced.entities

    unspaced = redact("IBAN DE89370400440532013000 gehört ihm.", language="de")
    assert "<IBAN>" in unspaced.text
    assert "IBAN_CODE" in unspaced.entities


def test_invalid_checksum_iban_is_not_flagged():
    """Presidio's IBAN_CODE recognizer validates the mod-97 checksum, not
    just the shape — a string that merely LOOKS like an IBAN (wrong check
    digits) is not real financial data, and is correctly left untouched
    (documented Presidio behaviour, not a bug in this module)."""
    result = redact("Die ungültige IBAN DE00 1234 5678 9012 3456 78 gehört ihm.", language="de")
    assert "IBAN_CODE" not in result.entities
    assert "DE00 1234 5678 9012 3456 78" in result.text


def test_credit_card_is_redacted():
    result = redact("Card 4111 1111 1111 1111 was used.", language="en")
    assert "<CREDIT_CARD>" in result.text
    assert "CREDIT_CARD" in result.entities


def test_ipv4_is_redacted():
    result = redact("The request came from 192.168.1.5 last night.", language="en")
    assert "<IP>" in result.text
    assert "IP_ADDRESS" in result.entities


def test_german_name_is_redacted():
    result = redact("Mein Mandant ist Hans Müller und er hat eine Frage.", language="de")
    assert "<PERSON>" in result.text
    assert "Hans Müller" not in result.text
    assert "PERSON" in result.entities


def test_english_name_is_redacted():
    result = redact("John Smith asked about the AI Act.", language="en")
    assert "<PERSON>" in result.text
    assert "John Smith" not in result.text
    assert "PERSON" in result.entities


# --- false positives on legal identifiers (must stay untouched) ----------


@pytest.mark.parametrize(
    "text",
    [
        "Article 6",
        "Art. 3(1)",
        "CELEX 32024R1689",
        "Regulation (EU) 2016/679",
        "GDPR",
        "AI Act",
    ],
)
def test_legal_identifiers_are_never_redacted(text):
    result = redact(text, language="en")
    assert result.text == text
    assert result.entities == ()


# --- language detection ----------------------------------------------------


def test_detect_language_german_stopwords_and_umlauts():
    assert detect_language("Der Mandant hat eine Frage für die Kanzlei.") == "de"


def test_detect_language_english_default():
    assert detect_language("What is a high-risk AI system under the Act?") == "en"


def test_detect_language_defaults_to_english_with_no_stopwords():
    """PII-only text has no stopwords in either language — must not crash,
    and must default to "en" (both languages' recognizers cover
    EMAIL_ADDRESS/PHONE_NUMBER/IBAN_CODE regardless of language anyway)."""
    assert detect_language("hans@firma.de +49 151 23456789") == "en"


# --- mixed sentence: multiple entity types + legal words survive ----------


def test_mixed_sentence_redacts_pii_and_keeps_legal_words():
    text = (
        "Mein Mandant ist Hans Müller. Seine E-Mail ist hans@firma.de und seine "
        "Nummer ist +49 151 23456789. Ist das nach Art. 6 der AI Act ein "
        "Hochrisiko-KI-System?"
    )
    result = redact(text, language="de")
    assert "<PERSON>" in result.text
    assert "<EMAIL>" in result.text
    assert "<PHONE>" in result.text
    assert "Hans Müller" not in result.text
    assert "hans@firma.de" not in result.text
    assert "23456789" not in result.text
    assert "Art. 6" in result.text
    assert "AI Act" in result.text
    assert set(result.entities) == {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}


def test_redaction_result_entities_are_types_only_never_values():
    result = redact("Contact jane.doe@example.com now.", language="en")
    for entity in result.entities:
        assert entity.isupper()
        assert "@" not in entity
        assert "jane" not in entity.lower()


def test_clean_text_is_unchanged():
    result = redact("What is a high-risk AI system?", language="en")
    assert result.text == "What is a high-risk AI system?"
    assert result.entities == ()


# --- guard_in_node wiring ---------------------------------------------------


class _PassLLM:
    """Records the messages it was invoked with — lets a test assert what
    the answer LLM actually saw, same minimal-double pattern as
    tests/test_graph.py's `FakeLLM`."""

    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AnswerSchema(answer="ok", citations=[])


class _UnusedLLM:
    def invoke(self, messages):
        raise AssertionError("LLM must not be called for a PII-only (refused) question")


def _run(question: str, llm, monkeypatch, retrieve_spy: list | None = None):
    def fake_retrieve(q, k, *, kinds, session, embeddings):
        if retrieve_spy is not None:
            retrieve_spy.append(q)
        return []

    monkeypatch.setattr("compliance_copilot.graph.nodes.retrieve", fake_retrieve)
    graph = build_graph()
    context = GraphContext(session=None, embeddings=None, llm=llm)
    return graph.invoke({"question": question}, context=context)


def test_question_with_pii_reaches_retrieve_and_llm_only_redacted(monkeypatch):
    """The retriever and the answer LLM must both receive REDACTED text
    (placeholders), never the raw name/email — proves guard_in's
    `question`-overwrite (nodes.py) actually propagates downstream."""
    retrieve_calls: list[str] = []
    llm = _PassLLM()

    state = _run(
        "My client Anna Schmidt, anna@x.de, asks: is she a deployer under the AI Act?",
        llm,
        monkeypatch,
        retrieve_spy=retrieve_calls,
    )

    assert retrieve_calls, "retrieve was never called"
    for q in retrieve_calls:
        assert "Anna Schmidt" not in q
        assert "anna@x.de" not in q
        assert "<PERSON>" in q or "<EMAIL>" in q

    # HTML-escaped inside the <question> tag (_build_messages, nodes.py) —
    # "<" -> "&lt;" — same as any other question text, so this checks for
    # the escaped form the LLM actually receives.
    human_content = llm.messages[-1][1]
    assert "Anna Schmidt" not in human_content
    assert "anna@x.de" not in human_content
    assert "&lt;PERSON&gt;" in human_content
    assert "&lt;EMAIL&gt;" in human_content

    assert "PERSON" in state["pii_entities"]
    assert "EMAIL_ADDRESS" in state["pii_entities"]
    assert state.get("refused") is not True


def test_pii_only_question_is_refused_with_pii_only_reason(monkeypatch):
    state = _run("hans@firma.de +49 151 23456789", _UnusedLLM(), monkeypatch)

    assert state["refused"] is True
    assert state["guard"].flagged is True
    assert state["guard"].reasons == ("pii_only",)


def test_pii_redaction_disabled_leaves_question_untouched(monkeypatch):
    monkeypatch.setattr(settings, "pii_redaction_enabled", False)
    llm = _PassLLM()

    state = _run("Contact John Smith at john@x.com about this.", llm, monkeypatch)

    assert "pii_entities" not in state
    human_content = llm.messages[-1][1]
    assert "John Smith" in human_content
    assert "john@x.com" in human_content


def test_guard_in_logs_entity_types_only_never_the_value(monkeypatch, caplog):
    monkeypatch.setattr("compliance_copilot.graph.nodes.retrieve", lambda *a, **kw: [])
    llm = _PassLLM()
    graph = build_graph()
    context = GraphContext(session=None, embeddings=None, llm=llm)

    with caplog.at_level(logging.INFO, logger="compliance_copilot.graph.nodes"):
        graph.invoke(
            {"question": "Reach John Smith at john.smith@example.com please."}, context=context
        )

    pii_logs = [r for r in caplog.records if "guard_in pii redacted" in r.getMessage()]
    assert len(pii_logs) == 1
    message = pii_logs[0].getMessage()
    assert "john.smith@example.com" not in message
    assert "John Smith" not in message
    assert "PERSON" in message or "EMAIL_ADDRESS" in message


def test_german_person_name_redacted_via_en_model_union():
    """de_core_news_sm alone missed this name (measured); the en+de union
    must catch it, and the legal reference must survive."""
    out = redact("Mein Mandant Hans Müller fragt nach Artikel 6 der Verordnung (EU) 2016/679.")
    assert "Hans Müller" not in out.text
    assert "PERSON" in out.entities
    assert "Artikel 6" in out.text and "2016/679" in out.text


def test_german_lowercase_phrase_not_over_redacted():
    """The en model tags 'zur Bewertung von Bewerbern' as PERSON; the
    name-shape filter must keep that content."""
    out = redact("Hans Müller betreibt ein KI-System zur Bewertung von Bewerbern.")
    assert "Hans Müller" not in out.text
    assert "zur Bewertung von Bewerbern" in out.text
