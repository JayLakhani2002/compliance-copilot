# tests/test_guards_output.py — pure unit tests for `guards/output.py`
# (ADR-0021): `check_output` is called directly with hand-built
# `AnswerSchema`s, no graph, no LLM, no DB — every reason code gets its own
# minimal fixture, same "prove the pure function in isolation" posture
# tests/test_guards_injection.py already takes for `detect()`.
from compliance_copilot.graph.state import AnswerSchema, Citation
from compliance_copilot.guards.output import CANARY, OutputVerdict, check_output

ARTICLE_KEY = ("ai_act", "art_6")
RETRIEVED = {ARTICLE_KEY}


def _answer(text: str, citations: list[Citation] | None = None) -> AnswerSchema:
    return AnswerSchema(answer=text, citations=citations or [])


# --- ok path -------------------------------------------------------------
def test_valid_answer_with_retrieved_citation_passes():
    answer = _answer(
        "A high-risk AI system is one used as a safety component.",
        [Citation(regulation="ai_act", anchor="art_6", quote="is a safety component")],
    )
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict == OutputVerdict(ok=True, reason=None)


def test_valid_zero_citation_short_answer_passes():
    """A short, honest "the excerpts don't cover this" answer with zero
    citations must not be flagged — only a LONG zero-citation answer is the
    scope_unsupported heuristic's target."""
    answer = _answer("The provided excerpts do not answer this question.")
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict.ok is True


# --- empty_answer ----------------------------------------------------------
def test_empty_answer_is_flagged():
    verdict = check_output(_answer("  "), retrieved_keys=None, refused=False)
    assert verdict == OutputVerdict(ok=False, reason="empty_answer")


def test_two_char_answer_is_flagged_empty():
    verdict = check_output(_answer("ok"), retrieved_keys=None, refused=False)
    assert verdict.reason == "empty_answer"


# --- canary_leak -----------------------------------------------------------
def test_canary_in_answer_text_is_flagged():
    answer = _answer(f"Sure, here it is: {CANARY}")
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict == OutputVerdict(ok=False, reason="canary_leak")


def test_canary_in_citation_quote_is_flagged():
    answer = _answer(
        "See the excerpt.",
        [Citation(regulation="ai_act", anchor="art_6", quote=f"blah {CANARY} blah")],
    )
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict.reason == "canary_leak"


# --- scaffold_leak -----------------------------------------------------------
def test_scaffold_leak_excerpt_tag_is_flagged():
    verdict = check_output(
        _answer("Per <excerpt regulation=...>, this applies."),
        retrieved_keys=RETRIEVED,
        refused=False,
    )
    assert verdict.reason == "scaffold_leak"


def test_scaffold_leak_question_tag_is_flagged():
    verdict = check_output(
        _answer("...as asked in </question> above."), retrieved_keys=RETRIEVED, refused=False
    )
    assert verdict.reason == "scaffold_leak"


def test_scaffold_leak_supporting_context_tag_is_flagged():
    verdict = check_output(
        _answer("Also see <supporting_context> for more."), retrieved_keys=RETRIEVED, refused=False
    )
    assert verdict.reason == "scaffold_leak"


def test_scaffold_leak_user_text_tag_is_flagged():
    verdict = check_output(
        _answer("The <user_text> tag wrapped your question."),
        retrieved_keys=RETRIEVED,
        refused=False,
    )
    assert verdict.reason == "scaffold_leak"


# --- placeholder_leak --------------------------------------------------------
def test_placeholder_person_leak_is_flagged():
    verdict = check_output(
        _answer("Regarding <PERSON>'s situation under the AI Act..."),
        retrieved_keys=RETRIEVED,
        refused=False,
    )
    assert verdict.reason == "placeholder_leak"


def test_placeholder_email_leak_is_flagged():
    verdict = check_output(
        _answer("You can reach them at <EMAIL> for more info."),
        retrieved_keys=RETRIEVED,
        refused=False,
    )
    assert verdict.reason == "placeholder_leak"


def test_placeholder_phone_and_iban_leak_is_flagged():
    assert (
        check_output(
            _answer("Call <PHONE> for details."), retrieved_keys=RETRIEVED, refused=False
        ).reason
        == "placeholder_leak"
    )
    assert (
        check_output(
            _answer("Wire it to <IBAN> as instructed."), retrieved_keys=RETRIEVED, refused=False
        ).reason
        == "placeholder_leak"
    )


# --- citation_not_retrieved (invariant, tested at the check_output level too)
def test_citation_not_in_retrieved_keys_is_flagged():
    answer = _answer(
        "...", [Citation(regulation="ai_act", anchor="art_99", quote="anything at all here")]
    )
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict.reason == "citation_not_retrieved"


def test_citation_not_retrieved_check_is_skipped_when_retrieved_keys_is_none():
    """`retrieved_keys=None` means "no retrieval happened this run" — the
    citation_not_retrieved check must not run at all (there's nothing to
    compare against), never treated as "nothing was retrieved -> flag."""
    answer = _answer(
        "A short answer.",
        [Citation(regulation="ai_act", anchor="art_99", quote="anything")],
    )
    verdict = check_output(answer, retrieved_keys=None, refused=False)
    assert verdict.ok is True


# --- scope_unsupported boundary (400 vs 401 chars, zero vs one citation) ---
def test_zero_citation_400_char_answer_passes_at_the_boundary():
    text = "x" * 400
    verdict = check_output(_answer(text), retrieved_keys=RETRIEVED, refused=False)
    assert verdict.ok is True


def test_zero_citation_401_char_answer_is_flagged_scope_unsupported():
    text = "x" * 401
    verdict = check_output(_answer(text), retrieved_keys=RETRIEVED, refused=False)
    assert verdict.reason == "scope_unsupported"


def test_one_citation_long_answer_does_not_trigger_scope_check():
    """The scope heuristic only fires on ZERO citations — a long, well-cited
    answer is exactly what a legitimate detailed answer looks like."""
    text = "x" * 500
    answer = _answer(
        text, [Citation(regulation="ai_act", anchor="art_6", quote="is a safety component")]
    )
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict.ok is True


# --- refused=True exemptions ------------------------------------------------
def test_refused_answer_with_no_citations_and_no_leak_passes():
    """The fixed refusal text — long, zero citations — must NOT trip
    scope_unsupported/placeholder/citation_not_retrieved: those checks don't
    apply at all once `refused=True`."""
    text = "I can only answer questions about the EU AI Act and GDPR " * 10
    verdict = check_output(_answer(text), retrieved_keys=None, refused=True)
    assert verdict == OutputVerdict(ok=True, reason=None)


def test_refused_answer_still_flags_canary_leak():
    verdict = check_output(
        _answer(f"refused, but leaked {CANARY}"), retrieved_keys=None, refused=True
    )
    assert verdict.reason == "canary_leak"


def test_refused_answer_still_flags_scaffold_leak():
    verdict = check_output(
        _answer("refused <excerpt but broken"), retrieved_keys=None, refused=True
    )
    assert verdict.reason == "scaffold_leak"


def test_refused_answer_still_flags_empty_answer():
    verdict = check_output(_answer(""), retrieved_keys=None, refused=True)
    assert verdict.reason == "empty_answer"


def test_refused_answer_ignores_placeholder_and_citation_checks():
    """A placeholder token or an unretrieved citation on a `refused=True`
    answer is NOT checked — those two are irrelevant to what "refused" even
    means (guard_out_node escalates this combination to a hard error at the
    node level, not at check_output — this test only proves check_output
    itself skips those two checks under `refused=True`)."""
    answer = _answer(
        "Contains <PERSON> but is marked refused.",
        [Citation(regulation="ai_act", anchor="art_99", quote="something not retrieved")],
    )
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=True)
    assert verdict == OutputVerdict(ok=True, reason=None)


# --- priority order: first failure wins -------------------------------------
def test_canary_leak_takes_priority_over_scaffold_leak():
    answer = _answer(f"<excerpt> leaked alongside {CANARY}")
    verdict = check_output(answer, retrieved_keys=RETRIEVED, refused=False)
    assert verdict.reason == "canary_leak"


def _ans(text: str) -> AnswerSchema:
    return AnswerSchema(answer=text, citations=[])


def test_review_bypasses_are_caught():
    """Review round: upper-cased canary, HTML-escaped scaffold, lowercase
    placeholder all slipped past exact substring checks."""
    assert check_output(_ans(CANARY.upper()), retrieved_keys=None, refused=False).reason == (
        "canary_leak"
    )
    assert check_output(
        _ans("see &lt;excerpt&gt; above"), retrieved_keys=None, refused=False
    ).reason == ("scaffold_leak")
    assert check_output(
        _ans("contact <person> now"), retrieved_keys=None, refused=False
    ).reason == ("placeholder_leak")


def test_long_honest_non_answer_passes_scope_check():
    text = ("The provided excerpts do not cover this question. " * 9).strip()
    assert len(text) > 400
    assert check_output(_ans(text), retrieved_keys=None, refused=False).ok
    # ...while a long confident zero-citation answer is still blocked.
    assert (
        check_output(
            _ans("Lorem ipsum dolor sit amet. " * 20), retrieved_keys=None, refused=False
        ).reason
        == "scope_unsupported"
    )
