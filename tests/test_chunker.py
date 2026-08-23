# tests/test_chunker.py — unit tests for ingest/chunker.py. No network, no
# DB: pure string-in/ChunkPart-out checks against hand-built ArticleChunks.
import re

from compliance_copilot.ingest.chunker import split_article
from compliance_copilot.ingest.eurlex import ArticleChunk


def _article(text: str, number: int = 9, kind: str = "article") -> ArticleChunk:
    # number defaults to 9 (art_9), not 3 — chunker.py's _DEFINITION_ARTICLES
    # routes exactly ("ai_act", "art_3") through the new definition-splitting
    # path (Day 6a); art_9 stays on the pre-existing generic paragraph path
    # these tests are about, so a plain "give me some oversize article" call
    # here doesn't silently start exercising different logic than intended.
    return ArticleChunk(
        regulation="ai_act",
        celex="32024R1689",
        kind=kind,
        number=number,
        title="Definitions" if kind == "article" else None,
        text=text,
        anchor_id=f"art_{number}" if kind == "article" else f"rct_{number}",
        content_hash="deadbeef",
    )


def test_whole_article_under_cap_is_unchanged():
    article = _article("Short article text.")
    parts = split_article(article, max_chars=6000)
    assert len(parts) == 1
    assert parts[0].text == "Short article text."
    assert parts[0].part == 0
    assert parts[0].part_count == 1
    assert parts[0].anchor_id == "art_9"


def test_oversize_article_splits_into_multiple_contiguous_parts():
    # Build a numbered-paragraph article well over the cap, mirroring
    # EUR-Lex's own in-text numbering style ("1. ...", "2. ...").
    paragraphs = [f"{i}. " + ("word " * 200).strip() + "." for i in range(1, 20)]
    text = " ".join(paragraphs)
    article = _article(text)

    parts = split_article(article, max_chars=1500)

    assert len(parts) > 1
    assert all(p.part_count == len(parts) for p in parts)
    assert [p.part for p in parts] == list(range(len(parts)))
    # every part keeps the parent's identity
    assert all(p.anchor_id == "art_9" for p in parts)
    assert all(p.regulation == "ai_act" for p in parts)
    assert all(p.title == "Definitions" for p in parts)
    # parts cover all the original text (paragraphs, not exact whitespace)
    rejoined = " ".join(p.text for p in parts)
    for para in paragraphs:
        assert para in rejoined
    # no part blew past the cap by an unreasonable margin (a single
    # over-cap paragraph is allowed to stand alone; packed multi-paragraph
    # parts must respect it)
    assert all(len(p.text) <= 1500 or paragraphs.count(p.text) for p in parts)
    # each part's content_hash is of ITS OWN text, not the parent article's
    # — every part here has different text, so every hash must differ too
    # (a shared/parent hash would make the pipeline's skip-if-unchanged
    # check blind to the chunker's own splitting changing).
    hashes = [p.content_hash for p in parts]
    assert len(set(hashes)) == len(parts)
    assert all(h != "deadbeef" for h in hashes)  # not the parent ArticleChunk's own hash


def test_recital_never_split_even_if_oversize():
    long_text = "word " * 3000  # way over any reasonable cap
    recital = _article(long_text, number=1, kind="recital")
    parts = split_article(recital, max_chars=100)
    assert len(parts) == 1
    assert parts[0].text == long_text
    assert parts[0].kind == "recital"


def test_article_3_realistic_size_splits_into_about_ten_parts():
    # ~17k chars (Art. 3, AI Act, is ~17k chars in reality), with "(N) " +
    # curly-quote definition-list numbering like the real Article 3.
    #
    # WHY THIS CHANGED (definition-aware splitting, ADR-0013 amendment):
    # before it, ("ai_act", "art_3") was
    # just another oversize article, packed ~3 parts at the 6000-char cap
    # (this test's old name/expectation). Now that exact (regulation,
    # anchor_id) pair is one of chunker.py's two designated definitions
    # articles (_DEFINITION_ARTICLES) — it takes the new definition-boundary
    # path, packed at the smaller 2000-char definition_max_chars, producing
    # ~10 parts instead. This test must use `number=3` explicitly (not the
    # `_article()` default of 9) specifically to exercise that path, and a
    # curly quote (’) after each "(N)" — the real definition-boundary marker
    # (chunker.py's _DEFINITION_BOUNDARY_RE) — not just any "(N) " numbering.
    paragraphs = [
        f"({i}) ‘term{i}’ means " + ("term definition text " * 11).strip() + ";"
        for i in range(1, 68)
    ]
    text = " ".join(paragraphs)
    assert len(text) > 15000

    article = _article(text, number=3)
    parts = split_article(article, max_chars=6000)

    assert 8 <= len(parts) <= 12
    assert all(len(p.text) <= 2000 for p in parts)
    assert "".join(p.text for p in parts) == text  # definition path is lossless


# --- Day 6a: definition-aware splitting (ADR-0013's residual gap, docs/
# handoffs/week2-day6a-defchunks/coder.md) ------------------------------
#
# Real excerpt pulled directly from the ingested `chunk` table (AI Act
# Art. 3, definitions (1)-(5)) via `SELECT text FROM chunk WHERE
# anchor_id = 'art_3'` against the local corpus — not hand-written, so the
# curly-quote characters and EUR-Lex's exact phrasing match what
# _DEFINITION_BOUNDARY_RE actually has to parse in production. Definition
# (1) ('AI system') is the long one — 373 chars, deliberately kept whole
# below with a max_chars small enough that it alone exceeds the cap.
_ART_3_EXCERPT = (
    "(1) ‘AI system’ means a machine-based system that is designed to operate "
    "with varying levels of autonomy and that may exhibit adaptiveness after "
    "deployment, and that, for explicit or implicit objectives, infers, from the "
    "input it receives, how to generate outputs such as predictions, content, "
    "recommendations, or decisions that can influence physical or virtual "
    "environments; "
    "(2) ‘risk’ means the combination of the probability of an occurrence of "
    "harm and the severity of that harm; "
    "(3) ‘provider’ means a natural or legal person, public authority, agency "
    "or other body that develops an AI system or a general-purpose AI model or "
    "that has an AI system or a general-purpose AI model developed and places it "
    "on the market or puts the AI system into service under its own name or "
    "trademark, whether for payment or free of charge; "
    "(4) ‘deployer’ means a natural or legal person, public authority, agency "
    "or other body using an AI system under its authority except where the AI "
    "system is used in the course of a personal non-professional activity; "
    "(5) ‘authorised representative’ means a natural or legal person located "
    "or established in the Union who has received and accepted a written mandate "
    "from a provider of an AI system or a general-purpose AI model to, "
    "respectively, perform and carry out on its behalf the obligations and "
    "procedures established by this Regulation; "
)

_DEFINITION_START_RE = re.compile(r"\(\d{1,3}\) ‘")


def _definitions_article(text: str) -> ArticleChunk:
    # anchor_id/regulation MUST be ("ai_act", "art_3") — that pair is the
    # only thing that routes split_article through the definition path
    # (chunker.py's _DEFINITION_ARTICLES).
    return ArticleChunk(
        regulation="ai_act",
        celex="32024R1689",
        kind="article",
        number=3,
        title="Definitions",
        text=text,
        anchor_id="art_3",
        content_hash="deadbeef",
    )


def test_definition_article_splits_at_definition_boundaries():
    article = _definitions_article(_ART_3_EXCERPT)
    # max_chars stays at the default 6000 (excerpt is far under it, so the
    # whole-article early return would otherwise fire) — force the
    # definition path with a small definition_max_chars instead, which is
    # exactly what production does with the real ~17k-char Article 3.
    parts = split_article(article, max_chars=100, definition_max_chars=500)

    assert len(parts) > 1
    # every part boundary lands right on a "(N) '...'" definition start —
    # never mid-sentence, never mid-definition.
    for part in parts:
        assert _DEFINITION_START_RE.match(part.text), (
            f"part {part.part} doesn't start on a definition boundary: {part.text[:40]!r}"
        )
    # "deployer" (definition 4) is whole and findable in exactly one part.
    deployer_parts = [p for p in parts if "‘deployer’ means" in p.text]
    assert len(deployer_parts) == 1
    assert (
        "except where the AI system is used in the course of a personal" in deployer_parts[0].text
    )


def test_definition_splitting_loses_no_text():
    article = _definitions_article(_ART_3_EXCERPT)
    parts = split_article(article, max_chars=100, definition_max_chars=500)
    # sep="" packing (chunker.py's _split_definitions) means concatenating
    # every part's text in order must reproduce the source exactly — not
    # just "contains the same words", byte-for-byte.
    assert "".join(p.text for p in parts) == _ART_3_EXCERPT


def test_definition_longer_than_cap_gets_its_own_part_unsplit():
    article = _definitions_article(_ART_3_EXCERPT)
    # Definition (1) ('AI system') is ~373 chars on its own — bigger than
    # this tiny cap, so it must still come back whole in its own part
    # rather than being cut mid-sentence (chunker.py's "never break inside
    # one definition" rule).
    parts = split_article(article, max_chars=100, definition_max_chars=50)
    ai_system_parts = [p for p in parts if p.text.startswith("(1) ‘AI system’ means")]
    assert len(ai_system_parts) == 1
    assert ai_system_parts[0].text.endswith(
        "recommendations, or decisions that can influence physical or virtual environments; "
    )


def test_non_definition_article_unaffected_by_definition_splitting():
    # Same definition-shaped text, but anchor_id "art_9" isn't in
    # _DEFINITION_ARTICLES — must take the old paragraph-boundary path,
    # which packs by `max_chars` only and never looks at
    # `definition_max_chars` at all. Proof: two calls that disagree wildly
    # on `definition_max_chars` (500 vs 5) must produce byte-identical
    # output for a non-designated article — that parameter is simply dead
    # code on this path, i.e. the old behaviour is genuinely untouched.
    article = ArticleChunk(
        regulation="ai_act",
        celex="32024R1689",
        kind="article",
        number=9,
        title="Risk management system",
        text=_ART_3_EXCERPT,
        anchor_id="art_9",
        content_hash="deadbeef",
    )
    route_a = split_article(article, max_chars=500, definition_max_chars=500)
    route_b = split_article(article, max_chars=500, definition_max_chars=5)
    assert [p.text for p in route_a] == [p.text for p in route_b]
    # And it's still the pre-existing paragraph splitter, not a no-op: text
    # got split (excerpt is 1372 chars, well over the 500-char max_chars).
    assert len(route_a) > 1
