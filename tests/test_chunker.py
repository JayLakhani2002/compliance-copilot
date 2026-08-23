# tests/test_chunker.py — unit tests for ingest/chunker.py. No network, no
# DB: pure string-in/ChunkPart-out checks against hand-built ArticleChunks.
from compliance_copilot.ingest.chunker import split_article
from compliance_copilot.ingest.eurlex import ArticleChunk


def _article(text: str, number: int = 3, kind: str = "article") -> ArticleChunk:
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
    assert parts[0].anchor_id == "art_3"


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
    assert all(p.anchor_id == "art_3" for p in parts)
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


def test_article_3_realistic_size_splits_into_about_three_parts():
    # ~17k chars (Art. 3, AI Act, is ~17k chars in reality — ADR-0004),
    # expecting ~3 parts at the 6000-char cap, with "(N) " definition-list
    # numbering like the real Article 3 (see tests/fixtures/eurlex_sample.xhtml
    # for the actual "(1) 'AI system' means..." style this mirrors).
    paragraphs = [f"({i}) " + ("term definition text " * 11).strip() for i in range(1, 68)]
    text = " ".join(paragraphs)
    assert len(text) > 15000

    article = _article(text)
    parts = split_article(article, max_chars=6000)

    assert 2 <= len(parts) <= 4
    assert all(len(p.text) <= 6000 for p in parts)
