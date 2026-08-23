# tests/test_ingest_eurlex.py — unit tests for the EUR-Lex parser, no
# network: parses the small committed fixture (Recitals 1-2, Articles 1-3 of
# the real AI Act XHTML — tests/fixtures/eurlex_sample.xhtml) and checks
# structure/metadata come out right, plus that fetch_xhtml's cache
# short-circuits the network entirely.
from pathlib import Path

import pytest

from compliance_copilot.ingest.eurlex import fetch_xhtml, parse_articles

FIXTURE = Path(__file__).parent / "fixtures" / "eurlex_sample.xhtml"


@pytest.fixture
def sample_xhtml() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_articles_counts(sample_xhtml):
    chunks = parse_articles(sample_xhtml, "ai_act")
    articles = [c for c in chunks if c.kind == "article"]
    recitals = [c for c in chunks if c.kind == "recital"]
    assert len(articles) == 3
    assert len(recitals) == 2


def test_article_3_title_is_definitions(sample_xhtml):
    chunks = parse_articles(sample_xhtml, "ai_act")
    art3 = next(c for c in chunks if c.kind == "article" and c.number == 3)
    assert art3.title is not None
    assert "Definitions" in art3.title
    # The "Article 3" label and "Definitions" title line must not leak into
    # the body text — that's the whole point of decomposing them first.
    assert "Article 3" not in art3.text
    assert "Definitions" not in art3.text


def test_article_1_text_mentions_harmonised_rules(sample_xhtml):
    chunks = parse_articles(sample_xhtml, "ai_act")
    art1 = next(c for c in chunks if c.kind == "article" and c.number == 1)
    assert "harmonised rules" in art1.text


def test_anchor_ids_and_content_hashes_present(sample_xhtml):
    chunks = parse_articles(sample_xhtml, "ai_act")
    assert chunks  # sanity: fixture actually produced something
    for chunk in chunks:
        assert chunk.anchor_id  # e.g. "art_3" / "rct_1"
        assert chunk.content_hash
        assert len(chunk.content_hash) == 64  # sha256 hex digest length
        assert chunk.regulation == "ai_act"
        assert chunk.celex == "32024R1689"


def test_fetch_xhtml_uses_cache_no_network(tmp_path, monkeypatch):
    celex = "32024R1689"
    (tmp_path / f"{celex}.xhtml").write_text("<html>cached</html>", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("fetch_xhtml should not hit the network when cache exists")

    monkeypatch.setattr("httpx.Client.get", _boom)

    result = fetch_xhtml(celex, cache_dir=tmp_path)
    assert result == "<html>cached</html>"
