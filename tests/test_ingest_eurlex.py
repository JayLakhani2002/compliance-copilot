# tests/test_ingest_eurlex.py — unit tests for the EUR-Lex parser, no
# network: parses the small committed fixture (Recitals 1-2, Articles 1-3 of
# the real AI Act XHTML — tests/fixtures/eurlex_sample.xhtml) and checks
# structure/metadata come out right, plus that fetch_xhtml's cache
# short-circuits the network entirely.
from pathlib import Path

import httpx
import pytest

from compliance_copilot.ingest import eurlex as eurlex_module
from compliance_copilot.ingest.eurlex import (
    ArticleChunk,
    EurLexFetchError,
    fetch_xhtml,
    parse_articles,
)

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
    # Must be >= _MIN_CACHE_BYTES or fetch_xhtml treats it as a truncated
    # cache file (see test_fetch_xhtml_treats_undersized_cache_as_miss below)
    # and re-fetches instead of trusting it.
    cached_content = "<html>cached</html>" + " " * 2000
    (tmp_path / f"{celex}.xhtml").write_text(cached_content, encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("fetch_xhtml should not hit the network when cache exists")

    monkeypatch.setattr("httpx.Client.get", _boom)

    result = fetch_xhtml(celex, cache_dir=tmp_path)
    assert result == cached_content


def test_fetch_xhtml_treats_undersized_cache_as_miss(tmp_path, monkeypatch):
    # A cache file smaller than _MIN_CACHE_BYTES looks like a process was
    # killed mid-write — fetch_xhtml must not trust it and must re-fetch.
    celex = "32024R1689"
    (tmp_path / f"{celex}.xhtml").write_text("<html>truncated</html>", encoding="utf-8")

    class _FakeResponse:
        status_code = 200
        text = "<html>fresh</html>" + " " * 2000

    monkeypatch.setattr("httpx.Client.get", lambda self, url: _FakeResponse())

    result = fetch_xhtml(celex, cache_dir=tmp_path)
    assert result == _FakeResponse.text
    # and the on-disk cache is now replaced with the fresh (larger) content
    assert (tmp_path / f"{celex}.xhtml").read_text(encoding="utf-8") == _FakeResponse.text


def test_fetch_xhtml_raises_eurlex_error_on_non_200(tmp_path, monkeypatch):
    class _FakeResponse:
        status_code = 404

    monkeypatch.setattr("httpx.Client.get", lambda self, url: _FakeResponse())

    with pytest.raises(EurLexFetchError, match="404"):
        fetch_xhtml("32024R1689", cache_dir=tmp_path)


def test_fetch_xhtml_raises_eurlex_error_on_network_failure(tmp_path, monkeypatch):
    def _raise_connect_error(self, url):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("httpx.Client.get", _raise_connect_error)

    with pytest.raises(EurLexFetchError):
        fetch_xhtml("32024R1689", cache_dir=tmp_path)


def test_ingest_regulation_raises_on_recital_count_mismatch(monkeypatch):
    # 113 articles (matches expected_articles for ai_act) but only 1 recital
    # (expected_recitals is 180) — the recital check must fire on its own,
    # independent of the article check.
    fake_articles = [
        ArticleChunk(
            regulation="ai_act",
            celex="32024R1689",
            kind="article",
            number=i,
            title=None,
            text="x",
            anchor_id=f"art_{i}",
            content_hash="h",
        )
        for i in range(1, 114)
    ]
    fake_recitals = [
        ArticleChunk(
            regulation="ai_act",
            celex="32024R1689",
            kind="recital",
            number=1,
            title=None,
            text="x",
            anchor_id="rct_1",
            content_hash="h",
        )
    ]

    monkeypatch.setattr(eurlex_module, "fetch_xhtml", lambda celex, cache_dir=None: "<html/>")
    monkeypatch.setattr(
        eurlex_module, "parse_articles", lambda xhtml, key: fake_articles + fake_recitals
    )

    with pytest.raises(ValueError, match="recitals"):
        eurlex_module.ingest_regulation("ai_act")
