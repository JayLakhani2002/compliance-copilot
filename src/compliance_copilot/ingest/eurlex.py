# src/compliance_copilot/ingest/eurlex.py — fetch + parse the EU AI Act and
# GDPR from the Publications Office's Cellar API into article/recital-level
# chunks (ADR-0012). This module is
# pure: no DB writes, no embeddings (that's Day 4's `embeddings.py`) — it
# only turns "a CELEX id" into "a list of ArticleChunk objects" so it can be
# unit-tested with a fixture file and no network.
#
# Why Cellar and not eur-lex.europa.eu directly: the human-facing EUR-Lex
# site's WAF returns an empty HTTP 202 to non-browser clients (verified by
# the planner, see ADR-0012's Verification section). Cellar is the
# Publications Office's machine-readable endpoint for the same authentic
# text, served via content negotiation (Accept: application/xhtml+xml).
#
# Why selectolax over lxml/BeautifulSoup: it wraps the Lexbor engine (fast,
# and its docs — https://selectolax.readthedocs.io/ / github.com/rushter/
# selectolax — confirm plain CSS id/class selectors like `#art_3` and
# `div.eli-title` work directly via `.css()`/`.css_first()`), so there's no
# reason to reach for a heavier dependency for what's a straightforward
# id-selector + text-extraction job.
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"

# User-Agent identifies this as a polite, reproducible crawler (per-project
# convention, not a legal requirement of Cellar) — a repo URL so anyone
# looking at Cellar's access logs can see who's hitting the endpoint and
# why (a contact email would be more conventional still; the repo URL gets
# them to the maintainer just as well and is what's actually set below).
REQUEST_HEADERS = {
    "Accept": "application/xhtml+xml",
    "Accept-Language": "eng",
    "User-Agent": "compliance-copilot/0.1 (+https://github.com/JayLakhani2002/compliance-copilot)",
}

# Anything smaller than this on disk is a truncated/corrupt cache file, not a
# real regulation document (the smallest of the two is GDPR at ~0.8 MB) —
# fetch_xhtml treats it as a cache miss and re-fetches rather than trusting it.
_MIN_CACHE_BYTES = 1024

# The two regulations this portfolio project answers questions about
# (ADR-0012). expected_articles/expected_recitals are sanity
# checks, not magic numbers: EUR-Lex's own consolidated-text counts for each
# regulation, verified via ADR-0012 — if a parse doesn't match, either the
# fetch got the wrong/partial document or the HTML structure changed under
# us and the parser needs fixing, not silent bad data.
REGULATIONS = {
    "ai_act": {
        "celex": "32024R1689",
        "title": "Regulation (EU) 2024/1689 (AI Act)",
        "expected_articles": 113,
        "expected_recitals": 180,
    },
    "gdpr": {
        "celex": "32016R0679",
        "title": "Regulation (EU) 2016/679 (GDPR)",
        "expected_articles": 99,
        "expected_recitals": 173,
    },
}

# Matches the anchor ids EUR-Lex/Cellar uses for the two citable unit types
# we chunk by (ADR-0012) — e.g. "art_3", "rct_12". Anchored with fullmatch so
# it doesn't also catch sub-ids like "art_3.tit_1" (the title's own id,
# nested *inside* the article's div).
_ARTICLE_ID_RE = re.compile(r"art_(\d+)")
_RECITAL_ID_RE = re.compile(r"rct_(\d+)")


class EurLexFetchError(RuntimeError):
    """Raised when fetching a regulation's XHTML from Cellar fails, for any
    reason — a non-200 response or a network-level failure (DNS/connect/
    timeout) alike. One exception type so a caller (the CLI, ingest_regulation)
    can catch "the fetch failed" without needing to know httpx's exception
    hierarchy or guess whether a given failure mode raises or returns."""


@dataclass
class ArticleChunk:
    """One article or recital, the citable unit the rest of the system
    chunks/embeds/cites by (ADR-0012) — not a fixed-size text window."""

    regulation: str  # REGULATIONS key, e.g. "ai_act"
    celex: str
    kind: str  # "article" | "recital"
    number: int
    title: str | None  # e.g. "Definitions" — articles only, recitals have none
    text: str  # normalised whitespace, paragraph numbers ("1.") kept inline
    anchor_id: str  # e.g. "art_3" — the source HTML id, for future deep links
    content_hash: str  # sha256(text) — detects when a re-fetch actually changed


def _normalise_whitespace(text: str) -> str:
    # EUR-Lex's XHTML is pretty-printed (newlines + indentation between every
    # tag); selectolax's .text() concatenates each text node with a space,
    # which leaves runs of whitespace where tags used to be. Collapse them so
    # chunk text reads like prose, not like the source's indentation.
    return re.sub(r"\s+", " ", text).strip()


def fetch_xhtml(celex: str, cache_dir: Path = Path("data/raw")) -> str:
    """Fetch a regulation's XHTML from Cellar by CELEX id, caching to disk.

    Cache-first: a second call (e.g. re-running ingest, or CI re-running
    tests) reads the local file instead of re-hitting Cellar — politeness
    towards a public EU service we don't need to hammer, and reproducibility
    (the exact bytes we parsed are on disk, not re-fetched and possibly
    changed underneath us).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{celex}.xhtml"
    # size check, not just .exists(): a file smaller than _MIN_CACHE_BYTES is
    # what a killed-mid-write process would leave *if* the write weren't
    # already atomic below — belt-and-braces against any cache file that
    # predates this fix, or was dropped there by hand.
    if cache_path.exists() and cache_path.stat().st_size >= _MIN_CACHE_BYTES:
        return cache_path.read_text(encoding="utf-8")

    url = CELLAR_URL.format(celex=celex)
    try:
        with httpx.Client(headers=REQUEST_HEADERS, timeout=60, follow_redirects=True) as client:
            response = client.get(url)
    except httpx.RequestError as exc:
        # RequestError is httpx's base for network-level failures (connect,
        # DNS, timeout — see the exception hierarchy in httpx/_exceptions.py)
        # as opposed to HTTPStatusError (a response that came back, just with
        # a bad status — handled below). Normalise both into the same
        # EurLexFetchError so callers don't need to know httpx's hierarchy.
        raise EurLexFetchError(f"EUR-Lex/Cellar fetch failed for CELEX {celex}: {exc!r}") from exc

    if response.status_code != 200:
        raise EurLexFetchError(
            f"EUR-Lex/Cellar fetch failed for CELEX {celex}: HTTP {response.status_code} from {url}"
        )

    # Atomic write: write to a temp file in the same directory, then
    # os.replace() into place. os.replace() is atomic on POSIX and Windows,
    # so a process killed mid-write (OOM, disk full, SIGKILL) never leaves a
    # half-written file at cache_path for a later call to mistake for a
    # valid cache hit — either the old cache (if any) or nothing is there.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(response.text, encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return response.text


def parse_articles(xhtml: str, regulation_key: str) -> list[ArticleChunk]:
    """Parse a regulation's XHTML into one ArticleChunk per article and
    per recital.

    Structure (inspected directly from the real AI Act XHTML, see ADR-0012
    for the endpoint and counts): each article/recital is a
    `<div class="eli-subdivision" id="art_N">` / `id="rct_N"`. An article's
    div additionally contains a `<p class="oj-ti-art">Article N</p>` label
    and a `<div class="eli-title"><p class="oj-sti-art">Title</p></div>` —
    both excluded from the extracted body text (spec: "text = all text of
    that element excluding the 'Article N' label and title line"). Recitals
    have neither label nor title, just numbered body content.
    """
    celex = REGULATIONS[regulation_key]["celex"]
    tree = HTMLParser(xhtml)
    chunks: list[ArticleChunk] = []

    for div in tree.css("div.eli-subdivision"):
        anchor_id = div.attributes.get("id") or ""

        if art_match := _ARTICLE_ID_RE.fullmatch(anchor_id):
            title_node = div.css_first("div.eli-title")
            title = _normalise_whitespace(title_node.text(deep=True)) if title_node else None
            if title_node:
                title_node.decompose()  # exclude from body text below

            label_node = div.css_first("p.oj-ti-art")  # "Article N" line
            if label_node:
                label_node.decompose()

            text = _normalise_whitespace(div.text(deep=True, separator=" "))
            chunks.append(
                ArticleChunk(
                    regulation=regulation_key,
                    celex=celex,
                    kind="article",
                    number=int(art_match.group(1)),
                    title=title,
                    text=text,
                    anchor_id=anchor_id,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
        elif rct_match := _RECITAL_ID_RE.fullmatch(anchor_id):
            text = _normalise_whitespace(div.text(deep=True, separator=" "))
            chunks.append(
                ArticleChunk(
                    regulation=regulation_key,
                    celex=celex,
                    kind="recital",
                    number=int(rct_match.group(1)),
                    title=None,
                    text=text,
                    anchor_id=anchor_id,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
        # else: some other eli-subdivision (chapter, section, annex) — not a
        # citable unit we chunk by (ADR-0012), skip it.

    return chunks


def ingest_regulation(key: str) -> list[ArticleChunk]:
    """Fetch + parse one regulation, then sanity-check both the article and
    recital counts against REGULATIONS[key]'s expected_articles/expected_recitals.

    Why assert here and not just trust the parser: this is the "count
    sanity" check ADR-0012 calls out — if EUR-Lex changes its
    HTML structure, a naive parser would silently return 0 (or a partial)
    list instead of failing loudly. A count mismatch is the cheapest
    possible signal that something upstream broke. Recitals get the same
    check as articles (not just articles) — a different id scheme change
    could break `rct_N` parsing while leaving `art_N` intact, and this is
    the only check the production ingest/CLI path (not just the integration
    test) actually runs.
    """
    if key not in REGULATIONS:
        raise ValueError(f"Unknown regulation key {key!r} — choices: {list(REGULATIONS)}")

    meta = REGULATIONS[key]
    xhtml = fetch_xhtml(meta["celex"])
    chunks = parse_articles(xhtml, key)
    articles = [c for c in chunks if c.kind == "article"]
    recitals = [c for c in chunks if c.kind == "recital"]

    if len(articles) != meta["expected_articles"]:
        raise ValueError(
            f"{key}: expected {meta['expected_articles']} articles, parsed {len(articles)}. "
            f"EUR-Lex's HTML structure may have changed — re-inspect the raw XHTML "
            f"(data/raw/{meta['celex']}.xhtml) before trusting this parse."
        )
    if len(recitals) != meta["expected_recitals"]:
        raise ValueError(
            f"{key}: expected {meta['expected_recitals']} recitals, parsed {len(recitals)}. "
            f"EUR-Lex's HTML structure may have changed — re-inspect the raw XHTML "
            f"(data/raw/{meta['celex']}.xhtml) before trusting this parse."
        )

    return chunks
