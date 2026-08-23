# src/compliance_copilot/ingest/chunker.py — splits an oversize ArticleChunk
# (eurlex.py) into embeddable-sized parts (ADR-0004's re-ingestion/
# consistency notes on why chunk-level identity has to be stable). Pure
# function, no DB/network, so it's unit-testable against fixture-sized
# strings alone.
#
# Why 6000 chars, not a token count: OpenAI's text-embedding-3-small input
# limit is 8191 *tokens*; counting tokens exactly means importing tiktoken
# and running it per-chunk just to pick a split point, for a limit we want
# well clear of anyway (context: legal text also needs to stay small enough
# that a retrieved chunk is a useful citation, not a wall of text). English
# legal prose runs roughly 4 chars/token, so 6000 chars ≈ 1500 tokens — ~5x
# headroom under the 8191-token cap, plenty for the char/token ratio to be
# off by a lot and still be safe. Most oversize articles split ~3 parts
# under this cap; Article 3 (AI Act) and Article 4 (GDPR) are the two
# exceptions — definitions articles, routed through the smaller
# _DEFINITION_PART_MAX_CHARS cap below instead (Day 6a / ADR-0013's
# residual gap), giving Article 3 ~10 parts and Article 4 ~6.
#
# Why hand-rolled over langchain-text-splitters' RecursiveCharacterTextSplitter:
# that splitter's job is generic "keep chunks under N, prefer clean
# separators" — it doesn't know this text is EUR-Lex's own paragraph
# numbering ("1.", "2.", ...), which is the boundary that actually matters
# here (a split must never separate a numbered paragraph's number from its
# text, and citations should stay legible as "Art. 3(12)"-shaped chunks).
# Matching that exactly is not what RecursiveCharacterTextSplitter promises,
# so a dependency wouldn't save real code — the paragraph-boundary regex below
# plus greedy packing is the whole job.
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from compliance_copilot.ingest.eurlex import ArticleChunk


def _hash(text: str) -> str:
    # sha256 of THIS PART's own text, not the parent article's content_hash —
    # the pipeline's skip-if-unchanged check (pipeline.py) is keyed per part,
    # so two parts of the same oversize article must never share a hash: a
    # shared hash would make the skip check blind to the chunker's own
    # splitting logic changing (e.g. re-tuning max_chars) while the source
    # article text stays the same.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Splits text right before a numbered-paragraph marker — either "12. " style
# (used by most articles' numbered paragraphs) or "(1) " style (used by
# definition lists like Article 3 — the real 17k-char oversize case this
# module exists for), both EUR-Lex's own in-text numbering kept inline by
# eurlex.py's parser — or a blank-line run, whichever boundaries the text
# actually has. Lookahead (not consuming the delimiter) so the number stays
# with the paragraph it starts.
_PARAGRAPH_BOUNDARY_RE = re.compile(r"(?=\s\d{1,3}\.\s)|(?=\s\(\d{1,3}\)\s)|\n{2,}")

# (regulation, anchor_id) pairs whose article is a "definitions" article —
# ADR-0013's "known residual gap": Art. 3 (AI Act, 68 definitions) and Art. 4
# (GDPR, 26 definitions) are single oversize articles where one query term
# ("deployer", "controller") gets diluted inside a part covering dozens of
# unrelated terms. Definition-level splitting (below) is scoped to exactly
# these two articles — every other oversize article keeps the existing
# paragraph-boundary path unchanged.
_DEFINITION_ARTICLES = {("ai_act", "art_3"), ("gdpr", "art_4")}

# Definition parts are packed smaller than the general 6000-char cap (see
# split_article's max_chars) so each part holds "a handful" of related
# definitions, not dozens — the whole point is cutting dilution, so packing
# all the way back up to 6000 would reproduce most of the original problem.
_DEFINITION_PART_MAX_CHARS = 2000

# Matches right before a definition entry: "(N) '<term>'" — verified against
# the real ingested chunk text (SELECT text FROM chunk WHERE anchor_id =
# 'art_3' / 'art_4', see docs/handoffs/week2-day6a-defchunks/coder.md for the
# query + full evidence). Real examples pulled from that inspection:
#   AI Act art_3: "(4) ‘deployer’ means a natural or legal person, ..."
#   GDPR   art_4: "(7) ‘controller’ means the natural or legal person, ..."
#   GDPR   art_4: "(11) ‘consent’ of the data subject means any freely ..."
# Note the quote characters are Unicode curly quotes U+2018/U+2019 (‘ ’),
# not ASCII "'" — EUR-Lex's XHTML uses them throughout, confirmed by
# inspecting the parsed text (no straight quotes appear anywhere in either
# article). Deliberately NOT anchored on "... means" right after the closing
# quote: definition (58) in Art. 3 reads "(58) ‘subject’, for the purpose of
# real-world testing, means ..." and GDPR's (11) above has "‘consent’ of the
# data subject means ..." — both insert a clause between the term and
# "means", which a "quote then means" regex silently merges into the
# previous definition (verified: that stricter pattern found 67/68 Art. 3
# definitions, dropping (58); the pattern below finds all 68/68 and all
# 26/26 GDPR ones, with zero false positives from in-body cross-references
# like "Article 4, point (1), of Regulation (EU) 2016/679" — those never
# have a quote character immediately after the "(N)", so they don't match).
_DEFINITION_BOUNDARY_RE = re.compile(r"(?=\(\d{1,3}\)\s+‘)")


@dataclass
class ChunkPart:
    """One embeddable-sized piece of an ArticleChunk — carries everything a
    Chunk row (db.py) needs: the parent's identity (regulation/anchor_id/
    number/title/kind) plus this part's position among its siblings."""

    regulation: str
    celex: str
    kind: str
    number: int
    title: str | None
    anchor_id: str
    text: str
    content_hash: str
    part: int
    part_count: int


def _pack_paragraphs(paragraphs: list[str], max_chars: int, sep: str = " ") -> list[str]:
    """Greedily packs paragraphs into parts under max_chars each. A single
    paragraph longer than max_chars gets its own oversize part rather than
    being cut mid-sentence — "never mid-sentence if avoidable" means avoiding
    it when there's a paragraph boundary to use, not guaranteeing every part
    obeys the cap when a single paragraph itself doesn't.

    `sep` joins consecutive units within one packed part: " " for the
    stripped/normalised paragraphs the default caller passes, "" for
    definition units (_split_definitions below), whose text already carries
    its own original trailing whitespace up to the next definition's start —
    joining those with an extra " " would insert a space that isn't in the
    source text and break the "rejoin equals source" property definition
    splitting relies on.
    """
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        added_len = para_len + (len(sep) if current else 0)
        if current and current_len + added_len > max_chars:
            parts.append(sep.join(current))
            current, current_len = [], 0
            added_len = para_len
        current.append(para)
        current_len += added_len
    if current:
        parts.append(sep.join(current))
    return parts


def _split_definitions(text: str, max_chars: int) -> list[str]:
    """Splits a definitions article's text into per-definition units, then
    greedily packs those units up to max_chars per part (never splitting
    inside one definition — see _DEFINITION_PART_MAX_CHARS's comment).

    _DEFINITION_BOUNDARY_RE.split() on a definitions article yields
    [preamble, def_1, def_2, ..., def_N] — "preamble" is the short lead-in
    before the first "(1) '...'" (e.g. "For the purposes of this
    Regulation, the following definitions apply: "), too small to be a
    useful chunk on its own, so it's glued onto definition 1 rather than
    kept as a separate unit. Every split piece keeps its own trailing
    whitespace (the lookahead regex doesn't consume anything), so packing
    with sep="" and concatenating all returned parts in order reproduces
    the original text exactly.
    """
    segments = _DEFINITION_BOUNDARY_RE.split(text)
    if len(segments) <= 1:
        # No "(N) '...'" boundaries found — text isn't definition-shaped
        # (shouldn't happen for the two designated articles in practice, but
        # falling back to "whole text, one part" is safer than crashing).
        return [text]
    units = [segments[0] + segments[1], *segments[2:]]
    return _pack_paragraphs(units, max_chars, sep="")


def split_article(
    chunk: ArticleChunk,
    max_chars: int = 6000,
    definition_max_chars: int = _DEFINITION_PART_MAX_CHARS,
) -> list[ChunkPart]:
    """Splits one ArticleChunk into one or more ChunkParts.

    Under the cap: returned whole as a single part=0/part_count=1 part — the
    common case (most articles), unchanged text.

    Over the cap, for the two designated definitions articles (ADR-0013's
    residual gap — _DEFINITION_ARTICLES): split at definition boundaries via
    _split_definitions, packed up to definition_max_chars per part instead
    of max_chars, so a part holds a handful of related definitions rather
    than dozens of unrelated ones diluting each other's embedding.

    Over the cap, for every other article: split on paragraph boundaries,
    then greedily repacked up to max_chars per part, so a part is never
    smaller than it needs to be just because a paragraph landed near the
    cap. Unchanged from before this task.

    Recitals are never split (ADR-0004): a recital is short prose, one idea,
    and repeated across a regulation like the AI Act's 180 of them —
    splitting them would multiply row count for no retrieval benefit.
    """
    if chunk.kind == "recital" or len(chunk.text) <= max_chars:
        return [
            ChunkPart(
                regulation=chunk.regulation,
                celex=chunk.celex,
                kind=chunk.kind,
                number=chunk.number,
                title=chunk.title,
                anchor_id=chunk.anchor_id,
                text=chunk.text,
                content_hash=_hash(chunk.text),
                part=0,
                part_count=1,
            )
        ]

    if (chunk.regulation, chunk.anchor_id) in _DEFINITION_ARTICLES:
        texts = _split_definitions(chunk.text, definition_max_chars)
    else:
        paragraphs = [p.strip() for p in _PARAGRAPH_BOUNDARY_RE.split(chunk.text) if p.strip()]
        texts = _pack_paragraphs(paragraphs, max_chars)
    return [
        ChunkPart(
            regulation=chunk.regulation,
            celex=chunk.celex,
            kind=chunk.kind,
            number=chunk.number,
            title=chunk.title,
            anchor_id=chunk.anchor_id,
            text=text,
            content_hash=_hash(text),
            part=i,
            part_count=len(texts),
        )
        for i, text in enumerate(texts)
    ]
