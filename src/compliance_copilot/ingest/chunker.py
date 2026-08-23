# src/compliance_copilot/ingest/chunker.py — splits an oversize ArticleChunk
# (eurlex.py) into embeddable-sized parts (ADR-0004's "long articles" note,
# docs/lessons/04_embeddings_and_hnsw.md). Pure function, no DB/network, so
# it's unit-testable against fixture-sized strings alone.
#
# Why 6000 chars, not a token count: OpenAI's text-embedding-3-small input
# limit is 8191 *tokens*; counting tokens exactly means importing tiktoken
# and running it per-chunk just to pick a split point, for a limit we want
# well clear of anyway (context: legal text also needs to stay small enough
# that a retrieved chunk is a useful citation, not a wall of text). English
# legal prose runs roughly 4 chars/token, so 6000 chars ≈ 1500 tokens — ~5x
# headroom under the 8191-token cap, plenty for the char/token ratio to be
# off by a lot and still be safe. Article 3 (AI Act, ~17k chars) becomes
# ~3 parts under this cap.
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

import re
from dataclasses import dataclass

from compliance_copilot.ingest.eurlex import ArticleChunk

# Splits text right before a numbered-paragraph marker — either "12. " style
# (used by most articles' numbered paragraphs) or "(1) " style (used by
# definition lists like Article 3 — the real 17k-char oversize case this
# module exists for), both EUR-Lex's own in-text numbering kept inline by
# eurlex.py's parser — or a blank-line run, whichever boundaries the text
# actually has. Lookahead (not consuming the delimiter) so the number stays
# with the paragraph it starts.
_PARAGRAPH_BOUNDARY_RE = re.compile(r"(?=\s\d{1,3}\.\s)|(?=\s\(\d{1,3}\)\s)|\n{2,}")


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


def _pack_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedily packs paragraphs into parts under max_chars each. A single
    paragraph longer than max_chars gets its own oversize part rather than
    being cut mid-sentence — "never mid-sentence if avoidable" means avoiding
    it when there's a paragraph boundary to use, not guaranteeing every part
    obeys the cap when a single paragraph itself doesn't."""
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        added_len = para_len + (1 if current else 0)  # +1 for the joining space
        if current and current_len + added_len > max_chars:
            parts.append(" ".join(current))
            current, current_len = [], 0
            added_len = para_len
        current.append(para)
        current_len += added_len
    if current:
        parts.append(" ".join(current))
    return parts


def split_article(chunk: ArticleChunk, max_chars: int = 6000) -> list[ChunkPart]:
    """Splits one ArticleChunk into one or more ChunkParts.

    Under the cap: returned whole as a single part=0/part_count=1 part — the
    common case (most articles), unchanged text.

    Over the cap: split on paragraph boundaries, then greedily repacked up to
    max_chars per part, so a part is never smaller than it needs to be just
    because a paragraph landed near the cap.

    Recitals are never split (ADR-0004/lesson 04): a recital is short prose,
    one idea, and repeated across a regulation like the AI Act's 180 of them
    — splitting them would multiply row count for no retrieval benefit.
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
                content_hash=chunk.content_hash,
                part=0,
                part_count=1,
            )
        ]

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
            content_hash=chunk.content_hash,
            part=i,
            part_count=len(texts),
        )
        for i, text in enumerate(texts)
    ]
