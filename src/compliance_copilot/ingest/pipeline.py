# src/compliance_copilot/ingest/pipeline.py — wires eurlex.py (fetch+parse)
# -> chunker.py (split oversize articles) -> embeddings.py (vectorise) -> db.py
# (upsert) into the one function the CLI's `ingest` command calls
# (docs/ARCHITECTURE.md §1, ADR-0012/ADR-0004). Everything above this layer
# is pure/no-side-effects; this is where the DB actually gets written to.
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk, Document
from compliance_copilot.ingest.chunker import ChunkPart, split_article
from compliance_copilot.ingest.eurlex import REGULATIONS, fetch_xhtml, parse_articles

logger = logging.getLogger(__name__)

# How many texts go into one embed_documents() call. OpenAI's embeddings
# endpoint accepts up to 2048 inputs per request, but batching at a smaller
# size keeps a single failed batch's blast radius (and retry cost) small —
# 64 is comfortably under any per-request payload-size limit for
# 6000-char-max chunks (64 * 6000 chars ~= 384KB, well within request-body
# limits) while still cutting round-trips ~64x versus one-call-per-chunk.
EMBED_BATCH_SIZE = 64


@dataclass
class IngestStats:
    documents: int
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int


def _embed_in_batches(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
    return vectors


def _upsert_document(session: Session, key: str) -> Document:
    meta = REGULATIONS[key]
    existing = session.scalars(select(Document).where(Document.celex == meta["celex"])).first()
    if existing:
        return existing
    doc = Document(
        regulation=key,
        celex=meta["celex"],
        title=meta["title"],
        source_url=f"https://publications.europa.eu/resource/celex/{meta['celex']}",
        fetched_at=datetime.now(UTC),
    )
    session.add(doc)
    session.flush()  # assigns doc.id for the chunks below, without committing yet
    return doc


def ingest(
    regulation_key: str,
    embeddings: Embeddings,
    session: Session,
    *,
    dry_run: bool = False,
) -> IngestStats:
    """Fetch+parse `regulation_key`, split oversize articles, embed and
    upsert every part into `chunk`. Idempotent: a part whose
    `(document_id, anchor_id, part)` already exists with the same
    `content_hash` is skipped entirely — no re-embed, no DB write — so a
    second run against unchanged text costs nothing (lesson 04's
    idempotency point). `dry_run=True` does everything except the final
    commit, so callers can see the stats without writing anything.
    """
    xhtml = fetch_xhtml(REGULATIONS[regulation_key]["celex"])
    articles = parse_articles(xhtml, regulation_key)

    parts: list[ChunkPart] = [p for article in articles for p in split_article(article)]

    doc = _upsert_document(session, regulation_key)

    # Which (anchor_id, part) -> content_hash already exist for this
    # document, fetched once instead of one SELECT per part.
    existing_hashes: dict[tuple[str, int], str] = {
        (row.anchor_id, row.part): row.content_hash
        for row in session.scalars(select(Chunk).where(Chunk.document_id == doc.id))
    }

    to_embed = [p for p in parts if existing_hashes.get((p.anchor_id, p.part)) != p.content_hash]
    chunks_skipped = len(parts) - len(to_embed)

    vectors = _embed_in_batches(embeddings, [p.text for p in to_embed]) if to_embed else []
    # OpenAIEmbeddings carries its model name as `.model`; a fake/other
    # Embeddings implementation may not — record what's available rather
    # than importing OpenAIEmbeddings here just to type-check for it (that
    # would reintroduce the provider-specific coupling embeddings.py exists
    # to avoid, per ADR-0004).
    model_name = getattr(embeddings, "model", type(embeddings).__name__)

    for part, vector in zip(to_embed, vectors, strict=True):
        stmt = (
            insert(Chunk)
            .values(
                document_id=doc.id,
                kind=part.kind,
                number=part.number,
                anchor_id=part.anchor_id,
                title=part.title,
                text=part.text,
                part=part.part,
                part_count=part.part_count,
                content_hash=part.content_hash,
                embedding=vector,
                embedding_model=model_name,
            )
            .on_conflict_do_update(
                index_elements=["document_id", "anchor_id", "part"],
                set_={
                    "title": part.title,
                    "text": part.text,
                    "part_count": part.part_count,
                    "content_hash": part.content_hash,
                    "embedding": vector,
                    "embedding_model": model_name,
                },
            )
        )
        session.execute(stmt)

    if dry_run:
        session.rollback()
    else:
        session.commit()

    stats = IngestStats(
        documents=1,
        chunks_total=len(parts),
        chunks_embedded=len(to_embed),
        chunks_skipped=chunks_skipped,
    )
    logger.info(
        "ingest(%s): %d documents, %d chunks total, %d embedded, %d skipped (unchanged)",
        regulation_key,
        stats.documents,
        stats.chunks_total,
        stats.chunks_embedded,
        stats.chunks_skipped,
    )
    return stats
