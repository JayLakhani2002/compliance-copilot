# src/compliance_copilot/retriever.py — plain-function retrieval over the
# ingested chunk table (docs/ARCHITECTURE.md §1). Embeds a question with the
# same embeddings.py provider used at ingest time (ADR-0004's consistency
# requirement), then does cosine_distance ORDER BY ... LIMIT k — the same
# pattern cli.py's `search` command already uses, verified against
# pgvector-python's SQLAlchemy `Column.cosine_distance` API.
#
# Why a plain function, not a LangChain BaseRetriever subclass yet: the unit
# Day 5's eval (evals/run_retrieval_eval.py, ADR-0013) needs is "call this
# directly with two different `kinds` filters and measure hit@k/MRR on
# each" — wrapping it in LangChain's retriever interface now would add
# indirection with no benefit until the LangGraph agent actually needs it to
# be swappable inside a chain. Wrapping it in a thin BaseRetriever subclass
# later is a non-breaking addition (this function becomes its
# `_get_relevant_documents` body), not a rewrite.
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk, Document


@dataclass
class RetrievedChunk:
    """One retrieval result — everything a caller needs to cite or filter
    on, without holding onto the ORM row/session after this function
    returns (the session may be closed by then)."""

    anchor: str
    regulation: str
    kind: str
    number: int
    title: str | None
    text: str
    distance: float
    part: int


def retrieve(
    question: str,
    k: int = 5,
    *,
    kinds: tuple[str, ...] = ("article",),
    regulation: str | None = None,
    session: Session,
    embeddings: Embeddings,
) -> list[RetrievedChunk]:
    """Embeds `question` and returns the k nearest chunks by cosine distance.

    `kinds`: restrict to these Chunk.kind values. Default ("article",) is
    ADR-0013's decision (articles only — recitals fetched separately later
    as supporting context) after measuring both against the golden set;
    pass ("article", "recital") for the losing plain-cosine variant.

    `regulation`: restrict to one REGULATIONS key ("ai_act"/"gdpr"), or
    leave as None to search both (needed for the golden set's cross-
    regulation questions, evals/golden_retrieval.jsonl's "any" entries).
    """
    query_vector = embeddings.embed_query(question)
    # Cosine distance, not similarity: pgvector's cosine_distance is
    # 1 - cosine_similarity, so smaller is closer — ORDER BY it ascending
    # (no DESC) is what puts the nearest chunks first, matching cli.py's
    # `search` command and ADR-0003's choice of cosine as the metric for
    # OpenAI's normalised embeddings.
    distance = Chunk.embedding.cosine_distance(query_vector)

    stmt = (
        select(Chunk, Document.regulation, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.kind.in_(kinds))
    )
    if regulation is not None:
        stmt = stmt.where(Document.regulation == regulation)
    stmt = stmt.order_by(distance).limit(k)

    results = session.execute(stmt).all()
    return [
        RetrievedChunk(
            anchor=chunk.anchor_id,
            regulation=doc_regulation,
            kind=chunk.kind,
            number=chunk.number,
            title=chunk.title,
            text=chunk.text,
            distance=float(dist),
            part=chunk.part,
        )
        for chunk, doc_regulation, dist in results
    ]
