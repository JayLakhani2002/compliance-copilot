# src/compliance_copilot/mcp_server.py — the MCP server (docs/ARCHITECTURE.md
# §3, ADR-0007) exposing three read-only tools — `search_regulation`,
# `get_article`, `cite` — over the standard Model Context Protocol, so any
# MCP client (Day 17's `retrieve_node` via `langchain-mcp-adapters`, Claude
# Desktop, an IDE) can call the same retrieval logic this codebase already
# has in `retriever.py`/`db.py`, with no framework-specific glue.
#
# Built against `mcp` 1.29.1's `FastMCP` — the pre-rename v1 API, NOT the
# `MCPServer` class ADR-0007 originally named. `langchain-mcp-adapters`
# 0.3.2 pins `mcp<2.0.0`, so once both packages are actually installed
# together, `mcp>=2.0.0`'s `MCPServer` never resolves — see this file's ADR
# amendment for the full story.
#
# Each tool opens its own short-lived `Session(engine)` per call rather than
# holding one connection open for the server's whole lifetime — the engine
# (a connection pool, cheap to share) is the one thing built once, in
# `app_lifespan` below, mirroring `GraphContext` (graph/state.py).
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk, Document, get_engine
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph.nodes import _MIN_QUOTE_LENGTH, _normalise
from compliance_copilot.retriever import retrieve
from compliance_copilot.settings import settings

# Anchor id shape EUR-Lex/Cellar uses (eurlex.py's own `_ARTICLE_ID_RE`/
# `_RECITAL_ID_RE`, combined into one pattern) — validated again here since
# this module is the actual trust boundary an external MCP client's input
# crosses before it reaches a database query (ADR-0007 boundary #3). FastMCP
# turns the same pattern (passed via `Field(pattern=...)` below) into the
# tool's JSON schema, so a protocol-level `call_tool` is rejected before this
# function body even runs; the `match()` call here is what still protects a
# caller that invokes the plain Python function directly (this project's own
# unit tests, and any future in-process caller).
_ANCHOR_RE = re.compile(r"^(art|rct)_\d{1,3}$")


@dataclass
class AppContext:
    """Built once in `app_lifespan`, read by every tool call via
    `ctx.request_context.lifespan_context`. Holds a pooled `Engine` (not a
    live connection — each tool opens its own short `Session`) and an
    `Embeddings` provider, the same two dependencies `GraphContext`
    (graph/state.py) carries into the LangGraph nodes."""

    engine: Engine
    embeddings: object


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Runs once at server startup. `get_engine()`/`get_embeddings()` are
    both lazy constructors (see db.py's and embeddings.py's own "why lazy"
    docstrings) — building them here opens no DB connection and makes no
    OpenAI call, so `list_tools()` (schema discovery, no tool actually
    invoked) never touches either dependency."""
    yield AppContext(engine=get_engine(), embeddings=get_embeddings())


# host/port only matter for the "streamable-http" transport (main(), below)
# — FastMCP ignores them entirely for "stdio", so passing settings' values
# unconditionally here is simpler than branching on settings.mcp_transport
# just to decide whether to pass them.
mcp = FastMCP(
    "compliance-copilot", lifespan=app_lifespan, host=settings.mcp_host, port=settings.mcp_port
)


def _app_context(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
def search_regulation(
    question: Annotated[str, Field(min_length=3, max_length=2000)],
    ctx: Context,
    k: Annotated[int, Field(ge=1, le=20)] = 5,
    regulation: Literal["ai_act", "gdpr"] | None = None,
) -> list[dict]:
    """Searches the ingested EU AI Act and GDPR article text for the k passages
    nearest in meaning to `question`, returning each match's citation metadata
    and a short text snippet.

    question: what to search for, 3-2000 characters.
    k: how many results to return, 1-20.
    regulation: restrict to one regulation ("ai_act"/"gdpr"), or search both
    when omitted.
    """
    # Manual bounds re-check: FastMCP validates these `Field` constraints
    # against the JSON schema for a real protocol `call_tool`, but a direct
    # Python call (this project's own unit tests) bypasses that layer
    # entirely — see the module docstring above.
    if not 3 <= len(question) <= 2000:
        raise ValueError("question must be 3-2000 characters")
    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")
    app = _app_context(ctx)
    with Session(app.engine) as session:
        # kinds=("article",): ADR-0013's default — recitals are supporting
        # context, not a citable search target for this tool.
        chunks = retrieve(
            question,
            k=k,
            kinds=("article",),
            regulation=regulation,
            session=session,
            embeddings=app.embeddings,
        )
    return [
        {
            "regulation": c.regulation,
            "anchor": c.anchor,
            "title": c.title,
            "snippet": c.text[:300],
            "distance": c.distance,
            "part": c.part,
        }
        for c in chunks
    ]


@mcp.tool()
def get_article(
    regulation: Literal["ai_act", "gdpr"],
    anchor: Annotated[str, Field(pattern=_ANCHOR_RE.pattern)],
    ctx: Context,
) -> dict[str, Any]:
    """Fetches one full article or recital by its anchor id, joining all of its
    stored parts (chunker.py's oversize-article split) into one ordered text
    block.

    regulation: which regulation to look in ("ai_act" or "gdpr").
    anchor: the article/recital anchor id, e.g. "art_6" or "rct_12".
    """
    if not _ANCHOR_RE.match(anchor):
        raise ValueError(f"anchor must match {_ANCHOR_RE.pattern!r}, got {anchor!r}")
    app = _app_context(ctx)
    with Session(app.engine) as session:
        stmt = (
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.regulation == regulation, Chunk.anchor_id == anchor)
            .order_by(Chunk.part)
        )
        parts = session.execute(stmt).scalars().all()
    if not parts:
        raise ValueError(f"not found: {regulation} {anchor}")
    return {
        "regulation": regulation,
        "anchor": anchor,
        "title": parts[0].title,
        "text": " ".join(p.text for p in parts),
        "part_count": len(parts),
    }


@mcp.tool()
def cite(
    regulation: Literal["ai_act", "gdpr"],
    anchor: Annotated[str, Field(pattern=_ANCHOR_RE.pattern)],
    quote: str,
    ctx: Context,
) -> dict[str, Any]:
    """Checks whether `quote` appears verbatim (whitespace/case/quote-style
    insensitive) in the given article or recital, across all of its stored
    parts.

    regulation: which regulation the anchor belongs to.
    anchor: the article/recital anchor id, e.g. "art_6".
    quote: the exact excerpt to verify — at least 20 characters after
    normalisation.
    """
    if not _ANCHOR_RE.match(anchor):
        raise ValueError(f"anchor must match {_ANCHOR_RE.pattern!r}, got {anchor!r}")
    app = _app_context(ctx)
    with Session(app.engine) as session:
        stmt = (
            select(Chunk.text)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.regulation == regulation, Chunk.anchor_id == anchor)
        )
        parts = session.execute(stmt).scalars().all()
    if not parts:
        return {"valid": False, "reason": "not found"}
    # Reuses graph/nodes.py's own citation-verification logic (same
    # whitespace/case/curly-quote normalisation and minimum-length floor
    # `answer_node` already enforces) rather than a second implementation
    # that could silently drift from it — see this file's module docstring
    # and the researcher handoff's §5 for why importing from nodes.py (not
    # a new guards/quotes.py module) is the smaller diff: langchain-
    # anthropic/langchain-openai are already hard project dependencies, so
    # this import adds no new package, just an in-process function call.
    normalised_quote = _normalise(quote)
    if len(normalised_quote) < _MIN_QUOTE_LENGTH:
        return {"valid": False, "reason": "quote too short to verify"}
    if any(normalised_quote in _normalise(text) for text in parts):
        return {"valid": True, "reason": None}
    return {"valid": False, "reason": "quote not found verbatim"}


def main() -> None:
    """`python -m compliance_copilot.mcp_server` (Makefile's `make mcp`).
    Transport picked by `settings.mcp_transport` — stdio by default (dev/CI/
    `MultiServerMCPClient`'s spawn model), streamable-http only inside the
    Compose-internal network (ADR-0007, no auth on that transport yet)."""
    mcp.run(transport=settings.mcp_transport)


if __name__ == "__main__":
    main()
