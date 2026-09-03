# tests/test_mcp_server.py — unit tests for the MCP server
# (src/compliance_copilot/mcp_server.py, ADR-0007). No DB, no network: each
# tool function is called directly with a hand-built fake `Context` whose
# lifespan holds no real engine/embeddings, and `mcp_server.retrieve`/
# `mcp_server.Session` are monkeypatched to hand-made data — the same
# "monkeypatch the module-level name the code under test actually calls"
# pattern tests/test_graph.py already uses for `retrieve`.
import asyncio

import pytest

from compliance_copilot import mcp_server
from compliance_copilot.mcp_server import AppContext, cite, get_article, search_regulation
from compliance_copilot.retriever import RetrievedChunk


class _FakeRequestContext:
    def __init__(self, lifespan_context: AppContext) -> None:
        self.lifespan_context = lifespan_context


class FakeCtx:
    """Duck-types just enough of `mcp.server.fastmcp.Context` for
    `_app_context(ctx)` (mcp_server.py) to work: `.request_context.
    lifespan_context` holding an `AppContext`. Never a real MCP session."""

    def __init__(self, engine: object = None, embeddings: object = None) -> None:
        self.request_context = _FakeRequestContext(AppContext(engine=engine, embeddings=embeddings))


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeSession:
    """Stands in for `sqlalchemy.orm.Session` — `execute(stmt)` ignores the
    statement entirely and returns whatever rows the test configured, since
    `get_article`/`cite` only care about the shape of what comes back."""

    def __init__(self, rows: list) -> None:
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt):
        return _FakeResult(self.rows)


def _patch_session(monkeypatch, rows: list) -> None:
    monkeypatch.setattr(mcp_server, "Session", lambda engine: FakeSession(rows))


ARTICLE_6 = RetrievedChunk(
    anchor="art_6",
    regulation="ai_act",
    kind="article",
    number=6,
    title="Classification rules for high-risk AI systems",
    text="An AI system shall be considered high-risk where it is a safety component. " * 5,
    distance=0.12,
    part=0,
)


def test_search_regulation_shapes_results(monkeypatch):
    monkeypatch.setattr(mcp_server, "retrieve", lambda *a, **k: [ARTICLE_6])
    results = search_regulation("What is high-risk?", FakeCtx(), k=5, regulation="ai_act")
    assert results == [
        {
            "regulation": "ai_act",
            "anchor": "art_6",
            "title": "Classification rules for high-risk AI systems",
            "snippet": ARTICLE_6.text[:300],
            "distance": 0.12,
            "part": 0,
        }
    ]


def test_search_regulation_truncates_snippet_to_300_chars(monkeypatch):
    long_chunk = RetrievedChunk(
        anchor="art_3",
        regulation="ai_act",
        kind="article",
        number=3,
        title="Definitions",
        text="x" * 500,
        distance=0.0,
        part=0,
    )
    monkeypatch.setattr(mcp_server, "retrieve", lambda *a, **k: [long_chunk])
    results = search_regulation("definitions", FakeCtx())
    assert len(results[0]["snippet"]) == 300


@pytest.mark.parametrize("bad_k", [0, 21, -1])
def test_search_regulation_rejects_bad_k(monkeypatch, bad_k):
    monkeypatch.setattr(mcp_server, "retrieve", lambda *a, **kw: [])
    with pytest.raises(ValueError):
        search_regulation("a valid question", FakeCtx(), k=bad_k)


@pytest.mark.parametrize("question", ["ab", "x" * 2001])
def test_search_regulation_rejects_bad_question_length(monkeypatch, question):
    monkeypatch.setattr(mcp_server, "retrieve", lambda *a, **kw: [])
    with pytest.raises(ValueError):
        search_regulation(question, FakeCtx())


@pytest.mark.parametrize(
    "anchor", ["art_", "article_6", "art_6.tit_1", "rct_9999", "'; DROP TABLE chunk;--"]
)
def test_get_article_rejects_bad_anchor(monkeypatch, anchor):
    with pytest.raises(ValueError):
        get_article("ai_act", anchor, FakeCtx())


def test_get_article_joins_parts_in_order(monkeypatch):
    part0 = type("Row", (), {"title": "Definitions", "text": "Part one."})()
    part1 = type("Row", (), {"title": "Definitions", "text": "Part two."})()
    _patch_session(monkeypatch, [part0, part1])
    result = get_article("ai_act", "art_3", FakeCtx())
    assert result == {
        "regulation": "ai_act",
        "anchor": "art_3",
        "title": "Definitions",
        "text": "Part one. Part two.",
        "part_count": 2,
    }


def test_get_article_not_found_raises(monkeypatch):
    _patch_session(monkeypatch, [])
    with pytest.raises(ValueError, match="not found"):
        get_article("ai_act", "art_999", FakeCtx())


@pytest.mark.parametrize("anchor", ["not-an-anchor", "art_6.tit_1"])
def test_cite_rejects_bad_anchor(monkeypatch, anchor):
    with pytest.raises(ValueError):
        cite("ai_act", anchor, "a verbatim quote of sufficient length", FakeCtx())


def test_cite_valid_quote(monkeypatch):
    _patch_session(monkeypatch, ["An AI system shall be considered high-risk where necessary."])
    result = cite("ai_act", "art_6", "shall be considered high-risk", FakeCtx())
    assert result == {"valid": True, "reason": None}


def test_cite_invalid_quote_not_found(monkeypatch):
    _patch_session(monkeypatch, ["An AI system shall be considered high-risk where necessary."])
    result = cite("ai_act", "art_6", "this text was never in the source", FakeCtx())
    assert result == {"valid": False, "reason": "quote not found verbatim or close enough"}


def test_cite_quote_too_short(monkeypatch):
    _patch_session(monkeypatch, ["An AI system shall be considered high-risk where necessary."])
    result = cite("ai_act", "art_6", "AI system", FakeCtx())
    assert result == {"valid": False, "reason": "quote too short to verify"}


def test_cite_anchor_not_found(monkeypatch):
    _patch_session(monkeypatch, [])
    result = cite("ai_act", "art_999", "a verbatim quote of sufficient length", FakeCtx())
    assert result == {"valid": False, "reason": "not found"}


def test_list_tools_schema_matches_three_expected_tools():
    """In-process schema check (mcp.shared.memory, no subprocess, no DB):
    proves the pydantic -> JSON schema wiring produces exactly the 3 tools
    with the expected argument names/required-ness. Entering this context
    manager runs `app_lifespan` (builds an Engine + Embeddings, both lazy
    constructors — see mcp_server.py's docstrings), so this never opens a
    DB connection or makes a network call."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp) as session:
            result = await session.list_tools()
            by_name = {t.name: t for t in result.tools}
            assert set(by_name) == {"search_regulation", "get_article", "cite"}

            search_schema = by_name["search_regulation"].inputSchema
            assert set(search_schema["properties"]) == {"question", "k", "regulation"}
            assert search_schema["required"] == ["question"]

            article_schema = by_name["get_article"].inputSchema
            assert set(article_schema["properties"]) == {"regulation", "anchor"}
            assert set(article_schema["required"]) == {"regulation", "anchor"}

            cite_schema = by_name["cite"].inputSchema
            assert set(cite_schema["properties"]) == {"regulation", "anchor", "quote"}
            assert set(cite_schema["required"]) == {"regulation", "anchor", "quote"}

    asyncio.run(_run())
