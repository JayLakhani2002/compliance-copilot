# tests/test_mcp_server_integration.py — integration tests for the MCP
# server (src/compliance_copilot/mcp_server.py, ADR-0007) against a real
# Postgres+pgvector test DB (conftest.py's `test_engine`) with the small,
# real AI Act fixture ingested (`fixture_regulations`) using FakeEmbeddings
# (tests/fake_embeddings.py) — no OpenAI cost, no network call.
#
# Two client paths, both real MCP protocol round-trips (unlike
# test_mcp_server.py's direct function calls): the in-process client
# (mcp.shared.memory, fast, no subprocess) for most assertions, plus ONE
# real stdio subprocess round-trip (`mcp.client.stdio`) — the actual Day-17
# production transport — that only exercises the DB-only, no-embeddings
# paths (list_tools, get_article), since FakeEmbeddings can't cross a
# process boundary and this test doesn't want a real OpenAI call.
import asyncio
import os

import pytest
from fake_embeddings import FakeEmbeddings
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session

from compliance_copilot import mcp_server
from compliance_copilot.ingest import pipeline

pytestmark = pytest.mark.integration


@pytest.fixture
def patched_mcp(monkeypatch, test_engine, fixture_regulations):
    """Ingests the small real AI Act fixture (art_1/art_2/art_3, rct_1/
    rct_2 — conftest.py's `fixture_regulations`) into the test DB with
    FakeEmbeddings, then monkeypatches mcp_server's lazily-built engine/
    embeddings (`app_lifespan` calls these by name) to point at THIS test
    DB/provider instead of the real dev DB / OpenAI — same dependency-swap
    pattern test_graph.py already uses for `retrieve`."""
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)
    monkeypatch.setattr(mcp_server, "get_engine", lambda: test_engine)
    monkeypatch.setattr(mcp_server, "get_embeddings", lambda: embeddings)
    return mcp_server.mcp


async def _call_tool(mcp, name: str, arguments: dict):
    async with create_connected_server_and_client_session(mcp) as session:
        return await session.call_tool(name, arguments)


def test_search_regulation_returns_fixture_articles(patched_mcp):
    result = asyncio.run(
        _call_tool(
            patched_mcp, "search_regulation", {"question": "What are the definitions?", "k": 3}
        )
    )
    assert not result.isError
    rows = result.structuredContent["result"]
    assert len(rows) >= 1
    assert all(row["regulation"] == "ai_act" for row in rows)
    assert all(row["anchor"].startswith("art_") for row in rows)


def test_get_article_art_1_returns_joined_text(patched_mcp):
    result = asyncio.run(
        _call_tool(patched_mcp, "get_article", {"regulation": "ai_act", "anchor": "art_1"})
    )
    assert not result.isError
    payload = result.structuredContent
    assert payload["regulation"] == "ai_act"
    assert payload["anchor"] == "art_1"
    assert payload["text"]
    assert payload["part_count"] >= 1


def test_cite_valid_and_invalid_against_real_fixture_text(patched_mcp):
    article = asyncio.run(
        _call_tool(patched_mcp, "get_article", {"regulation": "ai_act", "anchor": "art_1"})
    )
    real_quote = article.structuredContent["text"][:40]

    valid = asyncio.run(
        _call_tool(
            patched_mcp, "cite", {"regulation": "ai_act", "anchor": "art_1", "quote": real_quote}
        )
    )
    assert valid.structuredContent == {"valid": True, "reason": None}

    invalid = asyncio.run(
        _call_tool(
            patched_mcp,
            "cite",
            {
                "regulation": "ai_act",
                "anchor": "art_1",
                "quote": "this exact sentence was fabricated and never appears",
            },
        )
    )
    assert invalid.structuredContent == {"valid": False, "reason": "quote not found verbatim"}


def test_stdio_roundtrip_list_tools_and_get_article(test_engine, fixture_regulations):
    """The ONE real subprocess test (lesson 16: "one real stdio round-trip
    keeps the protocol boundary honest") — spawns `uv run --frozen python -m
    compliance_copilot.mcp_server` for real over stdio, matching the actual
    Day-17 production transport. Only touches `list_tools` (no DB) and
    `get_article` (DB only) — no `search_regulation`/embeddings call, since
    FakeEmbeddings lives in this test process and can't cross into the
    spawned child (see module docstring)."""
    embeddings_for_ingest = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings_for_ingest, session)

    # Merge into the CURRENT environment (not a bare dict) so the spawned
    # `uv run` subprocess still has PATH/HOME/etc. to find `uv` and resolve
    # the project — StdioServerParameters.env, if given, REPLACES the
    # default-inherited set entirely rather than adding to it (verified
    # against the installed mcp.client.stdio source).
    # render_as_string(hide_password=False), not str(url): str() masks the
    # password as "***" (conftest.py's own `_resolve_test_database_url`
    # makes the same point) — useless as an actual connection string for
    # the child process to connect with.
    env = {**os.environ, "DATABASE_URL": test_engine.url.render_as_string(hide_password=False)}
    params = StdioServerParameters(
        command="uv",
        args=["run", "--frozen", "python", "-m", "compliance_copilot.mcp_server"],
        env=env,
        cwd=os.getcwd(),
    )

    async def _run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == {
                    "search_regulation",
                    "get_article",
                    "cite",
                }
                result = await session.call_tool(
                    "get_article", {"regulation": "ai_act", "anchor": "art_1"}
                )
                assert not result.isError
                assert result.structuredContent["anchor"] == "art_1"
                assert result.structuredContent["text"]

    asyncio.run(_run())
