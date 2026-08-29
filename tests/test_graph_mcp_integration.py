# tests/test_graph_mcp_integration.py — integration tests proving the
# compiled graph actually retrieves through a REAL MCP tool/protocol round
# trip (ADR-0007's Day-17 amendment) — not a fake tool double
# (tests/fake_mcp_tools.py, unit-tested in tests/test_graph.py), and not
# just the MCP server in isolation (tests/test_mcp_server_integration.py).
# Uses the same real Postgres test-DB fixture corpus (conftest.py's
# `fixture_regulations`) the server-side integration tests use, and a
# fixed, zero-citation `_FakeLLM` (no answer-LLM cost) since the point here
# is proving the RETRIEVAL plumbing works end-to-end, not answer quality
# (evals/run_answer_eval.py's job).
from __future__ import annotations

import asyncio
import os

import pytest
from fake_embeddings import FakeEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session

from compliance_copilot import mcp_server
from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.state import AnswerSchema, GraphContext
from compliance_copilot.ingest import pipeline

pytestmark = pytest.mark.integration


class _FakeLLM:
    """A fixed, zero-citation answer (same shape
    tests/test_graph.py's `test_zero_citations_with_cannot_answer_text_is_
    accepted` uses) — no answer-LLM call, no citation to validate against
    whatever `FakeEmbeddings`-driven search actually ranked first."""

    def invoke(self, messages):
        return AnswerSchema(answer="The excerpts do not answer this question.", citations=[])


@pytest.fixture
def patched_mcp(monkeypatch, test_engine, fixture_regulations):
    """Same dependency-swap pattern tests/test_mcp_server_integration.py
    uses: points mcp_server's lazily-built engine/embeddings at the test DB
    / `FakeEmbeddings` instead of the real dev DB / OpenAI."""
    embeddings = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings, session)
    monkeypatch.setattr(mcp_server, "get_engine", lambda: test_engine)
    monkeypatch.setattr(mcp_server, "get_embeddings", lambda: embeddings)
    return mcp_server.mcp


async def _run_graph_with_tools(tools: dict) -> dict:
    graph = build_graph()
    context = GraphContext(session=None, embeddings=None, llm=_FakeLLM(), tools=tools)
    return await graph.ainvoke(
        {"question": "What are the definitions in this regulation?"}, context=context
    )


def test_graph_retrieves_through_in_process_mcp_session(patched_mcp):
    """`mcp.shared.memory.create_connected_server_and_client_session` +
    `langchain_mcp_adapters.tools.load_mcp_tools` — no subprocess, no port
    (lesson 17 §7's "in-process test server") — proves `retrieve_node`'s
    two-hop `search_regulation` -> `get_article` lookup, and the whole
    `guard_in -> retrieve -> answer -> guard_out` path, works through a
    REAL MCP protocol round trip against the test DB."""

    async def _run():
        async with create_connected_server_and_client_session(patched_mcp) as session:
            tools = {tool.name: tool for tool in await load_mcp_tools(session)}
            return await _run_graph_with_tools(tools)

    state = asyncio.run(_run())
    assert state["articles"]
    assert all(a.regulation == "ai_act" for a in state["articles"])
    # ADR-0007's Day-17 amendment: no MCP tool exposes recital search.
    assert state["recitals"] == []
    assert state["answer"].answer == "The excerpts do not answer this question."
    assert state.get("refused") is not True


# Gated the same way tests/test_search_real_embeddings_integration.py is —
# a `skipif` on just this ONE test, not a module-level skip (the
# in-process test above needs no network and must always run): a real
# `search_regulation` call over stdio needs a real OpenAI embeddings call
# (the spawned server process builds its OWN embeddings via
# `get_embeddings()` — `FakeEmbeddings` can't cross the subprocess boundary,
# same constraint ADR-0007's amendment already documents for the server-only
# stdio test). Never runs in CI's default job.
@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="RUN_NETWORK_TESTS != 1 or OPENAI_API_KEY unset — skipping real-embedding "
    "stdio-subprocess graph test",
)
def test_graph_retrieves_through_stdio_mcp_subprocess(test_engine, fixture_regulations):
    """The actual Day-17 production transport: `MultiServerMCPClient`
    spawns `uv run --frozen python -m compliance_copilot.mcp_server` as a
    real subprocess — exactly what `build.make_mcp_tools()` does — and the
    graph retrieves through it end to end."""
    embeddings_for_ingest = FakeEmbeddings()
    with Session(test_engine) as session:
        pipeline.ingest("ai_act", embeddings_for_ingest, session)

    env = {**os.environ, "DATABASE_URL": test_engine.url.render_as_string(hide_password=False)}
    connection = {
        "copilot": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "--frozen", "python", "-m", "compliance_copilot.mcp_server"],
            "env": env,
            "cwd": os.getcwd(),
        }
    }

    async def _run():
        client = MultiServerMCPClient(connection)
        tools = {tool.name: tool for tool in await client.get_tools()}
        return await _run_graph_with_tools(tools)

    state = asyncio.run(_run())
    assert state["articles"]
    assert state["answer"].answer == "The excerpts do not answer this question."
