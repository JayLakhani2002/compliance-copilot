# tests/test_persistence_integration.py — integration test for ADR-0024's
# Postgres checkpointer against a REAL Postgres instance (not `InMemorySaver`
# — tests/test_graph.py already covers the graph-side reset/history/prompt
# logic against that). This file proves the parts only real Postgres can
# prove: `.setup()` actually creates working tables, state genuinely
# survives a fresh connection pool + saver object (not just in-process
# object reuse), and `delete_thread` actually erases rows.
#
# Marked `integration` (pyproject.toml's pytest marker) — skipped by the
# default `pytest -m "not integration"` run; uses conftest.py's disposable
# "_test"-suffixed database (`test_database_url`), never DATABASE_URL
# directly, same reasoning tests/test_db_integration.py's module docstring
# already gives (a `reset=True` integration test against a real dev DB is
# exactly how you lose data — this file never resets a schema, but reuses
# the same fixture for the same "always a disposable DB" guarantee).
#
# `monkeypatch.setattr(settings, "database_url", test_database_url)`: the
# real `compliance_copilot.checkpointer.build_checkpointer()` reads
# `settings.database_url` — pointing that at the test DB (rather than
# reimplementing pool-building logic here) means this test exercises the
# actual production code path (`_checkpointer_dsn`'s URL surgery,
# `AsyncConnectionPool`, `AsyncPostgresSaver.setup()`), not a parallel copy
# of it.
import asyncio

import pytest
from fake_mcp_tools import tools_from_articles

from compliance_copilot.checkpointer import build_checkpointer
from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.state import AnswerSchema, GraphContext
from compliance_copilot.retriever import RetrievedChunk
from compliance_copilot.settings import settings

pytestmark = pytest.mark.integration

ARTICLES = [
    RetrievedChunk(
        anchor="art_6",
        regulation="ai_act",
        kind="article",
        number=6,
        title="Classification rules for high-risk AI systems",
        text="An AI system shall be considered high-risk where it is a safety component.",
        distance=0.1,
        part=0,
    ),
]

TURNS = [
    ("What is a high-risk AI system?", "High-risk means a safety component."),
    ("And does that include medical devices?", "Yes, per the same article."),
]


class FakeLLM:
    """Stands in for `runtime.context.llm` — same minimal double every other
    test file's `FakeLLM` uses (a real `ChatOpenAI`/`ChatAnthropic` +
    `with_structured_output` can't be faked with LangChain's own fakes, see
    tests/test_graph.py's module docstring)."""

    def __init__(self, response: AnswerSchema):
        self._response = response

    def invoke(self, messages):
        return self._response


async def _run_two_turns(thread_id: str) -> None:
    """Simulates one app process handling two `/ask` calls on the same
    thread_id — one `build_checkpointer()` open, one compiled graph, reused
    for both turns (mirrors how `api.py`'s `lifespan` opens the pool once
    and `cli.py`'s `_run_ask` opens it once per invocation)."""
    async with build_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        tools = tools_from_articles(ARTICLES)
        config = {"configurable": {"thread_id": thread_id}}
        for question, answer_text in TURNS:
            context = GraphContext(
                session=None,
                embeddings=None,
                llm=FakeLLM(AnswerSchema(answer=answer_text, citations=[])),
                tools=tools,
            )
            await graph.ainvoke({"question": question}, context=context, config=config)


async def _reload_history(thread_id: str) -> list:
    """A BRAND NEW `build_checkpointer()` call — a fresh `AsyncConnectionPool`
    + `AsyncPostgresSaver` instance, not the one `_run_two_turns` used — is
    what actually proves durability across a "process restart" rather than
    just in-process object reuse. `build_graph`'s `lru_cache` keys on the
    checkpointer object itself (identity-hashed), so a genuinely different
    saver instance is a cache MISS: this really does compile a new graph
    bound to the new saver, not silently reuse the old compiled graph."""
    async with build_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("history", [])


async def _delete_and_reload(thread_id: str) -> dict:
    async with build_checkpointer() as checkpointer:
        await checkpointer.adelete_thread(thread_id)
        graph = build_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values


def test_two_turn_conversation_survives_a_fresh_checkpointer_then_erases(
    test_database_url, monkeypatch
):
    monkeypatch.setattr(settings, "database_url", test_database_url)
    thread_id = "55555555-5555-4555-8555-555555555555"

    asyncio.run(_run_two_turns(thread_id))

    # Reload via a FRESH saver — proves durability, not just in-process
    # reuse (see `_reload_history`'s docstring).
    history = asyncio.run(_reload_history(thread_id))
    assert len(history) == 2
    assert [t.question for t in history] == [q for q, _ in TURNS]
    assert [t.answer for t in history] == [a for _, a in TURNS]

    # ADR-0024's erasure path (GDPR-flavoured): delete_thread, then confirm
    # a reload finds nothing at all — not just an empty `history` key.
    remaining_state = asyncio.run(_delete_and_reload(thread_id))
    assert remaining_state == {}
