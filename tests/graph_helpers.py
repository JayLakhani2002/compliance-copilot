# tests/graph_helpers.py — shared graph-test fixtures and fake LLM doubles,
# used by both tests/test_graph.py (ADR-0014/0023/0024/0025's own unit
# tests) and tests/evals/test_trajectory.py (lesson 21's trajectory evals,
# ADR-0026). Extracted out of test_graph.py rather than duplicated: both
# files need the exact same `FakeLLM`/`StatefulLLM`/`_run`/`_run_stream`
# doubles to drive the compiled graph with zero network — a second
# hand-copied set would drift the moment either file's fixture changed.
#
# No `test_` prefix, so pytest never collects this file as its own test
# module — it's a plain importable helper, same role `fake_mcp_tools.py`
# already plays for the MCP tool doubles.
from __future__ import annotations

import asyncio

from fake_mcp_tools import tools_from_articles
from langgraph.types import Command

from compliance_copilot.critic import CriticVerdict
from compliance_copilot.graph import AnswerSchema, GraphContext
from compliance_copilot.graph.build import build_graph
from compliance_copilot.retriever import RetrievedChunk
from compliance_copilot.router import RouterVerdict

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
    RetrievedChunk(
        anchor="art_9",
        regulation="ai_act",
        kind="article",
        number=9,
        title="Risk management system",
        text="A risk management system shall be established for high-risk AI systems.",
        distance=0.2,
        part=0,
    ),
]


class FakeLLM:
    """Stands in for `runtime.context.llm` — the only contract answer_node()
    depends on is `.invoke(messages) -> AnswerSchema` (see nodes.py's
    module docstring and ADR-0014)."""

    def __init__(self, response: AnswerSchema):
        self._response = response
        self.messages: list[tuple[str, str]] | None = None

    def invoke(self, messages):
        self.messages = messages  # captured for the prompt-content assertions
        return self._response


class StatefulLLM:
    """Returns each response in order on successive `.invoke()` calls, and
    records every call's messages (not just the last, unlike `FakeLLM`) —
    drives the retry-once loop (bad citation on call 1, good on call 2) with
    no network, same hand-written-double approach `FakeLLM` above already
    uses (see this file's module docstring)."""

    def __init__(self, responses: list[AnswerSchema]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeRouterLLM:
    """Stands in for `runtime.context.router` — the only contract
    `router_node` depends on is `.invoke(messages) -> RouterVerdict` (see
    router.py's module docstring), same hand-written-double approach as
    `FakeLLM` above."""

    def __init__(self, verdict: RouterVerdict):
        self._verdict = verdict
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self._verdict


class FakeCriticLLM:
    """Stands in for `runtime.context.critic` — the only contract
    `critic_node` depends on is `.invoke(messages) -> CriticVerdict`."""

    def __init__(self, verdict: CriticVerdict):
        self._verdict = verdict
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self._verdict


class RaisingLLM:
    """A minimal double whose `.invoke()` always raises — simulates an LLM
    outage for `route()`/`critique()`'s fail-* paths without a real network
    call."""

    def invoke(self, messages):
        raise ConnectionError("simulated outage")


class CountingLLM:
    """Like `FakeCriticLLM`/`FakeLLM` but counts `.invoke()` calls — used
    wherever a test needs an exact call count, not just "did it raise"."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self._verdict


class _UnusedLLM:
    """Raises on any `.invoke()` — used on the RESUME side of a paused-run
    test, where no node between `hitl` and `guard_out` ever calls an LLM.
    A call here would mean something re-ran an LLM-backed node on resume."""

    def invoke(self, messages):
        raise AssertionError("no LLM call should happen on resume")


def _low_confidence_critic(reasoning: str = "not well supported") -> FakeCriticLLM:
    return FakeCriticLLM(CriticVerdict(faithful=False, confidence=0.1, reasoning=reasoning))


# --- ADR-0023: the router — ten hand-labelled questions ------------------
# Question 7 is deliberately the exact cross-regulation example lesson 18's
# own "Check yourself" section names — a regression pin for the collision
# case the router exists to solve (art_3 is a real anchor in BOTH the AI Act
# and GDPR). Reused by tests/test_graph.py's own per-question mechanism
# tests AND tests/evals/test_trajectory.py's aggregate accuracy eval and the
# gated real-model test (tests/test_router_real_integration.py) — one
# fixture, three consumers, never copied.
ROUTER_FIXTURE = [
    (
        "What obligations does a provider have when placing a high-risk AI system on the market?",
        "ai_act",
    ),
    (
        "Under what conditions can an AI system be classified as high-risk under Article 6?",
        "ai_act",
    ),
    (
        "What are the risk management requirements for providers of high-risk AI "
        "systems under Article 9?",
        "ai_act",
    ),
    (
        "What is the legal basis required for processing special category data under GDPR?",
        "gdpr",
    ),
    (
        "What rights does a data subject have to obtain human intervention in an "
        "automated decision under Article 22?",
        "gdpr",
    ),
    ("What is a data protection impact assessment and when is it required?", "gdpr"),
    (
        "Does Article 6 of the AI Act interact with GDPR's consent requirements "
        "for automated decisions?",
        "both",
    ),
    (
        "How do the AI Act's transparency obligations for high-risk systems relate "
        "to GDPR's data-subject information rights?",
        "both",
    ),
    ("What is the best recipe for a German sauerbraten?", "out_of_scope"),
    ("Can you help me write a Python script to scrape a website?", "out_of_scope"),
]


def _run(
    llm,
    question: str = "What is a high-risk AI system?",
    *,
    articles: list[RetrievedChunk] = ARTICLES,
    tools: dict | None = None,
    classifier=None,
    router=None,
    critic=None,
):
    """Runs the compiled graph once via `graph.ainvoke` (wrapped in
    `asyncio.run` — see the module docstring). `tools` defaults to fake
    `search_regulation`/`get_article` doubles built from `articles`
    (`fake_mcp_tools.tools_from_articles`, ADR-0007's Day-17 amendment) —
    pass an explicit `tools=` to simulate a tool failure instead.
    `classifier`/`router`/`critic` (ADR-0019/0023) default `None` —
    disabled, same "existing caller unaffected" contract every
    `GraphContext` field already has."""
    graph = build_graph()
    if tools is None:
        tools = tools_from_articles(articles)
    context = GraphContext(
        session=None,
        embeddings=None,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )
    return asyncio.run(graph.ainvoke({"question": question}, context=context))


def _run_stream(
    llm,
    question: str = "What is a high-risk AI system?",
    *,
    articles: list[RetrievedChunk] = ARTICLES,
    tools: dict | None = None,
    classifier=None,
    router=None,
    critic=None,
) -> list[str]:
    """Same as `_run` but drives `graph.astream(..., stream_mode='updates')`
    and returns just the ordered list of node names visited — the async
    equivalent of the old sync `graph.stream(...)` loop these tests used
    before `retrieve_node` became `async def`. A paused run's `__interrupt__`
    chunk (ADR-0025, verified live: a distinct `{"__interrupt__": (...)}`
    chunk, never folded into a node's own update) shows up as the literal
    string `"__interrupt__"` in the returned list, same as every real node
    name — `hitl_node` itself never appears as its own entry on the run
    that pauses (execution halts INSIDE it, before it returns an update);
    it only appears (once, alongside `guard_out`) on the RESUME call that
    follows (`_resume_turn_stream` below)."""
    graph = build_graph()
    if tools is None:
        tools = tools_from_articles(articles)
    context = GraphContext(
        session=None,
        embeddings=None,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools,
    )

    async def _collect():
        return [
            list(update)[0]
            async for update in graph.astream(
                {"question": question}, context=context, stream_mode="updates"
            )
        ]

    return asyncio.run(_collect())


# --- ADR-0024: durable state — checkpointer, thread_id, history --------
# `InMemorySaver` (langgraph.checkpoint.memory) — no Postgres needed for
# these: `build_graph(checkpointer=...)` doesn't care WHICH saver
# implementation it gets (that's the whole point of the checkpointer
# abstraction), so an in-process saver proves the graph-side wiring
# without a real DB.


def _run_turn(
    graph,
    llm,
    question: str,
    *,
    thread_id: str,
    articles: list[RetrievedChunk] = ARTICLES,
    classifier=None,
    router=None,
    critic=None,
):
    """One turn of a checkpointed conversation against an already-compiled
    `graph` (built with a checkpointer) — same shape as `_run` above, but
    takes the compiled `graph` and a `thread_id` instead of building a
    fresh (uncheckpointed) graph itself. Sharing `thread_id` across calls
    is what makes state persist/accumulate (ADR-0024); a fresh `GraphContext`
    every call mirrors how a real caller (api.py, cli.py) builds one per
    request — `GraphContext` itself is never persisted, only `GraphState`."""
    context = GraphContext(
        session=None,
        embeddings=None,
        llm=llm,
        classifier=classifier,
        router=router,
        critic=critic,
        tools=tools_from_articles(articles),
    )
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(graph.ainvoke({"question": question}, context=context, config=config))


def _run_turn_stream(
    graph,
    llm,
    question: str,
    *,
    thread_id: str,
    articles: list[RetrievedChunk] = ARTICLES,
    critic=None,
) -> list[str]:
    """`_run_turn`'s stream sibling (lesson 21) — same checkpointed-turn
    shape, but returns the ordered node-name list like `_run_stream`,
    needed to pin exactly where a trajectory PAUSES (the last entry is
    `"__interrupt__"`, not `"hitl"` — see `_run_stream`'s docstring)."""
    context = GraphContext(
        session=None, embeddings=None, llm=llm, critic=critic, tools=tools_from_articles(articles)
    )
    config = {"configurable": {"thread_id": thread_id}}

    async def _collect():
        return [
            list(update)[0]
            async for update in graph.astream(
                {"question": question}, context=context, config=config, stream_mode="updates"
            )
        ]

    return asyncio.run(_collect())


def _resume_turn(
    graph, decision: str, edited_answer: str | None = None, *, thread_id: str, llm=None
):
    """Resumes a paused run via `Command(resume=...)` — same shape
    `api.py`'s `_stream_resume`/`cli.py`'s `resume` command build. `llm`
    defaults to `_UnusedLLM()`: nothing between `hitl` and `guard_out`
    should ever call it, so a default that raises on any `.invoke()` is a
    stronger default than a double that just returns something unused."""
    context = GraphContext(
        session=None,
        embeddings=None,
        llm=llm or _UnusedLLM(),
        tools=tools_from_articles(ARTICLES),
    )
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(
        graph.ainvoke(
            Command(resume={"decision": decision, "edited_answer": edited_answer}),
            context=context,
            config=config,
        )
    )


def _resume_turn_stream(
    graph, decision: str, edited_answer: str | None = None, *, thread_id: str, llm=None
) -> list[str]:
    """`_resume_turn`'s stream sibling (lesson 21) — returns the ordered
    node-name list for the RESUME half of a pause+resume trajectory
    (`["hitl", "guard_out"]` on approve, verified live — see
    `_run_stream`'s docstring)."""
    context = GraphContext(
        session=None, embeddings=None, llm=llm or _UnusedLLM(), tools=tools_from_articles(ARTICLES)
    )
    config = {"configurable": {"thread_id": thread_id}}

    async def _collect():
        return [
            list(update)[0]
            async for update in graph.astream(
                Command(resume={"decision": decision, "edited_answer": edited_answer}),
                context=context,
                config=config,
                stream_mode="updates",
            )
        ]

    return asyncio.run(_collect())
