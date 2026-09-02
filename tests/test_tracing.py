# tests/test_tracing.py — unit tests for tracing.py (ADR-0009 amendment).
# No network, no real Langfuse account (Jay hasn't created one yet — see
# tests/test_tracing_real_integration.py for the gated real-account test).
# Two things are verified here: (a) every function is a true no-op when
# LANGFUSE_* keys are absent, including never importing `langfuse` itself,
# and (b) LangGraph actually propagates `config={"callbacks": [...]}` down
# into node execution — the mechanism `tracing.run_config()`'s callbacks
# rely on, verified with a plain spy handler and no Langfuse involved at all.
from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest
from fake_mcp_tools import tools_from_articles
from langchain_core.callbacks import BaseCallbackHandler

from compliance_copilot import tracing
from compliance_copilot.graph import AnswerSchema, ask
from compliance_copilot.retriever import RetrievedChunk

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTICLES = [
    RetrievedChunk(
        anchor="art_6",
        regulation="ai_act",
        kind="article",
        number=6,
        title="Classification rules",
        text="An AI system shall be considered high-risk where it is a safety component.",
        distance=0.1,
        part=0,
    ),
]


class FakeLLM:
    """Same minimal double as test_graph.py/test_api.py — duplicated rather
    than imported (three lines, no other reason for this file to depend on
    those modules)."""

    def __init__(self, response: AnswerSchema):
        self._response = response

    def invoke(self, messages):
        return self._response


@pytest.fixture(autouse=True)
def _no_langfuse_env(monkeypatch):
    # The "disabled by default" contract this whole feature rests on: no
    # account exists yet, and this must be true in CI too.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


# --- (a) fully disabled when keys are absent --------------------------------
def test_tracing_enabled_false_when_keys_absent():
    assert tracing.tracing_enabled() is False


def test_get_callbacks_empty_when_disabled():
    assert tracing.get_callbacks() == []


def test_run_config_has_empty_callbacks_when_disabled():
    config = tracing.run_config()
    assert config["callbacks"] == []
    assert "metadata" in config  # tags still set — harmless with no callbacks to read them


def test_current_trace_id_none_when_disabled():
    config = tracing.run_config()
    assert tracing.current_trace_id(config) is None


def test_flush_and_score_are_noops_when_disabled():
    tracing.flush()  # must not raise
    tracing.shutdown()  # must not raise
    tracing.score("citation_valid", 1.0, trace_id=None)  # must not raise
    tracing.score("citation_valid", 1.0, trace_id="some-trace-id")  # still a no-op: disabled


def test_importing_tracing_never_imports_langfuse():
    """Subprocess, not a sys.modules check in-process: pytest's own
    collection may import `langfuse` via some other test file first (e.g.
    the gated real-integration test module), which would make an in-process
    `"langfuse" not in sys.modules` check order-dependent. A fresh
    interpreter that only imports `compliance_copilot.tracing` and calls the
    disabled-path functions is the robust way to prove this module's own
    import graph never pulls in langfuse/OpenTelemetry (tracing.py's module
    docstring: that's the whole point of the lazy import)."""
    script = (
        "import sys\n"
        "import compliance_copilot.tracing as tracing\n"
        "assert 'langfuse' not in sys.modules\n"
        "tracing.get_callbacks()\n"  # disabled path: still must not import it
        "assert 'langfuse' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},  # no LANGFUSE_* leaks in from the parent env
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# --- (b) callback pass-through through the real compiled graph -------------
class SpyHandler(BaseCallbackHandler):
    """Records which LangGraph node each chain run belongs to — installed
    `langchain_core`'s `on_chain_start` gets a `metadata` dict carrying
    `langgraph_node` for every node-level run (verified empirically against
    this repo's installed langgraph 1.2.11: `metadata.get('langgraph_node')`
    is exactly 'retrieve'/'answer' for this graph's two nodes)."""

    def __init__(self):
        self.node_names: list[str] = []

    def on_chain_start(self, serialized, inputs, *, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node is not None:
            self.node_names.append(node)


def test_config_callbacks_propagate_to_graph_node_runs():
    answer = AnswerSchema(answer="...", citations=[])
    spy = SpyHandler()

    asyncio.run(
        ask(
            "What is a high-risk AI system?",
            session=None,
            embeddings=None,
            llm=FakeLLM(answer),
            tools=tools_from_articles(ARTICLES),
            config={"callbacks": [spy]},
        )
    )

    assert "retrieve" in spy.node_names
    assert "answer" in spy.node_names
