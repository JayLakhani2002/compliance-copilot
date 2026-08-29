# tests/fake_mcp_tools.py — a stand-in for the LangChain `BaseTool` objects
# `retrieve_node` (graph/nodes.py, ADR-0007's Day-17 amendment) calls via
# `runtime.context.tools`, used by every test that needs `search_regulation`/
# `get_article` tools but must not spawn a real MCP server subprocess — same
# "fake double, no real dependency" reasoning as tests/fake_embeddings.py.
#
# Mirrors the exact calling convention `_call_tool` (graph/nodes.py) uses:
# `.ainvoke({"type": "tool_call", "name": ..., "args": ..., "id": ...})`
# returning a `ToolMessage`-shaped object with `.status` and
# `.artifact["structured_content"]` — verified live against a real
# MCP-backed tool (ADR-0007's Day-17 amendment: a plain-dict `.ainvoke()`
# discards the artifact, only the `ToolCall`-shaped input builds one), not
# guessed.
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage

from compliance_copilot.retriever import RetrievedChunk


class ToolExecutionError(Exception):
    """Raise this from a `FakeMCPTool`'s `fn` to simulate the real tool's
    own `status="error"` result (e.g. a bad anchor's `ValueError`) — a
    business-logic failure, never retried by `_call_tool`. Simulate a
    TRANSPORT failure instead by raising any other exception (e.g.
    `ConnectionError`), which `FakeMCPTool.ainvoke` lets propagate
    unchanged."""


class FakeMCPTool:
    """A minimal double for one MCP-backed LangChain tool: `fn(args) ->
    Any` returns the tool's structured result directly (a dict/list, same
    shape `search_regulation`/`get_article` return), wrapped here into the
    `ToolMessage` shape `_call_tool` (graph/nodes.py) expects. Records every
    call's `args` (never wrapped/hidden) so tests can assert what
    `retrieve_node` actually asked for."""

    def __init__(self, fn: Callable[[dict], Any], name: str = "fake_tool") -> None:
        self.name = name
        self._fn = fn
        self.calls: list[dict] = []

    async def ainvoke(self, call: dict) -> ToolMessage:
        self.calls.append(call["args"])
        try:
            result = self._fn(call["args"])
        except ToolExecutionError as exc:
            return ToolMessage(
                content=str(exc),
                artifact=None,
                tool_call_id=call["id"],
                name=self.name,
                status="error",
            )
        return ToolMessage(
            content="ok",
            artifact={"structured_content": result},
            tool_call_id=call["id"],
            name=self.name,
            status="success",
        )


def tools_from_articles(articles: list[RetrievedChunk]) -> dict[str, FakeMCPTool]:
    """Builds `search_regulation`/`get_article` fake tools whose combined
    behaviour reproduces `articles` — mirrors `retrieve_node`'s real
    two-hop lookup (rank via `search_regulation`, then fetch each unique
    anchor's full joined text via `get_article`), so a test can hand this
    the same `RetrievedChunk` fixtures the old direct-`retrieve()` tests
    used and get the same `state["articles"]` shape back out."""
    parts_by_key: dict[tuple[str, str], list[RetrievedChunk]] = {}
    order: list[tuple[str, str]] = []
    for a in articles:
        key = (a.regulation, a.anchor)
        if key not in parts_by_key:
            order.append(key)
        parts_by_key.setdefault(key, []).append(a)

    def _search(args: dict) -> list[dict]:
        return [
            {
                "regulation": reg,
                "anchor": anchor,
                "title": parts_by_key[(reg, anchor)][0].title,
                "snippet": parts_by_key[(reg, anchor)][0].text[:300],
                "distance": parts_by_key[(reg, anchor)][0].distance,
                "part": parts_by_key[(reg, anchor)][0].part,
            }
            for reg, anchor in order
        ]

    def _get_article(args: dict) -> dict:
        parts = parts_by_key[(args["regulation"], args["anchor"])]
        return {
            "regulation": args["regulation"],
            "anchor": args["anchor"],
            "title": parts[0].title,
            "text": " ".join(p.text for p in parts),
            "part_count": len(parts),
        }

    return {
        "search_regulation": FakeMCPTool(_search, name="search_regulation"),
        "get_article": FakeMCPTool(_get_article, name="get_article"),
    }
