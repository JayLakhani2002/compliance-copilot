# src/compliance_copilot/graph/__init__.py — package surface for the
# LangGraph graph (docs/ARCHITECTURE.md §4). Callers (cli.py, tests) import
# from `compliance_copilot.graph` rather than reaching into the submodules
# directly, so the retrieve/answer/state split can change without breaking
# callers.
from compliance_copilot.graph.build import ask, build_graph, make_mcp_tools
from compliance_copilot.graph.nodes import REFUSAL_TEXT
from compliance_copilot.graph.state import (
    AnswerSchema,
    Citation,
    CitationError,
    GraphContext,
    ToolCallError,
)
from compliance_copilot.guards.output import OutputGuardError

__all__ = [
    "REFUSAL_TEXT",
    "AnswerSchema",
    "Citation",
    "CitationError",
    "GraphContext",
    "OutputGuardError",
    "ToolCallError",
    "ask",
    "build_graph",
    "make_mcp_tools",
]
