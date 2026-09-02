# src/compliance_copilot/guards/quotes.py — verbatim-quote normalisation,
# shared by the graph's own citation guard (graph/nodes.py's
# `_validate_citations`, ADR-0014) and the MCP server's `cite` tool
# (mcp_server.py, ADR-0007). Moved out of graph/nodes.py (Day-16 review nit,
# ADR-0007's Day-17 amendment): `mcp_server.py` used to import
# `_normalise`/`_MIN_QUOTE_LENGTH` straight from `graph.nodes`, which drags
# `langchain_anthropic`/`langchain_openai` into the MCP server process at
# import time (a working-but-wasteful side effect, not a bug) — this module
# has no LLM-client imports at all, so either caller can import it cheaply.
from __future__ import annotations

import html
import re

# The corpus's own text uses Unicode curly quotes (U+2018/U+2019, U+201C/
# U+201D) around every defined term (chunker.py's definition-boundary regex
# matches '‘' for the same reason) — the model may reproduce a quote with
# straight ASCII quotes instead. Map both to straight so that difference
# alone never causes a false-positive citation rejection (reviewer round 1).
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

# Minimum normalised-quote length: an empty or 1-2 character quote is a
# substring of almost any text, so it would pass the verbatim check with
# zero actual information content (reviewer round 1) — reject it outright
# rather than let it through as a "cited" claim.
_MIN_QUOTE_LENGTH = 20


def _normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded, curly-quotes-to-straight —
    applied to both a model's quote and the source chunk text before the
    substring check, so a quote that only differs by spacing, letter case,
    or quote-mark style still counts as verbatim (ADR-0014).

    `html.unescape` runs first: `graph/nodes.py`'s `_render_chunk`
    HTML-escapes chunk text before showing it to the model (`&`/`<`/`>` ->
    `&amp;`/`&lt;`/`&gt;`), and the system prompt tells the model to copy
    its citation quote "verbatim... from the cited excerpt's text" — i.e.
    from what it was shown, escaped form included. Unescaping both sides
    here before comparing against the raw DB text is what makes a
    genuinely-verbatim quote containing one of those characters still pass
    (round-1 review finding: previously only the escape direction was
    implemented, not the reverse, so this was a live false-positive-
    `CitationError` bug for any chunk containing `&`, `<`, or `>`)."""
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip().casefold().translate(_QUOTE_MAP)
