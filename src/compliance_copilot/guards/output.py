# src/compliance_copilot/guards/output.py — `guard_out`: the final,
# independent output-side gate (docs/ARCHITECTURE.md §4, ADR-0021) that runs
# immediately before END on EVERY path through the graph — a good answer, a
# `guard_in` refusal, and an exhausted-retry citation failure all funnel
# through here. Zero LLM calls: every check below is deterministic
# string/schema logic (substring/regex/length), the same "cheap, fast,
# testable" posture the input-side heuristic layer already set
# (guards/injection.py).
#
# Why this exists when `answer_node` already validates citations
# (`_validate_citations`, graph/nodes.py, ADR-0014): that check validates the
# answer node's OWN retrieval/model call. Defence in depth means `guard_out`
# doesn't trust that self-check either — it's an independent final
# checkpoint that also catches classes of failure citation validation was
# never built to look for: prompt/scaffold leakage, a Day-13 redaction
# placeholder echoed back into prose, a leaked canary token, and off-corpus
# scope (a well-cited answer about the wrong subject entirely).
#
# `CANARY` is defined here (not graph/nodes.py, despite the task naming that
# module) so this module never has to import graph/nodes.py at runtime —
# `nodes.py` imports `CANARY` (plus `check_output`/`OutputGuardError`) FROM
# here and embeds it into `SYSTEM_PROMPT`, so `nodes.CANARY` still resolves
# for any caller that expects it "in nodes.py". Keeping the one real
# definition here, where the leak-check lives, avoids a state.py <-> nodes.py
# <-> guards.output import cycle for no benefit.
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `from __future__ import annotations` (above) means this hint is never
    # evaluated at runtime, so this import never actually executes — it
    # exists for type checkers/IDEs only. That's what keeps this module
    # import-cycle-free: graph/state.py imports `OutputVerdict` from HERE,
    # so this module must not import graph.state for real.
    from compliance_copilot.graph.state import AnswerSchema

# rebuff-style canary (protectai/rebuff, Apache-2.0, already cited
# ADR-0018/ADR-0019's classifier prompt precedent): a short constant embedded
# in the system prompt with an explicit "never repeat this" instruction;
# `check_output` scans the answer for an exact-string match as a leak signal.
# Fixed, not per-run-random — rebuff's own production default is a random
# canary per request, but a fixed one is the right simplification for a
# diffable test fixture here: nothing about this app's canary is meant to
# survive an attacker who already knows the constant, it only proves "did
# the system prompt leak."
CANARY = "ccp-canary-7f3a9e"


@dataclass(frozen=True)
class OutputVerdict:
    """Mirrors guards/injection.py's `GuardResult` shape: always safe to log
    or put in an SSE event — `reason` is a short code (or `None`), never the
    answer text itself."""

    ok: bool
    reason: str | None


class OutputGuardError(ValueError):
    """Raised only when `guard_out_node` (graph/nodes.py) decides a check
    failure is an INVARIANT break, not a policy violation — see
    `check_output`'s docstring for the full pass/refuse/raise split. Message
    is the reason code only (e.g. "citation_not_retrieved"), never the
    answer text — same "never echo content" rule `CitationError` already
    follows (graph/state.py)."""


# Scaffold substrings a legitimate answer should never contain — the model
# echoing its own prompt scaffolding (an XML tag it was shown, never meant
# for the user) back into user-visible prose. Near-zero false-positive risk:
# none of these ever appear in ordinary legal-answer prose.
_SCAFFOLD_SUBSTRINGS = (
    "<excerpt",
    "</excerpt>",
    "<question>",
    "</question>",
    "<supporting_context",
    "<user_text",
)

# Day-13 redaction placeholder tokens (guards/pii.py's `_OPERATORS` map) — a
# redacted span must never round-trip back out into the answer's prose.
# Refusals are exempt from this check (see `check_output` below): a refusal
# is `REFUSAL_TEXT` verbatim, which never contains one anyway.
_PLACEHOLDER_RE = re.compile(r"<\s*(?:person|email|phone|iban|credit_card|ip|pii)\s*>")

# ponytail: the ceiling this heuristic accepts — a genuinely long "the
# excerpts don't cover this, here's why" reply with zero citations would
# also trip this, since raw length alone can't tell "confidently answered
# off-corpus" apart from "explained at length why it can't answer." 400 is a
# starting point (red-team research), not asserted as correct — tune against
# real red-team/benign data, upgrade path is a smarter scope classifier.
_SCOPE_LENGTH_THRESHOLD = 400
# A long zero-citation answer is allowed when it is visibly a "the excerpts
# don't cover this" reply — SYSTEM_PROMPT instructs exactly that shape, and
# review showed a realistic 401-char honest non-answer being refused on
# length alone. ponytail: keyword allowlist, tune with red-team data.
_NON_ANSWER_MARKERS = ("not cover", "do not", "does not", "cannot", "excerpt", "not answer")
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff]")


def _norm(text: str) -> str:
    """Casefold + HTML-unescape + zero-width strip before every substring
    check — review bypassed the canary with plain upper-casing and the
    scaffold/placeholder checks with `&lt;excerpt&gt;`/`<person>`."""
    return _ZERO_WIDTH_RE.sub("", html.unescape(text)).casefold()


def check_output(
    answer: AnswerSchema,
    *,
    retrieved_keys: set[tuple[str, str]] | None,
    refused: bool,
) -> OutputVerdict:
    """Deterministic output-side checks, in priority order — returns the
    FIRST failing reason, never a full checklist (a pass/fail gate, not a
    summed score like guards/injection.py's `GuardResult`, since any one of
    these failing is enough on its own to block).

    `retrieved_keys`: the `(regulation, anchor)` pairs `retrieve_node`
    actually fetched this run, or `None` to skip the `citation_not_retrieved`
    check entirely (there was no retrieval — e.g. a `guard_in` refusal).
    `refused`: `True` only for the fixed refusal shape (`refuse_node`'s
    output, or a citation-retry exhaustion) — a narrower check set applies,
    since a refusal is `REFUSAL_TEXT` verbatim, zero citations, by
    construction.

    The caller (`guard_out_node`, graph/nodes.py) turns this into one of
    three outcomes: `ok=True` -> pass through unchanged; `ok=False` with a
    POLICY reason (`empty_answer`, `canary_leak`, `scaffold_leak`,
    `placeholder_leak`, `scope_unsupported`) -> rewrite to the fixed
    refusal, guard blocks, never repairs (ADR-0014's rule, reused here);
    `ok=False` with `citation_not_retrieved`, OR any failure at all when
    `refused=True` -> the caller raises `OutputGuardError` instead —
    something upstream is buggy (a citation `answer_node` claims it already
    validated, or a refusal whose fixed text somehow fails its own checks),
    not a policy call to quietly refuse a second time."""
    text = answer.answer
    norm = _norm(text)

    if len(text.strip()) < 3:
        return OutputVerdict(ok=False, reason="empty_answer")

    if CANARY in norm or any(CANARY in _norm(c.quote) for c in answer.citations):
        return OutputVerdict(ok=False, reason="canary_leak")

    if any(marker in norm for marker in _SCAFFOLD_SUBSTRINGS):
        return OutputVerdict(ok=False, reason="scaffold_leak")

    if refused:
        # A refusal is REFUSAL_TEXT verbatim, zero citations, by
        # construction — nothing past this point applies to it. Any of the
        # three checks above tripping HERE means REFUSAL_TEXT itself (or
        # whatever produced this "refusal") is broken, which is exactly why
        # guard_out_node escalates ANY failure on a refused answer to a hard
        # error rather than treating it as a policy call.
        return OutputVerdict(ok=True, reason=None)

    if _PLACEHOLDER_RE.search(norm):
        return OutputVerdict(ok=False, reason="placeholder_leak")

    if retrieved_keys is not None:
        for citation in answer.citations:
            if (citation.regulation, citation.anchor) not in retrieved_keys:
                return OutputVerdict(ok=False, reason="citation_not_retrieved")

    if (
        not answer.citations
        and len(text) > _SCOPE_LENGTH_THRESHOLD
        and not any(m in norm for m in _NON_ANSWER_MARKERS)
    ):
        return OutputVerdict(ok=False, reason="scope_unsupported")

    return OutputVerdict(ok=True, reason=None)
