# src/compliance_copilot/costing.py — turns per-model token usage into a
# USD/EUR cost estimate (ADR-0029, lesson 24). Sits next to settings.py:
# pure bookkeeping over numbers a LangChain callback already collects
# (`langchain_core.callbacks.UsageMetadataCallbackHandler` — every chat
# model response already carries token counts, so this is zero-network
# arithmetic, not an extra API call). Used by evals/run_cost_report.py.
#
# Not wired into the live request path (SSE final event / Langfuse score)
# today — that's a natural next step (mirroring how tracing.score() already
# attaches citation_valid to a trace) but out of scope for this feature,
# named as future work in ADR-0029 rather than half-built here.
from __future__ import annotations

import re

from compliance_copilot.settings import settings

# Price per 1,000,000 tokens ("MTok"), USD. `cached_input` is the
# discounted rate a provider charges when input tokens hit its automatic
# prompt-prefix cache — `None` where a model has no such rate (embeddings
# have no prefix-caching concept; Anthropic's cache needs an explicit
# `cache_control` block this app doesn't set, ADR-0002). `output` is `None`
# for embeddings — that endpoint has no output tokens at all.
#
# Every row is dated — this is a point-in-time snapshot, not a live price
# feed (same "won't silently drift and lie" discipline ADR-0002's own
# pricing notes already use). Re-verify before trusting an old row for a
# real budget decision.
PRICES: dict[str, dict[str, float | None]] = {
    # Verified developers.openai.com/api/docs/pricing, 2026-08-30.
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    # Verified developers.openai.com/api/docs/pricing, 2026-08-30. ADR-0004
    # doesn't itself record a figure for this model — pulled from the same
    # pricing page as the two rows above, same verification date.
    "text-embedding-3-small": {"input": 0.02, "output": None, "cached_input": None},
    # ALTERNATIVE path only — ADR-0002's documented Anthropic/Bedrock
    # production target, not what's shipped today (ADR-0002's 2026-08-24
    # amendment: OpenAI is the interim provider). Per ADR-0002, recorded
    # 2026-08-23 — reverify before the Bedrock path ships.
    "claude-sonnet-5": {"input": 3.00, "output": 15.00, "cached_input": None},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cached_input": None},
}


# OpenAI's response_metadata reports the resolved dated SNAPSHOT id (e.g.
# "gpt-4.1-mini-2025-04-14"), not the bare alias `make_llm()` requested
# ("gpt-4.1-mini") — observed live, 2026-09-02, running this module's own
# report. Both ids bill at the same published rate (the alias just always
# points at the latest snapshot), so stripping a trailing `-YYYY-MM-DD`
# before the `PRICES` lookup is a correctness fix, not a guess: without it
# every real run would `KeyError` on the alias's own resolved name.
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _price_key(model_name: str) -> str:
    if model_name in PRICES:
        return model_name
    stripped = _SNAPSHOT_SUFFIX.sub("", model_name)
    return stripped if stripped in PRICES else model_name


def estimate_cost(usage_by_model: dict[str, dict]) -> dict:
    """`usage_by_model`: the shape `UsageMetadataCallbackHandler.usage_
    metadata` produces — `{model_name: UsageMetadata}`, where `UsageMetadata`
    is (at minimum) `{"input_tokens": int, "output_tokens": int,
    "input_token_details": {"cache_read": int, ...}}` (verified against
    installed `langchain_core.messages.ai`, 2026-08-30). A plain dict works
    fine here too — this function only reads keys via `.get`, never assumes
    the langchain_core TypedDict class itself.

    Returns `{"usd_total", "eur_total", "by_model": {model_name: {"usd",
    "input_tokens", "output_tokens", "cached_tokens"}}}` — keyed by the
    ORIGINAL `model_name` from `usage_by_model`, even when `_price_key()`
    matched it via the snapshot-suffix fallback, so the report still shows
    exactly which snapshot answered.

    Raises `KeyError` (naming the model) on a model with no `PRICES` entry
    (after the snapshot-suffix fallback) — fail loud: a silently-priced-at-$0
    model would under-report a real bill, which is worse than a crash during
    a report run."""
    by_model: dict[str, dict] = {}
    usd_total = 0.0
    for model_name, usage in usage_by_model.items():
        price_key = _price_key(model_name)
        if price_key not in PRICES:
            raise KeyError(
                f"No price entry for model {model_name!r} — add it to costing.PRICES first"
            )
        price = PRICES[price_key]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0) or 0
        # Cached tokens bill at the discounted rate, the rest of input at
        # standard rate. min() guards a malformed usage dict (cached >
        # total) from producing a negative "uncached" count.
        cached_tokens = min(cached_tokens, input_tokens)
        uncached_tokens = input_tokens - cached_tokens
        cached_rate = price["cached_input"] if price["cached_input"] is not None else price["input"]
        usd = (
            (uncached_tokens / 1_000_000) * price["input"]
            + (cached_tokens / 1_000_000) * cached_rate
            + (output_tokens / 1_000_000) * (price["output"] or 0.0)
        )
        by_model[model_name] = {
            "usd": usd,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
        }
        usd_total += usd
    return {
        "usd_total": usd_total,
        "eur_total": usd_total * settings.eur_usd_rate,
        "by_model": by_model,
    }
