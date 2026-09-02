# evals/run_cost_report.py — measures actual per-question LLM/embedding
# cost from a real run of the golden set (ADR-0029, lesson 24). Reuses
# evals/run_answer_eval.py's graph-running pieces (GoldenAnswer,
# load_golden_answers) rather than re-implementing them — same graph, same
# golden set, a different question ("what did this cost", not "was this
# answer good").
#
# Runs the FULL production call shape per question — guard_in's classifier,
# the router, the answer call, the critic (whichever of those
# `settings.*_enabled` turns on) — not just retrieve->answer, since that's
# what a real `/ask` request actually pays for. `get_usage_metadata_
# callback()` (langchain_core, verified in installed source) pools usage by
# MODEL NAME automatically: classifier+router+critic all share
# `gpt-4.1-nano` today, so their token counts land in one pooled bucket —
# reported honestly as "nano (classifier+router+critic, pooled)" below,
# not split per node, because the callback has no node-identity to split on.
#
# Embeddings are the one call this can't observe via a callback: `retrieve_
# node` calls the `search_regulation` MCP tool, and the actual
# `embed_query()` call happens INSIDE the MCP server subprocess
# (mcp_server.py, a separate process talking stdio) — no LangChain callback
# in THIS process ever sees it. Instead we count the query's tokens locally
# with tiktoken (same BPE OpenAI's embeddings endpoint uses) and price that
# count — a deterministic, zero-network substitute for "the real call's
# usage", not a guess about token count.
#
# Costs real money — same "run it locally, occasionally" caveat run_answer_
# eval.py already carries:
#     set -a; source .env; set +a
#     uv run python -m evals.run_cost_report
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.costing import _SNAPSHOT_SUFFIX, estimate_cost
from compliance_copilot.critic import make_critic_llm
from compliance_copilot.db import get_engine
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import CitationError, make_mcp_tools
from compliance_copilot.graph import ask as ask_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.guards.classifier import make_classifier_llm
from compliance_copilot.router import make_router_llm
from compliance_copilot.settings import settings
from evals.run_answer_eval import GoldenAnswer, load_golden_answers

# cl100k_base: the BPE every current OpenAI embedding/chat model uses
# (tiktoken has no separate per-embedding-model encoding table) — used only
# to COUNT tokens locally, never to call the API, per the module docstring.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# ponytail: a wall-clock backstop this SCRIPT owns, not app config — observed
# live (2026-09-02) that the MCP stdio subprocess can wedge completely after
# many `search_regulation`/`get_article` round-trips (zero CPU, no further
# log lines, no `mcp_tool timeout` — i.e. stuck somewhere `retrieve_node`'s
# own per-call `asyncio.wait_for(mcp_tool_timeout_s)` doesn't cover), which
# would otherwise hang this whole 10-question report forever. `tools` is
# rebuilt (a fresh subprocess) after a timeout so the STUCK pipe can't take
# out every question after it. Upgrade path: if this fires often, the real
# fix belongs in `retrieve_node`/`make_mcp_tools()` (a request-wide budget
# around the whole tool-fetch loop, not per-call only) — out of scope here.
_QUESTION_TIMEOUT_S = 90.0


@dataclass
class QuestionCost:
    id: str
    usd: float
    eur: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    embedding_tokens: int
    degraded: bool  # a refusal or a stuck/timed-out run — near-zero marginal cost, noted not zeroed
    # Per-CHAT-model breakdown (embeddings excluded — tracked separately via
    # `embedding_tokens` since it's counted locally, not observed via a
    # callback). Keyed by model name (e.g. "gpt-4.1-mini-2025-04-14") —
    # `aggregate()` sums this across questions so the report can show
    # answer-tier (mini) vs. guard-tier (nano, pooled classifier+router+
    # critic) cost separately, per the spec's "distinguishable by model" ask.
    by_model: dict[str, dict] = field(default_factory=dict)


def _embedding_tokens(question: str) -> int:
    return len(_ENCODING.encode(question))


async def _run_one(
    golden: GoldenAnswer,
    *,
    session: Session,
    embeddings,
    llm,
    classifier,
    router,
    critic,
    tools,
) -> tuple[QuestionCost, bool]:
    config = tracing.run_config(tags=["cost-report", f"golden:{golden.id}"])
    # ADR-0024's checkpointer needs a `thread_id` per run (verified live:
    # LangGraph raises `ValueError` without one the moment ANY checkpointer
    # is passed) — one question = one throwaway thread, never reused, since
    # each golden question here is independent (no multi-turn history).
    config["configurable"] = {"thread_id": f"cost-report-{golden.id}"}
    embed_tokens = _embedding_tokens(golden.question)
    degraded = False
    # `get_usage_metadata_callback()` registers a context-local hook
    # (verified installed source: `register_configure_hook(...,
    # inheritable=True)`) that every chat-model call made anywhere inside
    # this `with` block reports into — no need to thread it through
    # `config["callbacks"]` by hand.
    timed_out = False
    with get_usage_metadata_callback() as cb:
        try:
            await asyncio.wait_for(
                ask_graph(
                    golden.question,
                    session=session,
                    embeddings=embeddings,
                    llm=llm,
                    classifier=classifier,
                    router=router,
                    critic=critic,
                    tools=tools,
                    config=config,
                    # ADR-0025: hitl_node's interrupt() requires a
                    # checkpointer to exist at all — InMemorySaver is enough
                    # to satisfy that (a paused draft still has its `answer`
                    # key set, since answer_node ran before hitl_node, so
                    # ask_graph() returns normally either way; no
                    # cross-question state needed).
                    checkpointer=InMemorySaver(),
                ),
                timeout=_QUESTION_TIMEOUT_S,
            )
        except CitationError:
            # ADR-0014's hard-refusal path: whatever calls happened before
            # the refusal are still in `cb.usage_metadata` — reported as-is,
            # just flagged `degraded` rather than silently zeroed.
            degraded = True
        except TimeoutError:
            # See `_QUESTION_TIMEOUT_S`'s comment — a stuck run, not a normal
            # refusal. Whatever usage happened before the stall is still
            # reported; the caller respawns `tools` before the next question.
            degraded = True
            timed_out = True

    usage = dict(cb.usage_metadata)
    usage[settings.embedding_model] = {"input_tokens": embed_tokens, "output_tokens": 0}
    result = estimate_cost(usage)

    chat_models = {k: v for k, v in result["by_model"].items() if k != settings.embedding_model}
    return (
        QuestionCost(
            id=golden.id,
            usd=result["usd_total"],
            eur=result["eur_total"],
            input_tokens=sum(v["input_tokens"] for v in chat_models.values()),
            output_tokens=sum(v["output_tokens"] for v in chat_models.values()),
            cached_tokens=sum(v["cached_tokens"] for v in chat_models.values()),
            embedding_tokens=embed_tokens,
            degraded=degraded,
            by_model=chat_models,
        ),
        timed_out,
    )


async def _run_goldens(goldens: list[GoldenAnswer]) -> list[QuestionCost]:
    embeddings = get_embeddings()
    llm = make_llm()
    classifier = make_classifier_llm() if settings.classifier_enabled else None
    router = make_router_llm() if settings.router_enabled else None
    critic = make_critic_llm() if settings.critic_enabled else None
    # Reviewer SHOULD 1 (ADR-0029): the spawn itself can wedge the same way a
    # tool call can — bound it too.
    tools = await asyncio.wait_for(make_mcp_tools(), timeout=60)

    costs: list[QuestionCost] = []
    with Session(get_engine()) as session:
        for golden in goldens:
            cost, timed_out = await _run_one(
                golden,
                session=session,
                embeddings=embeddings,
                llm=llm,
                classifier=classifier,
                router=router,
                critic=critic,
                tools=tools,
            )
            costs.append(cost)
            print(
                f"... {golden.id} done (eur={cost.eur:.5f}, degraded={cost.degraded})", flush=True
            )
            if timed_out:
                # See `_QUESTION_TIMEOUT_S`'s comment — the stdio pipe to the
                # old subprocess may be wedged; a fresh one is cheap (a local
                # process spawn) next to the LLM calls it's guarding.
                print("... respawning MCP subprocess after a stuck question", flush=True)
                tools = await asyncio.wait_for(make_mcp_tools(), timeout=60)
    return costs


# Display-only grouping: OpenAI reports a dated snapshot id
# ("gpt-4.1-mini-2025-04-14") on each response (see costing.py's own
# `_price_key` comment) — grouped here under the bare alias so the
# per-model breakdown reads as "mini" / "nano", not one row per snapshot.


def aggregate(costs: list[QuestionCost]) -> dict:
    n = len(costs)
    if n == 0:
        return {
            "n": 0,
            "eur_per_question": 0.0,
            "eur_per_100": 0.0,
            "cached_fraction": 0.0,
            "avg_embedding_tokens": 0.0,
            "n_degraded": 0,
            "by_model": {},
        }
    total_input = sum(c.input_tokens for c in costs)
    total_cached = sum(c.cached_tokens for c in costs)
    total_eur = sum(c.eur for c in costs)

    # Per-model totals across every question — this is the "nano
    # (classifier+router+critic) pooled vs. mini (answer)" breakdown the
    # spec asks for: every guard-tier call shares the nano model id, so
    # summing by model name pools them automatically.
    by_model: dict[str, dict] = {}
    for c in costs:
        for model_name, usage in c.by_model.items():
            key = _SNAPSHOT_SUFFIX.sub("", model_name)
            row = by_model.setdefault(
                key, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
            )
            row["usd"] += usage["usd"]
            row["input_tokens"] += usage["input_tokens"]
            row["output_tokens"] += usage["output_tokens"]
            row["cached_tokens"] += usage["cached_tokens"]

    return {
        "n": n,
        "eur_per_question": total_eur / n,
        "eur_per_100": (total_eur / n) * 100,
        "cached_fraction": (total_cached / total_input) if total_input else 0.0,
        "avg_embedding_tokens": sum(c.embedding_tokens for c in costs) / n,
        "n_degraded": sum(c.degraded for c in costs),
        "by_model": by_model,
    }


def print_report(costs: list[QuestionCost], aggregates: dict) -> None:
    print("\n=== cost report (measured, real run) ===\n")
    print(
        f"{'id':<6} {'usd':>8} {'eur':>8} {'in_tok':>8} {'out_tok':>8} "
        f"{'cached':>8} {'embed_tok':>10} note"
    )
    for c in costs:
        note = "degraded (refused/timed-out)" if c.degraded else ""
        print(
            f"{c.id:<6} {c.usd:>8.5f} {c.eur:>8.5f} {c.input_tokens:>8} "
            f"{c.output_tokens:>8} {c.cached_tokens:>8} {c.embedding_tokens:>10} {note}"
        )
    print("\n--- summary ---")
    print(
        f"n={aggregates['n']}  "
        f"€/question={aggregates['eur_per_question']:.5f}  "
        f"€/100 questions={aggregates['eur_per_100']:.3f}  "
        f"cached_fraction={aggregates['cached_fraction']:.3f}  "
        f"avg_embedding_tokens={aggregates['avg_embedding_tokens']:.1f}  "
        f"n_degraded={aggregates['n_degraded']}"
    )
    print("\n--- by model (summed over all questions) ---")
    for model_name, row in aggregates["by_model"].items():
        cached_frac = row["cached_tokens"] / row["input_tokens"] if row["input_tokens"] else 0.0
        print(
            f"{model_name:<20} usd={row['usd']:.5f}  in_tok={row['input_tokens']:<8} "
            f"out_tok={row['output_tokens']:<8} cached_frac={cached_frac:.3f}"
        )
    print(
        "\nnote: gpt-4.1-nano pools classifier+router+critic — they share "
        "one model id, and get_usage_metadata_callback() has no per-node "
        "identity to split them by."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=None, help="Dump the full report to this path."
    )
    args = parser.parse_args()

    goldens = load_golden_answers()
    costs = asyncio.run(_run_goldens(goldens))
    aggregates = aggregate(costs)
    print_report(costs, aggregates)

    if args.json:
        args.json.write_text(
            json.dumps({"costs": [asdict(c) for c in costs], "aggregates": aggregates}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
