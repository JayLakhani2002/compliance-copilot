# src/compliance_copilot/cli.py — small operator-facing entry point
# (docs/ARCHITECTURE.md's "operator / Jay" actor runs ingestion and admin
# commands from here). Commands: `init-db` (creates the schema against
# whatever DATABASE_URL points at; `--reset` drops and recreates it — dev
# only, see db.py's init_db docstring), `ingest` (fetch+parse+embed+upsert a
# regulation; `--dry-run` fetches/parses/embeds but rolls back instead of
# committing), `search` (embed a question, print the top-k nearest chunks by
# cosine distance — the first retrieval smoke test, ADR-0004), `ask` (runs
# the graph for one question, ADR-0024's `--thread-id` continues a prior
# conversation; prints an "under review" line instead of an answer when
# `hitl_node` pauses it, ADR-0025), `delete-thread` (ADR-0024's GDPR-
# flavoured erasure path — drops every checkpoint for one thread_id), and
# `resume` (ADR-0025: continues a paused run with an approve/edit/reject
# decision).
# `python -m compliance_copilot.cli init-db --reset` / `... ingest
# --regulation all` / `... search "What is a high-risk AI system?"` /
# `... ask "What is a high-risk AI system?"` / `... ask "..." --thread-id
# <uuid>` / `... delete-thread <uuid>` / `... resume <uuid> --decision
# approve` / `... resume <uuid> --decision edit --answer "..."`.
#
# Exit code table (`ask`/`resume`, ADR-0025 round 2 SHOULD 3 — each number
# means exactly one thing, never reused across commands for a different
# condition):
#   2  ask         REFUSED — CitationError, retries exhausted (ADR-0014)
#   3  ask/resume  REFUSED — input or output guard blocked (ADR-0018/0021)
#   4  ask/resume  INTERNAL — OutputGuardError, an invariant broke (ADR-0021)
#   5  ask         INTERNAL — ToolCallError, an MCP call failed (ADR-0007)
#   6  ask         PAUSED — hitl_node just interrupted THIS run (ADR-0025)
#   7  resume      INTERNAL — thread_id is unknown (no checkpointed state)
#   8  resume      INTERNAL — thread_id exists but has nothing pending to
#                  resume (not currently paused)
#   9  ask         INTERNAL — thread_id (an EXISTING one, --thread-id) is
#                  ALREADY paused from an earlier call — call resume
#                  instead of silently superseding the pending review
import argparse
import asyncio
import sys
import uuid

from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.checkpointer import build_checkpointer, validate_thread_id
from compliance_copilot.critic import make_critic_llm
from compliance_copilot.db import Chunk, get_engine, init_db
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import (
    REFUSAL_TEXT,
    CitationError,
    GraphContext,
    OutputGuardError,
    ToolCallError,
    make_mcp_tools,
)
from compliance_copilot.graph.build import build_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.guards.classifier import make_classifier_llm
from compliance_copilot.ingest.eurlex import REGULATIONS
from compliance_copilot.ingest.pipeline import ingest
from compliance_copilot.logging_filter import install_pii_scrub
from compliance_copilot.router import make_router_llm
from compliance_copilot.settings import settings


async def _run_ask(question: str, thread_id: str | None = None) -> None:
    """The `ask` command's body (ADR-0007 Day-17 amendment): `async def`,
    driving `graph.ainvoke(...)` — a real installed-`langgraph` smoke test
    confirmed sync `graph.invoke()` raises `TypeError: No synchronous
    function provided` the moment a run reaches `retrieve_node` (now
    `async def`, since it awaits an MCP tool call), so `main()` below wraps
    this one call in `asyncio.run(...)` rather than keeping a sync path
    that no longer works.

    ADR-0024: `thread_id` is `None` on a first-turn CLI call — the server
    mints one (`uuid.uuid4()`) and prints it to stderr, the same "learn the
    id you'll need to continue" contract `api.py`'s `thread` SSE event
    gives an HTTP client; `main()`'s `--thread-id` flag is what a second
    invocation passes back in to continue that conversation.
    `build_checkpointer()` opens/closes its pool around this ONE call — a
    CLI invocation is a fresh OS process every time, unlike the FastAPI
    app's one long-lived pool (api.py's `lifespan`)."""
    embeddings = get_embeddings()
    llm = make_llm()
    # ADR-0019: `None` when CLASSIFIER_ENABLED=false — same "disabled
    # means skip it" contract `guard_in_node` already gives a `None`
    # classifier, so this is a one-line off switch here too.
    classifier = make_classifier_llm() if settings.classifier_enabled else None
    # ADR-0023: same "disabled means skip it" contract as `classifier` above.
    router = make_router_llm() if settings.router_enabled else None
    critic = make_critic_llm() if settings.critic_enabled else None
    # ADR-0007 Day-17 amendment: spawns the MCP server subprocess and loads
    # its tools once for this command invocation (`None` when
    # `settings.mcp_enabled=False` — never a silent fallback to direct
    # retrieval, see `make_mcp_tools`'s docstring).
    tools = await make_mcp_tools()
    resolved_thread_id = thread_id or str(uuid.uuid4())
    # Printed immediately, before the graph even runs — same "the caller
    # learns thread_id in time to reuse it, regardless of how this turn
    # turns out" contract api.py's `thread` SSE event gives (that event is
    # also the very first thing emitted, ADR-0024).
    print(f"thread: {resolved_thread_id}", file=sys.stderr)
    # ADR-0009 amendment: a no-op config (empty callbacks list) when no
    # Langfuse keys are set — `tracing.run_config()` is a fresh function
    # call per invocation, no different from calling `graph.invoke` with
    # no config= at all in that case. `configurable.thread_id` (ADR-0024)
    # rides alongside it, same shape api.py's `_stream_answer` builds.
    config = tracing.run_config()
    config["configurable"] = {"thread_id": resolved_thread_id}
    async with build_checkpointer() as checkpointer, Session(get_engine()) as session:
        # ADR-0020: calls the compiled graph directly (not the `ask()`
        # convenience wrapper) so this command can read `pii_entities`
        # off the final state — `ask()` deliberately keeps returning
        # just `AnswerSchema` for its other callers (tests,
        # test_graph_real_integration.py), so widening its signature
        # for this one extra field isn't worth it (ponytail).
        graph = build_graph(checkpointer=checkpointer)
        if thread_id is not None:
            # ADR-0025 round 2 (BLOCKER 2): only a caller-SUPPLIED
            # `--thread-id` can possibly already be paused (a freshly
            # minted one has no prior state) — reproduced live before this
            # fix: `graph.ainvoke({"question": ...}, config=same_thread_id)`
            # on an already-paused thread is NOT rejected by LangGraph, it
            # happily starts a new run from START and OVERWRITES the
            # paused checkpoint (the original draft/critic verdict/
            # interrupt vanish with no error) — the same bug `api.py`'s
            # `_reject_if_paused` fixes for `/ask`.
            snapshot = await graph.aget_state(config)
            if snapshot.next and snapshot.interrupts:
                print(
                    f"INTERNAL: thread {thread_id} is awaiting review — run "
                    f"'resume {thread_id} --decision approve|edit|reject' instead of ask",
                    file=sys.stderr,
                )
                sys.exit(9)
        context = GraphContext(
            session=session,
            embeddings=embeddings,
            llm=llm,
            classifier=classifier,
            router=router,
            critic=critic,
            tools=tools,
        )
        try:
            state = await graph.ainvoke({"question": question}, context=context, config=config)
        except CitationError as exc:
            # Never print a half-validated answer alongside a refusal —
            # the answer text and citations are simply not printed at
            # all here (ADR-0014's hard-error path).
            print(f"REFUSED: {exc}", file=sys.stderr)
            sys.exit(2)
        except OutputGuardError as exc:
            # ADR-0021: `guard_out` found an INVARIANT broken (e.g. a
            # citation `answer_node` claims it already validated), not a
            # policy violation — this is a bug report, not a refusal,
            # so it gets its own exit code and an "INTERNAL:" prefix
            # rather than looking like an ordinary REFUSED response.
            print(f"INTERNAL: output guard invariant failed ({exc})", file=sys.stderr)
            sys.exit(4)
        except ToolCallError as exc:
            # ADR-0007 Day-17 amendment: an MCP transport failure/timeout/
            # malformed result — an infra failure, not a policy decision,
            # so it gets the same "INTERNAL:" treatment as an output-guard
            # invariant break, on its own exit code.
            print(f"INTERNAL: retrieval tool call failed ({exc})", file=sys.stderr)
            sys.exit(5)
        finally:
            # Flush (not shutdown): the CLI process exits right after
            # this command anyway, but a blocking flush is what
            # guarantees this one trace is actually sent before that
            # happens (tracing.py) — a no-op when tracing is disabled.
            tracing.flush()
        # ADR-0025: `hitl_node` paused this run — `graph.ainvoke` returns
        # (rather than raises) with a top-level `__interrupt__` key
        # (installed `langgraph`'s documented pause shape) instead of
        # `state["answer"]` being the SETTLED answer. No answer/citations
        # print here — there ISN'T a final one yet, only a draft awaiting a
        # decision — print a status line and the exact command to resume it.
        interrupts = state.get("__interrupt__")
        if interrupts:
            draft = interrupts[0].value
            print(
                f"under review (thread_id {resolved_thread_id}): "
                f"critic confidence {draft['confidence']:.2f} below threshold. "
                f"Resume with: python -m compliance_copilot.cli resume "
                f"{resolved_thread_id} --decision approve|edit|reject",
                file=sys.stderr,
            )
            sys.exit(6)
        result = state["answer"]
        # Entity TYPE names only (guards/pii.py's `redact()`) — never
        # the redacted values, same "never echo the payload" rule the
        # SSE `pii` field (api.py) already follows.
        pii_entities = state.get("pii_entities") or ()
        if pii_entities:
            print(f"note: PII redacted ({', '.join(pii_entities)})", file=sys.stderr)
        # ADR-0028: `answer_node`'s answer-model-outage fallback — the
        # printed `result.answer` text below already says "Service
        # degraded" plainly, but a stderr note (same channel the PII/thread
        # lines above use) makes it visible even if a caller only greps
        # stdout for the answer body and skips reading it closely.
        if state.get("degraded"):
            print(
                "note: service degraded — answer model unavailable, showing "
                "retrieved articles only",
                file=sys.stderr,
            )
        # REFUSAL_TEXT is a fixed, module-level string (graph/nodes.py) —
        # comparing against it is how the CLI tells "the input guard
        # refused this" apart from "the model answered normally with no
        # citations" (there's no separate `refused` flag on
        # `AnswerSchema` itself, ADR-0018).
        if result.answer == REFUSAL_TEXT:
            # ADR-0021: `guard_out` can ALSO produce this exact text (a
            # policy-violation rewrite, e.g. a leaked canary). When
            # `guard_in` was the layer that refused, `guard_out` then
            # just passes that fixed refusal through clean, so
            # `output_guard.ok` is True — `not output_guard.ok` here
            # can therefore only mean `guard_out` itself did the
            # rewriting, never `guard_in`.
            output_guard = state.get("output_guard")
            if output_guard is not None and not output_guard.ok:
                print(
                    f"REFUSED (output guard: {output_guard.reason}): {REFUSAL_TEXT}",
                    file=sys.stderr,
                )
            else:
                print(f"REFUSED (input guard): {REFUSAL_TEXT}", file=sys.stderr)
            sys.exit(3)
        print(result.answer)
        for citation in result.citations:
            print(f"  [{citation.regulation} {citation.anchor}] {citation.quote!r}")
        trace_id = tracing.current_trace_id(config)
        if trace_id is not None:
            print(f"trace: {trace_id}", file=sys.stderr)


async def _run_resume(thread_id: str, decision: str, edited_answer: str | None) -> None:
    """The `resume` command's body (ADR-0025): continues a run `_run_ask`
    reported as "under review" — same dependency setup as `_run_ask` (a
    resumed run can still reach `guard_out`, which is deterministic, but
    the graph is rebuilt identically regardless of which nodes a given
    resume actually touches).

    Validates BEFORE resuming (same `snapshot.values`/`.next`/`.interrupts`
    check `api.py`'s `_require_paused_thread` uses) — an unknown or
    not-currently-paused `thread_id` is a clean, actionable error message
    here, not a confusing `Command(resume=...)` failure three calls deep.
    Distinct exit codes for the two conditions (ADR-0025 round 2, SHOULD 3
    — see this module's header table): 7 unknown, 8 not paused.

    ADR-0025 round 2 (SHOULD 1): prints the operator-facing draft answer,
    critic confidence, and reasoning — read straight off `snapshot.
    interrupts[0].value` — BEFORE applying the decision. This is the
    channel the full payload is actually meant for (the CLI is the
    operator's own terminal, `docs/ARCHITECTURE.md`'s "operator / Jay"
    actor); `/ask`'s HTTP `interrupt` SSE event deliberately does NOT carry
    this (SHOULD 1's other half — the end user sees "under review" only).
    No separate `--interrupt-id` flag: read straight off THIS freshly-
    fetched snapshot rather than asking the operator to have copied one
    down earlier — the simplest option that still guarantees the decision
    is applied to whatever is ACTUALLY pending right now, not a stale
    belief about what's pending."""
    embeddings = get_embeddings()
    llm = make_llm()
    classifier = make_classifier_llm() if settings.classifier_enabled else None
    router = make_router_llm() if settings.router_enabled else None
    critic = make_critic_llm() if settings.critic_enabled else None
    tools = await make_mcp_tools()
    config = tracing.run_config()
    config["configurable"] = {"thread_id": thread_id}
    async with build_checkpointer() as checkpointer, Session(get_engine()) as session:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            print(f"INTERNAL: unknown thread_id {thread_id}", file=sys.stderr)
            sys.exit(7)
        if not snapshot.next or not snapshot.interrupts:
            print(f"INTERNAL: thread {thread_id} is not currently paused", file=sys.stderr)
            sys.exit(8)
        pending = snapshot.interrupts[0]
        draft = pending.value
        print(f"interrupt_id: {pending.id}", file=sys.stderr)
        print(f"question: {draft['question']}", file=sys.stderr)
        print(f"draft answer: {draft['answer']['answer']}", file=sys.stderr)
        print(f"critic confidence: {draft['confidence']:.2f}", file=sys.stderr)
        print(f"critic reasoning: {draft['reasoning']}", file=sys.stderr)
        context = GraphContext(
            session=session,
            embeddings=embeddings,
            llm=llm,
            classifier=classifier,
            router=router,
            critic=critic,
            tools=tools,
        )
        try:
            state = await graph.ainvoke(
                Command(resume={"decision": decision, "edited_answer": edited_answer}),
                context=context,
                config=config,
            )
        except CitationError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            sys.exit(2)
        except OutputGuardError as exc:
            print(f"INTERNAL: output guard invariant failed ({exc})", file=sys.stderr)
            sys.exit(4)
        finally:
            tracing.flush()
    result = state["answer"]
    if result.answer == REFUSAL_TEXT:
        output_guard = state.get("output_guard")
        if output_guard is not None and not output_guard.ok:
            print(f"REFUSED (output guard: {output_guard.reason}): {REFUSAL_TEXT}", file=sys.stderr)
        else:
            print(f"REFUSED: {REFUSAL_TEXT}", file=sys.stderr)
        sys.exit(3)
    print(result.answer)
    for citation in result.citations:
        print(f"  [{citation.regulation} {citation.anchor}] {citation.quote!r}")
    trace_id = tracing.current_trace_id(config)
    if trace_id is not None:
        print(f"trace: {trace_id}", file=sys.stderr)


def _valid_thread_id(value: str) -> str:
    """`argparse` `type=` callable for `--thread-id`/`delete-thread`'s
    positional arg — raises `argparse.ArgumentTypeError` (argparse's own
    convention for a bad `type=` conversion, prints a clean usage error
    instead of a raw traceback) on anything `validate_thread_id`
    (checkpointer.py) rejects, so both CLI entry points and `api.py`'s
    `AskRequest` enforce the exact same UUID4 rule from one place."""
    try:
        return validate_thread_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


async def _run_delete_thread(thread_id: str) -> None:
    """The `delete-thread` command's body (ADR-0024's GDPR-flavoured
    erasure path) — drops every checkpoint row for `thread_id`. Opens its
    own checkpointer for this one call, same "fresh process, fresh pool"
    reasoning as `_run_ask` above. `adelete_thread` is idempotent-in-effect
    (deleting an already-empty/unknown thread just deletes zero rows, per
    the installed `AsyncPostgresSaver.adelete_thread`'s own `DELETE ...
    WHERE thread_id = ...` implementation) — no separate "does this thread
    exist" check needed before calling it."""
    async with build_checkpointer() as checkpointer:
        await checkpointer.adelete_thread(thread_id)
    print(f"deleted all checkpoints for thread {thread_id}")


def main() -> None:
    # ADR-0020: logging backstop (defence-in-depth only, see
    # logging_filter.py's module docstring) — installed before any command
    # below can log anything.
    install_pii_scrub()
    parser = argparse.ArgumentParser(prog="compliance_copilot")
    # argparse subcommands over a bigger framework (click/typer): stdlib
    # covers "a handful of subcommands with a couple of flags" fine, and a
    # real CLI library isn't worth adding yet.
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser(
        "init-db", help="Create the vector extension, tables, and HNSW index."
    )
    init_db_parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP document/chunk first, then recreate empty. Dev-only — never run "
        "against a database anyone else's data is in.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest", help="Fetch + parse + embed a regulation, upserting it into the DB."
    )
    ingest_parser.add_argument(
        "--regulation",
        choices=[*REGULATIONS.keys(), "all"],
        required=True,
        help="Which regulation to ingest.",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch+parse+embed but roll back instead of committing — prints stats only.",
    )

    search_parser = subparsers.add_parser(
        "search", help="Embed a question and print the top-k nearest chunks."
    )
    search_parser.add_argument("question")
    search_parser.add_argument("--k", type=int, default=5)

    ask_parser = subparsers.add_parser(
        "ask", help="Run the retrieve -> answer graph and print a cited answer."
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--thread-id",
        type=_valid_thread_id,
        default=None,
        help="ADR-0024: continue a prior conversation (a UUID4 this command already printed "
        "to stderr on an earlier call). Omit to start a new one — the server mints one.",
    )

    delete_thread_parser = subparsers.add_parser(
        "delete-thread",
        help="ADR-0024: erase all checkpointed state for a thread_id (GDPR erasure path).",
    )
    delete_thread_parser.add_argument("thread_id", type=_valid_thread_id)

    resume_parser = subparsers.add_parser(
        "resume",
        help="ADR-0025: resume a run paused by hitl_node with an approve/edit/reject decision.",
    )
    resume_parser.add_argument("thread_id", type=_valid_thread_id)
    resume_parser.add_argument("--decision", choices=["approve", "edit", "reject"], required=True)
    resume_parser.add_argument(
        "--answer",
        dest="edited_answer",
        default=None,
        help="Replacement answer text — required (and only allowed) when --decision edit. "
        "Still has to pass guard_out (ADR-0025): never trusted as already-safe just "
        "because a human wrote it.",
    )

    args = parser.parse_args()

    if args.command == "init-db":
        engine = get_engine()
        if args.reset:
            # Loud confirmation before the destructive call — db.py's
            # init_db(reset=True) guard would refuse this against a
            # non-"_test" DB without force=True, so this print is the
            # operator's one chance to notice they're about to drop
            # `document`/`chunk` on the DB named below.
            print(
                f"--reset: dropping+recreating document/chunk on database {engine.url.database!r}"
            )
            init_db(engine, reset=True, force=True)
        else:
            init_db(engine)
        print("Database initialised.")
    elif args.command == "ingest":
        keys = list(REGULATIONS) if args.regulation == "all" else [args.regulation]
        embeddings = get_embeddings()
        with Session(get_engine()) as session:
            for key in keys:
                stats = ingest(key, embeddings, session, dry_run=args.dry_run)
                print(
                    f"{key}: {stats.chunks_total} chunks "
                    f"({stats.chunks_embedded} embedded, {stats.chunks_skipped} skipped)"
                    + (" [dry-run, not committed]" if args.dry_run else "")
                )
    elif args.command == "search":
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(args.question)
        distance = Chunk.embedding.cosine_distance(query_vector)
        with Session(get_engine()) as session:
            # Select the distance alongside the row (not just order by it) —
            # the smoke test is exactly "is the top result close", so the
            # number has to be visible, not just the ranking.
            results = session.execute(
                select(Chunk, distance.label("distance")).order_by(distance).limit(args.k)
            ).all()
            for chunk, dist in results:
                preview = chunk.text[:120].replace("\n", " ")
                print(
                    f"{chunk.document.regulation} {chunk.anchor_id} "
                    f"{chunk.title or ''!r} dist={dist:.4f} — {preview!r}"
                )
    elif args.command == "ask":
        # ADR-0007 Day-17 amendment: `_run_ask` is `async def` (it awaits
        # both the MCP tool loading and `graph.ainvoke(...)`) — `asyncio.run`
        # is the CLI's one entrypoint into the event loop, same pattern
        # `evals/run_redteam.py`/`evals/run_answer_eval.py`'s `main()`
        # functions use for the same reason.
        asyncio.run(_run_ask(args.question, args.thread_id))
    elif args.command == "delete-thread":
        asyncio.run(_run_delete_thread(args.thread_id))
    elif args.command == "resume":
        if args.decision == "edit" and not args.edited_answer:
            parser.error("--answer is required when --decision edit")
        if args.decision != "edit" and args.edited_answer is not None:
            parser.error("--answer is only allowed when --decision edit")
        asyncio.run(_run_resume(args.thread_id, args.decision, args.edited_answer))


if __name__ == "__main__":
    main()
