# src/compliance_copilot/cli.py — small operator-facing entry point
# (docs/ARCHITECTURE.md's "operator / Jay" actor runs ingestion and admin
# commands from here). Commands: `init-db` (creates the schema against
# whatever DATABASE_URL points at; `--reset` drops and recreates it — dev
# only, see db.py's init_db docstring), `ingest` (fetch+parse+embed+upsert a
# regulation; `--dry-run` fetches/parses/embeds but rolls back instead of
# committing), and `search` (embed a question, print the top-k nearest
# chunks by cosine distance — the first retrieval smoke test, ADR-0004).
# `python -m compliance_copilot.cli init-db --reset` / `... ingest
# --regulation all` / `... search "What is a high-risk AI system?"` /
# `... ask "What is a high-risk AI system?"`.
import argparse
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot import tracing
from compliance_copilot.db import Chunk, get_engine, init_db
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.graph import REFUSAL_TEXT, CitationError
from compliance_copilot.graph import ask as ask_graph
from compliance_copilot.graph.nodes import make_llm
from compliance_copilot.guards.classifier import make_classifier_llm
from compliance_copilot.ingest.eurlex import REGULATIONS
from compliance_copilot.ingest.pipeline import ingest
from compliance_copilot.settings import settings


def main() -> None:
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
        embeddings = get_embeddings()
        llm = make_llm()
        # ADR-0019: `None` when CLASSIFIER_ENABLED=false — same "disabled
        # means skip it" contract `ask_graph()`/`guard_in_node` already give
        # a `None` classifier, so this is a one-line off switch here too.
        classifier = make_classifier_llm() if settings.classifier_enabled else None
        # ADR-0009 amendment: a no-op config (empty callbacks list) when no
        # Langfuse keys are set — `tracing.run_config()` is a fresh function
        # call per invocation, no different from calling `ask_graph` with no
        # config= at all in that case.
        config = tracing.run_config()
        with Session(get_engine()) as session:
            try:
                result = ask_graph(
                    args.question,
                    session=session,
                    embeddings=embeddings,
                    llm=llm,
                    classifier=classifier,
                    config=config,
                )
            except CitationError as exc:
                # Never print a half-validated answer alongside a refusal —
                # the answer text and citations are simply not printed at
                # all here (ADR-0014's hard-error path).
                print(f"REFUSED: {exc}", file=sys.stderr)
                sys.exit(2)
            finally:
                # Flush (not shutdown): the CLI process exits right after
                # this command anyway, but a blocking flush is what
                # guarantees this one trace is actually sent before that
                # happens (tracing.py) — a no-op when tracing is disabled.
                tracing.flush()
            # REFUSAL_TEXT is a fixed, module-level string (graph/nodes.py) —
            # comparing against it is how the CLI tells "the input guard
            # refused this" apart from "the model answered normally with no
            # citations" (ask()'s signature stays `-> AnswerSchema` only, so
            # there's no separate `refused` flag to check here, ADR-0018).
            if result.answer == REFUSAL_TEXT:
                print(f"REFUSED (input guard): {REFUSAL_TEXT}", file=sys.stderr)
                sys.exit(3)
            print(result.answer)
            for citation in result.citations:
                print(f"  [{citation.regulation} {citation.anchor}] {citation.quote!r}")
            trace_id = tracing.current_trace_id(config)
            if trace_id is not None:
                print(f"trace: {trace_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
