# src/compliance_copilot/cli.py — small operator-facing entry point
# (docs/ARCHITECTURE.md's "operator / Jay" actor runs ingestion and admin
# commands from here). Commands: `init-db` (creates the schema against
# whatever DATABASE_URL points at; `--reset` drops and recreates it — dev
# only, see db.py's init_db docstring), `ingest` (fetch+parse+embed+upsert a
# regulation; `--dry-run` fetches/parses/embeds but rolls back instead of
# committing), and `search` (embed a question, print the top-k nearest
# chunks by cosine distance — the first retrieval smoke test, ADR-0004).
# `python -m compliance_copilot.cli init-db --reset` / `... ingest
# --regulation all` / `... search "What is a high-risk AI system?"`.
import argparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from compliance_copilot.db import Chunk, get_engine, init_db
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.ingest.eurlex import REGULATIONS
from compliance_copilot.ingest.pipeline import ingest


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

    args = parser.parse_args()

    if args.command == "init-db":
        init_db(get_engine(), reset=args.reset)
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


if __name__ == "__main__":
    main()
