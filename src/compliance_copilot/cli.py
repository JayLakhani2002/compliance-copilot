# src/compliance_copilot/cli.py — small operator-facing entry point
# (docs/ARCHITECTURE.md's "operator / Jay" actor runs ingestion and admin
# commands from here). Commands: `init-db` (creates the schema against
# whatever DATABASE_URL points at) and `ingest` (fetch+parse a regulation —
# `--dry-run` is the only mode today since embeddings/DB-writes land Day 4).
# `python -m compliance_copilot.cli init-db` / `... ingest --regulation all
# --dry-run`.
import argparse

from compliance_copilot.db import get_engine, init_db
from compliance_copilot.ingest.eurlex import REGULATIONS, ingest_regulation


def main() -> None:
    parser = argparse.ArgumentParser(prog="compliance_copilot")
    # argparse subcommands over a bigger framework (click/typer): stdlib
    # covers "a handful of subcommands with a couple of flags" fine, and a
    # real CLI library isn't worth adding yet.
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create the vector extension, tables, and HNSW index.")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Fetch + parse a regulation into article/recital chunks."
    )
    ingest_parser.add_argument(
        "--regulation",
        choices=[*REGULATIONS.keys(), "all"],
        required=True,
        help="Which regulation to ingest.",
    )
    # Required (not just accepted) for now: this command has no other mode
    # yet — DB writes/embeddings are Day 4 — so requiring the flag makes
    # that explicit at the call site instead of silently no-op-ing.
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Fetch+parse and print counts only — the only mode implemented so far.",
    )

    args = parser.parse_args()

    if args.command == "init-db":
        init_db(get_engine())
        print("Database initialised.")
    elif args.command == "ingest":
        keys = list(REGULATIONS) if args.regulation == "all" else [args.regulation]
        for key in keys:
            chunks = ingest_regulation(key)
            articles = [c for c in chunks if c.kind == "article"]
            recitals = [c for c in chunks if c.kind == "recital"]
            print(f"{key}: {len(articles)} articles, {len(recitals)} recitals")
            article_1 = next((c for c in articles if c.number == 1), None)
            if article_1:
                print(f"  Article 1 preview: {article_1.text[:200]!r}")


if __name__ == "__main__":
    main()
