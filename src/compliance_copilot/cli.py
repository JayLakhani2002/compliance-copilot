# src/compliance_copilot/cli.py — small operator-facing entry point
# (docs/ARCHITECTURE.md's "operator / Jay" actor runs ingestion and admin
# commands from here). Today: one command, `init-db`, which creates the
# schema against whatever DATABASE_URL points at. `python -m
# compliance_copilot.cli init-db`.
import argparse

from compliance_copilot.db import engine, init_db


def main() -> None:
    parser = argparse.ArgumentParser(prog="compliance_copilot")
    # argparse subcommands over a bigger framework (click/typer): stdlib
    # covers "one required positional command name" fine, and a real CLI
    # library isn't worth adding for a single command.
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create the vector extension, tables, and HNSW index.")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db(engine)
        print("Database initialised.")


if __name__ == "__main__":
    main()
