# src/compliance_copilot/ingest/__init__.py — the "ingestion" package
# (docs/ARCHITECTURE.md §1, Day 3/4 of docs/CURRICULUM.md). Today it holds
# only eurlex.py (fetch + parse EUR-Lex XHTML into article/recital chunks,
# ADR-0012); the embedding + DB-write step lands as a separate Day 4 module,
# not here, so this package stays pure/no-side-effects-beyond-the-cache.
