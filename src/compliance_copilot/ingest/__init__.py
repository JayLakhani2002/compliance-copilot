# src/compliance_copilot/ingest/__init__.py — the "ingestion" package
# (docs/ARCHITECTURE.md §1, ADR-0012). Today it holds
# only eurlex.py (fetch + parse EUR-Lex XHTML into article/recital chunks,
# ADR-0012); the embedding + DB-write step lands as a separate Day 4 module,
# not here, so this package stays pure/no-side-effects-beyond-the-cache.
