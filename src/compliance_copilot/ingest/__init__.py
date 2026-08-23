# src/compliance_copilot/ingest/__init__.py — the "ingestion" package
# (docs/ARCHITECTURE.md §1, ADR-0012/ADR-0004): eurlex.py (fetch + parse
# EUR-Lex XHTML into article/recital chunks) -> chunker.py (split oversize
# articles into embeddable-sized parts) -> pipeline.py (embed + upsert into
# Postgres — the only module here with DB/network side effects beyond
# eurlex.py's own on-disk XHTML cache).
