# Lesson 02 — One database: Postgres + pgvector (Week 1, Day 2)

**What it is.** Postgres is a relational database (tables, rows, SQL). `pgvector` is an extension that adds a `vector` column type plus distance operators (`<=>` cosine, `<->` L2) and indexes (HNSW, IVFFlat) so you can store embeddings next to your normal data and ask "which rows are closest to this vector?" in plain SQL.

**Why we need it here.** Our agent retrieves legal articles by meaning. Each article chunk gets an embedding (a list of ~1536 floats); a question gets one too; nearest neighbours = relevant articles. We also need ordinary tables: documents, chunks with metadata (regulation, article number), LangGraph checkpoints, eval results. pgvector lets all of that live in one database with one backup, one connection string, one set of transactions. ADR-0003.

**Why not the alternative.** Qdrant / Weaviate / Pinecone are dedicated vector databases. They win at very large scale (100M+ vectors) and have richer filtering DSLs. Cost: a second stateful service to run, secure, back up and keep in sync with Postgres (the classic "two sources of truth" bug: you delete a document in Postgres but its vectors stay searchable). Our corpus is ~1–2k chunks; pgvector's HNSW index answers in milliseconds. Pinecone can't be self-hosted in the EU at all. So: one database until measurements say otherwise — and we keep the retriever behind an interface so swapping is a one-file change.

**How a senior thinks about it.**
- *Failure modes:* DB down → API must return a clear 503, not hang. Missing index → queries go from 5 ms to seconds at scale (seq scan). Wrong distance metric vs. how the embedding model was trained → garbage neighbours (OpenAI embeddings are normalised → cosine).
- *Security:* DB is inside the trust boundary; never reachable from the internet; credentials only via env; app user has least privilege (no superuser in prod).
- *Cost:* a 4 GB VPS runs Postgres + our API fine. Storage: 1536 floats × 4 bytes ≈ 6 KB per chunk → 2k chunks ≈ 12 MB. Trivial.
- *Ops:* migrations are code (repeatable), not hand-typed SQL in a console. Docker Compose gives dev/prod parity.

**Analogy.** A library that keeps both the card catalogue (metadata) and a "books that feel similar to this one" shelf in the same building. A separate vector DB is a second building across town: faster for a city-sized collection, but now two caretakers must agree on which books exist.

**Interview question.** "When would you move from pgvector to a dedicated vector database, and how would you know it's time?" (Answer shape: measure p95 query latency and recall at your real N; > ~10–50M vectors, heavy filtered search, or multi-tenant isolation needs; you'd know from the retrieval eval + latency dashboard, not from a blog post.)

**Check yourself.** 1) Why cosine and not L2 here? 2) What bug does "one database" prevent?

> Note: the HNSW index is created already in Day 2's `db.py` (the schema is the natural place for it); Day 4's lesson explains *why* HNSW and what `m` / `ef_construction` trade off. Don't worry about those two numbers yet.
