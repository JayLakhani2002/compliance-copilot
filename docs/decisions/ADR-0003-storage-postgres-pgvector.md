# ADR-0003: Storage — PostgreSQL 16 + pgvector

## Status
Accepted. 2026-08-23.

## Context
The system needs to store four kinds of application data: (1) the ingested regulation documents and their metadata, (2) text chunks and their embedding vectors (for similarity search — pgvector is a Postgres extension that adds a vector data type and nearest-neighbor search operators, i.e. it makes Postgres double as a vector database), (3) LangGraph's checkpoints (durable graph state, ADR-0001), and (4) evaluation results (ADR-0005). Market research shows `pgvector` named directly in 6% of ads and `vector-db` (generic) in 28%, alongside Weaviate (12%), Pinecone (10%), Qdrant (4%) — vector databases are expected, but no single dedicated vector-DB product dominates the market signal enough to justify running it as a second stateful system for a 6-week solo build.

## Options considered
1. **PostgreSQL 16 + pgvector, as the single database for all four data kinds above.**
2. **Qdrant / Weaviate / Chroma** (dedicated open-source vector databases) — purpose-built for vector search, often faster/more feature-rich for pure similarity search at scale, but each is a second stateful service to run, back up, and secure, on top of Postgres (which the project needs anyway for LangGraph checkpoints — LangGraph's checkpointer doesn't run on a vector DB).
3. **Pinecone** (managed vector DB) — removes the ops burden entirely, but is a US-based managed SaaS with no EU-self-hostable option, which directly conflicts with the project's EU-residency goal (`docs/ARCHITECTURE.md` §8).

## Decision
**PostgreSQL 16 with the pgvector extension**, used as the single database for documents, chunks+embeddings, LangGraph checkpoints, and eval results. The vector store is accessed through **LangChain's `VectorStore` interface** (concretely, `langchain-postgres`'s `PGVectorStore`/`PGEngine`, ADR-0004) so the retriever is behind an interface — reversible to a different backend without touching graph/node code, per the project's stated goal of keeping this choice swappable.

**Verified API detail:** `langchain-postgres` (PyPI, version 0.0.17, verified 2026-08-23) has moved its recommended API to `PGEngine` (a connection-pool wrapper, created via `PGEngine.from_connection_string(...)` or `PGEngine.from_engine(...)`) + `PGVectorStore` (created via `PGVectorStore.create(engine=..., table_name=..., embedding_service=...)`), both **async-first**. The package's own examples include a migration notebook (`migrate_pgvector_to_pgvectorstore.ipynb`), confirming the older `PGVector` class is the **legacy** API and `PGVectorStore`/`PGEngine` is current — build against `PGVectorStore`/`PGEngine`, not the older class, to avoid building on a path the library itself is migrating users off of.

## Why not the others
- **Qdrant / Weaviate / Chroma**: rejected to avoid running a second stateful database purely for vectors when Postgres is already required for checkpoints and eval results — one fewer system to back up, monitor, and secure matters more for a solo 6-week build than the retrieval-speed edge a dedicated vector DB might offer at a scale this project won't reach.
- **Pinecone**: rejected specifically because it has no EU-self-hostable deployment option, which is a hard conflict with the EU-residency goal that is a named differentiator for this project's target job market (GDPR shows up as a requirement in German-language ads specifically).

## Security & cost implications
- **Security:** Postgres holds the most sensitive data in the system — checkpointed graph state may include (pre-redaction, if `guard_in` has a bug) user-submitted text, and the DB is behind the trust boundary described in `docs/ARCHITECTURE.md` §6 (internal Docker network only, no public exposure; least-privilege DB roles per service — `api`/`mcp-server` should not share a superuser role).
- **Cost:** no extra product/license cost (Postgres + pgvector are both open source); the cost is entirely the VPS resources it consumes (RAM for query performance, disk for embedding storage — embedding vectors for two regulations' worth of articles/recitals is a small dataset, well within a modest VPS's disk budget).
- **Important scope clarification** (surfaced while verifying ADR-0009/Langfuse): "the one database" is accurate for *this project's own application data*. It is **not** accurate for the full Docker Compose stack once self-hosted Langfuse is added — Langfuse's self-host deployment requires its own ClickHouse (trace analytics store), Redis (cache/queue), and S3-compatible blob storage as hard dependencies of the Langfuse product itself, not something this project chose to add. See ADR-0009 and `docs/ARCHITECTURE.md` §7/§9 for the resulting container/cost picture. This ADR's claim should be read as scoped to "the data this project's own code writes," not "every stateful service on the VPS."

## How to reverse
The retriever sits behind LangChain's `VectorStore` interface — swapping to Qdrant/Weaviate/Chroma later means writing a different `VectorStore` implementation (or using LangChain's existing integration for that backend) behind the same MCP tool boundary (ADR-0007); the graph nodes and MCP tool signatures (`search_regulation`, `get_article`) don't change. LangGraph checkpoints and eval results would stay on Postgres regardless (they're not vector data), so a vector-store swap is genuinely isolated to the retrieval path.

## References
- `langchain-postgres`, PyPI: 0.0.17 — https://pypi.org/project/langchain-postgres/ (verified 2026-08-23)
- `PGVectorStore`/`PGEngine` API and legacy `PGVector` migration notebook: Context7 `/langchain-ai/langchain-postgres`, sources `examples/pg_vectorstore_how_to.ipynb`, `examples/pg_vectorstore.ipynb`, `examples/migrate_pgvector_to_pgvectorstore.ipynb` (verified 2026-08-23)
- Market data on vector-DB naming frequency: `docs/research/market_research.md`
