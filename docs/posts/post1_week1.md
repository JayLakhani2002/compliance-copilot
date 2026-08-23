# LinkedIn Post #1 — Week 1 (publish after v0.1 tag + repo public)

**Attach:** screenshot of the eval table (plain vs articles-only) or the mermaid container diagram.

---

Week 1 of building an EU AI Act + GDPR compliance agent in public — and the most useful thing I shipped isn't the retrieval, it's the test that grades it.

What exists after week 1:
▸ Both regulations ingested from the Publications Office Cellar API (EUR-Lex blocks scripted HTTP — lesson learned: find the official machine endpoint before scraping) and chunked **by article**, not by token windows, so every future answer can cite "Art. 6(2)" like a lawyer would.
▸ Postgres + pgvector as the single database: 576 chunks, HNSW cosine index, idempotent ingest (re-runs re-embed only changed text).
▸ A 30-question golden set and a retrieval eval (hit@5, MRR) that runs in one command.

The eval already earned its keep. My first "obvious" retriever searched everything and answered "What is a high-risk AI system?" with **Recital 52** — the right *topic*, the wrong *authority* (recitals aren't binding law; Article 6 ranked 11th). Measured properly:

plain search: hit@5 0.63 · MRR 0.33
articles-first: hit@5 0.87 · MRR 0.70

So articles-first is now the design — decided by numbers, recorded in an ADR with the losing option's scores, reversible by one parameter. It also honestly records what still fails: 4 definition questions drown inside mega-articles like AI Act Art. 3 (68 definitions in one article). That's next week's chunking fix, not a footnote.

Stack so far: Python 3.12, SQLAlchemy 2, pgvector, LangChain embeddings interface (text-embedding-3-small, swappable for an EU-hosted model), pytest + GitHub Actions running integration tests against a real pgvector container on every PR.

Next week: the LangGraph agent, streamed answers with article citations, Langfuse tracing — and the eval suite becomes a merge gate: if faithfulness drops, the PR doesn't land.

Repo: github.com/JayLakhani2002/compliance-copilot

#AIEngineering #LangChain #RAG #pgvector #Evals #EUAIAct #GDPR #BuildInPublic
