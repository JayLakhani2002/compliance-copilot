# Lesson 03 — The corpus and chunking legal text (Week 1, Day 3)

**What it is.** Ingestion = fetch source documents → split into pieces ("chunks") → attach metadata → store. Chunking decides what a retrieval hit *is*. For the AI Act and GDPR we chunk by **article** (and separately by **recital**), not by fixed token windows.

**Why we need it here.** The user will ask "Is my CV-screening tool high-risk?" The best answer cites *Article 6 + Annex III* — whole legal units with a number a lawyer can look up. A 500-token window that starts mid-sentence in Art. 6(2) and ends inside Art. 7 is neither citable nor faithful. Structure-aware chunks also give us metadata (regulation, article no., title, chapter) for filtering and for the citation guard in Week 3.

**Why not the alternative.** Fixed-size windows (e.g. `RecursiveCharacterTextSplitter` at 512 tokens, 50 overlap) are the default in tutorials because they need no knowledge of the document. Cost: citations become fuzzy, long articles get split arbitrarily, short ones get merged with neighbours. We *will* need a size cap (some articles are > 2k tokens) — so the design is "article first, then split only oversize articles by paragraph, keeping the article id on every piece". Best of both.

**Where the data comes from (and a real-world lesson).** eur-lex.europa.eu blocks scripts (empty HTTP 202). The Publications Office *Cellar* API serves the same XHTML via content negotiation (`Accept: application/xhtml+xml`). Senior habit: look for the official machine endpoint before scraping a website; read the licence first (ADR-0012: reuse authorised under Decision 2011/833/EU; only the Official Journal is authentic → we show a disclaimer).

**How a senior thinks about it.**
- *Failure modes:* the HTML structure changes → parser silently returns 0 articles. Defence: unit tests on a committed fixture + a "count sanity" assertion at ingest time (AI Act must yield 113 articles).
- *Idempotency:* re-running ingest must not duplicate rows (upsert by `(regulation, article)`).
- *Provenance:* store `source_url`, CELEX id, fetch date, and a content hash; GDPR-style thinking applied to our own data.
- *Cost:* tiny corpus; the expensive part is embeddings (Day 4) — so ingest caches raw files and only re-embeds changed chunks.
- *Security:* retrieved text later enters an LLM prompt → it is **untrusted input** (indirect prompt injection). We tag chunks as "document" content and treat them accordingly in Week 3.

**Analogy.** Cutting a law book into index cards: one card per article, with the article number on top — not cutting every page into equal strips.

**Interview question.** "How would you chunk a 300-page contract for RAG, and how would you know your chunking is good?" (Shape: structure-aware + size cap + overlap only where needed; evaluate with a retrieval golden set (hit@k) and faithfulness, not by eyeballing.)

**Check yourself.** 1) Why does chunking affect citations? 2) What breaks silently if the HTML changes, and how do we catch it?
