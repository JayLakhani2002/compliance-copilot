# ADR-0004: Embeddings — LangChain Embeddings interface, OpenAI default / Cohere-on-Bedrock production option

## Status
Accepted. 2026-08-23.

## Context
Retrieval (ADR-0003) needs text turned into vectors — both when ingesting regulation chunks and when embedding a user's query at request time. The AI Act and GDPR corpus is bilingual-relevant (English is the working language, but German-speaking users in the target job market may query in German), so multilingual embedding quality matters more than it would for an English-only corpus.

## Options considered
1. **LangChain `Embeddings` interface, default = OpenAI `text-embedding-3-small`; documented production option = Cohere `embed-multilingual-v3` via AWS Bedrock `eu-central-1`.**
2. **Local `bge-m3`** (a strong open-source multilingual embedding model) — good quality and no per-call cost, but needs GPU (or at least non-trivial CPU) capacity to run at acceptable latency, which is the same ops-cost problem ADR-0002 already ruled out for the LLM tier — rejected for the same reason.
3. **Voyage AI embeddings** — competitive quality, but fewer EU-region hosting options than Cohere-via-Bedrock at the time of this decision, which matters given the project's EU-residency goal.

## Decision
**LangChain's `Embeddings` interface**, so the concrete embedding provider is swappable the same way the chat model is (ADR-0002). Default (day-1, development) = **OpenAI `text-embedding-3-small`** via `langchain-openai` — cheap, good enough quality, supports both English and German reasonably well for a portfolio-scale corpus. **Documented production option** = **Cohere `embed-multilingual-v3` via AWS Bedrock `eu-central-1`**, for the same EU-residency reason as ADR-0002's Bedrock production path, and because Cohere's multilingual model is a stronger fit for a corpus/user base that may mix English and German than the OpenAI default.

**Consistency requirement (not optional):** the embedding model used at ingestion time and the embedding model used to embed a query at request time **must be the same model** — vectors from two different embedding models are not comparable, so switching the default requires re-ingesting (re-embedding) the entire corpus, not just changing a config value for new queries. This is worth stating explicitly in code comments at the embedding call site, since it's an easy mistake to make when adding the Bedrock/Cohere path later without re-running ingestion.

## Why not the others
- **Local `bge-m3`**: rejected for the same ops-cost-in-6-weeks reason as ADR-0002's local-LLM rejection — running an embedding model locally at acceptable latency needs compute this project's CPU-only Hetzner VPS (ADR-0010) doesn't budget for.
- **Voyage AI**: rejected on EU-hosting-option grounds relative to Cohere-via-Bedrock at the time of this decision — worth re-checking if Voyage's EU/self-hostable story changes, since this is a fast-moving space.

## Security & cost implications
- **Security:** the query embedding call sends the user's (post-redaction) question to a third-party API, same trust-boundary concern as ADR-0002's LLM calls (`docs/ARCHITECTURE.md` §6) — `guard_in`'s PII redaction must run before embedding, not just before the chat-model call, since the *query itself* is what gets embedded and sent out.
- **Cost:** embedding cost is dominated by ingestion (a one-time/rare cost — two regulations' worth of chunks, embedded once) rather than by per-query cost (one query embedding per user request, small and cheap). `text-embedding-3-small` is priced for exactly this "cheap, frequent, small calls" pattern.

## How to reverse
Same mechanism as ADR-0002: the embedding provider is constructed once and passed into the vector store (ADR-0003) and the retrieval code — swapping providers is a constructor change, not a rewrite. The one non-trivial part of reversing this decision is the **re-ingestion requirement** above: changing the embedding model means re-running the ingestion job (ADR-0012) against the new model, not just redeploying code.

## References
- `langchain-openai` (Embeddings interface + OpenAI implementation), PyPI: 1.6.0 — https://pypi.org/project/langchain-openai/ (verified 2026-08-23)
- Cohere `embed-multilingual-v3` via Bedrock: referenced as the documented production option per project brief; exact Bedrock model id/region availability to be confirmed at deploy time the same way as ADR-0002's inference-profile caveat — flagged in `docs/handoffs/phase1-architecture/writer.md`
- Market data on `vector-db`/multilingual relevance: `docs/research/market_research.md` (German-language requirement in 46% of ads, informing the multilingual embedding choice)
