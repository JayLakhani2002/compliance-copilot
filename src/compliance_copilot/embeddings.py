# src/compliance_copilot/embeddings.py — the one place that constructs an
# embedding provider (docs/ARCHITECTURE.md §1, ADR-0004). Both the ingest
# pipeline (turning regulation text into stored vectors) and query-time
# search (turning a user's question into a vector to search with) call
# get_embeddings() rather than importing OpenAIEmbeddings directly, so
# swapping to the documented production option (Cohere embed-multilingual-v3
# via AWS Bedrock eu-central-1, ADR-0004) is a one-function change here, not
# a search-and-replace across the codebase.
#
# IMPORTANT (ADR-0004 "Consistency requirement"): the embedding model used at
# ingestion and the one used to embed a query at request time MUST be the
# same model — vectors from two different models are not comparable.
# Changing `settings.embedding_model` means re-ingesting the whole corpus,
# not just redeploying.
#
# EMBEDDINGS_PROVIDER=cached (ADR-0017, CI only): swaps in CachedEmbeddings,
# which reads REAL previously-computed vectors from evals/embeddings_cache/
# instead of calling OpenAI — lets the CI quality-gate job ingest+retrieve
# with no OPENAI_API_KEY and no network call. Read directly from the
# environment, not a Settings field, same reasoning as the API-key fields
# below: this is a CI-only escape hatch, not a value a real deploy ever sets.
import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from compliance_copilot.settings import settings


def get_embeddings() -> Embeddings:
    """Returns the app's embedding provider, typed as LangChain's `Embeddings`
    interface (not the concrete OpenAIEmbeddings class) — callers depend on
    the interface (embed_documents/embed_query), never on OpenAI-specific
    behaviour, which is what makes the Bedrock/Cohere swap possible later."""
    if os.environ.get("EMBEDDINGS_PROVIDER") == "cached":
        # Lazy import: only CI's quality-gate job ever takes this branch, so
        # a normal dev/prod run never even imports cached_embeddings.py.
        from compliance_copilot.cached_embeddings import CachedEmbeddings

        return CachedEmbeddings()
    return OpenAIEmbeddings(model=settings.embedding_model)
