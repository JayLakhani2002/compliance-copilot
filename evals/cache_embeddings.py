# evals/cache_embeddings.py — populates evals/embeddings_cache/{queries,
# corpus}.jsonl with REAL text-embedding-3-small vectors (ADR-0017), so the
# CI quality-gate job can retrieve against genuine vectors without ever
# calling OpenAI. Run this locally, with a real OPENAI_API_KEY, whenever the
# golden questions, the corpus snapshot, or the embedding model change:
#
#     set -a; source .env; set +a
#     uv run python -m evals.cache_embeddings
#
# Idempotent: an entry already present with the SAME model is skipped, so a
# re-run after adding one new golden question only pays for that one
# embedding call, not all 30 + 587 again.
#
# Corpus texts come from parsing evals/corpus_snapshot/*.xhtml through the
# SAME parse_articles -> split_article path ingest/pipeline.py's ingest()
# uses (imported here, not reimplemented) — so the cached vectors key on
# exactly the chunk texts the real pipeline will ever ask an Embeddings
# object to embed.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compliance_copilot.cached_embeddings import encode_vector
from compliance_copilot.embeddings import get_embeddings
from compliance_copilot.ingest.chunker import split_article
from compliance_copilot.ingest.eurlex import REGULATIONS, fetch_xhtml, parse_articles
from compliance_copilot.settings import settings
from evals.run_retrieval_eval import load_golden

CACHE_DIR = Path("evals/embeddings_cache")
CORPUS_SNAPSHOT_DIR = Path("evals/corpus_snapshot")
QUERIES_PATH = CACHE_DIR / "queries.jsonl"
CORPUS_PATH = CACHE_DIR / "corpus.jsonl"

# Batch size for embed_documents() calls — same reasoning as
# pipeline.py's EMBED_BATCH_SIZE (round-trip count vs. one giant request).
BATCH_SIZE = 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    entries: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries[entry["sha256"]] = entry
    return entries


def _write(path: Path, entries: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted by sha256: a diffable, deterministic file — a re-run that adds
    # one new entry only appends/reorders that one line's neighbourhood,
    # not the whole file in insertion order.
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(entries):
            f.write(json.dumps(entries[key]) + "\n")


def _corpus_texts() -> list[str]:
    """Every chunk text the real pipeline embeds for both regulations —
    fetch_xhtml's cache-first path reads the committed snapshot (no
    network), then the same parse_articles/split_article the pipeline
    imports turns it into ChunkParts (ADR-0017: not reimplemented here)."""
    texts: list[str] = []
    for key in REGULATIONS:
        xhtml = fetch_xhtml(REGULATIONS[key]["celex"], cache_dir=CORPUS_SNAPSHOT_DIR)
        articles = parse_articles(xhtml, key)
        parts = [p for article in articles for p in split_article(article)]
        texts.extend(p.text for p in parts)
    return texts


def _fill_cache(path: Path, texts: list[str], embeddings, model: str) -> tuple[int, int]:
    """Embeds every text in `texts` not already cached under `model`, merges
    into the existing file, writes it back. Returns (new, total)."""
    existing = _load_existing(path)
    to_embed = [t for t in texts if existing.get(_sha256(t), {}).get("model") != model]
    # Dedup: two chunks/questions with identical text would otherwise embed
    # twice in the same run for no reason (sha256 key is the same either way).
    seen: set[str] = set()
    unique_to_embed = [t for t in to_embed if not (_sha256(t) in seen or seen.add(_sha256(t)))]

    for i in range(0, len(unique_to_embed), BATCH_SIZE):
        batch = unique_to_embed[i : i + BATCH_SIZE]
        vectors = embeddings.embed_documents(batch)
        for text, vector in zip(batch, vectors, strict=True):
            existing[_sha256(text)] = {
                "sha256": _sha256(text),
                "model": model,
                "dim": len(vector),
                "vec": encode_vector(vector),
            }

    _write(path, existing)
    return len(unique_to_embed), len(existing)


def main() -> None:
    embeddings = get_embeddings()  # real OpenAIEmbeddings — never cached here
    model = settings.embedding_model

    questions = [entry.question for entry in load_golden()]
    new_q, total_q = _fill_cache(QUERIES_PATH, questions, embeddings, model)
    print(
        f"queries: {new_q} new, {total_q} total -> {QUERIES_PATH} "
        f"({QUERIES_PATH.stat().st_size:,} bytes)"
    )

    corpus_texts = _corpus_texts()
    new_c, total_c = _fill_cache(CORPUS_PATH, corpus_texts, embeddings, model)
    print(
        f"corpus:  {new_c} new, {total_c} total -> {CORPUS_PATH} "
        f"({CORPUS_PATH.stat().st_size:,} bytes)"
    )

    # Rough cost note (ADR-0002's verified text-embedding-3-small pricing is
    # per-token, not per-call; embedding calls are far cheaper than chat
    # completions — this is a sanity order-of-magnitude, not an invoice).
    embedded = new_q + new_c
    print(f"embedded {embedded} new text(s) this run — embeddings cost is a few cents at most.")


if __name__ == "__main__":
    main()
