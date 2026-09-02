# src/compliance_copilot/cached_embeddings.py — a drop-in `Embeddings`
# (ADR-0004's interface) backed by a committed JSONL cache of REAL vectors,
# never a fake/random one (ADR-0017). Lets CI run the retrieval quality gate
# (evals/run_retrieval_eval.py) and the answer-quality gate's ingest step
# with zero network calls and zero OPENAI_API_KEY, while still scoring the
# genuine text-embedding-3-small vectors `evals/cache_embeddings.py` computed
# locally and committed under evals/embeddings_cache/.
#
# Cache format (one JSON object per line, diffable): {"sha256": sha256(text),
# "model": embedding model name, "dim": vector length, "vec": base64 of the
# vector packed as stdlib `array` float32 bytes}. Keyed on sha256(text), not
# a row id — the same lookup pipeline.py already keys its own idempotency
# check on (`Chunk.content_hash`), so a chunk's cache entry is found the same
# way a chunk's "has this text changed" check already works.
#
# Why float32 (not JSON floats or float64): OpenAI's own vectors are
# float32-precision already, so packing at float32 loses nothing real, and
# raw binary (via stdlib `array`/`struct`, base64-encoded into the JSON
# string) is far smaller than a JSON array of decimal floats — no numpy
# needed for this (ponytail: stdlib `array` does the job in a few lines).
from __future__ import annotations

import array
import base64
import hashlib
import json
from functools import cache
from pathlib import Path

from langchain_core.embeddings import Embeddings

from compliance_copilot.settings import settings

# Two files, one merged lookup: queries.jsonl holds the 30 golden questions
# (looked up via embed_query at retrieval time), corpus.jsonl holds every
# ingested chunk's text (looked up via embed_documents at ingest time). A
# CachedEmbeddings instance doesn't need to know which caller wants which —
# both files use the identical {sha256, model, dim, vec} shape, so loading
# them into one dict keyed by sha256 just works for either lookup.
_CACHE_DIR = Path("evals/embeddings_cache")
DEFAULT_CACHE_PATHS = (_CACHE_DIR / "queries.jsonl", _CACHE_DIR / "corpus.jsonl")


def encode_vector(vector: list[float]) -> str:
    """Packs a vector as float32 bytes (stdlib `array`), base64-encoded for
    a JSONL text line. Shared by CachedEmbeddings' decode path and
    evals/cache_embeddings.py's write path, so the two never drift apart."""
    return base64.b64encode(array.array("f", vector).tobytes()).decode("ascii")


def decode_vector(vec_b64: str) -> list[float]:
    """Inverse of encode_vector — base64 -> float32 bytes -> list[float]."""
    arr = array.array("f")
    arr.frombytes(base64.b64decode(vec_b64))
    return list(arr)


@cache
def _load_cache(paths: tuple[Path, ...]) -> dict[str, dict]:
    """Reads every cache file's lines into one {sha256: entry} dict, once per
    distinct `paths` tuple per process (functools.lru_cache — "lazy": no file
    is opened until a CachedEmbeddings actually needs a lookup). A later
    entry for the same sha256 overwrites an earlier one, which only matters
    if two files somehow shared a key — they don't in practice (queries vs.
    chunk texts hash to disjoint sha256s), but last-wins is the harmless
    choice if it ever did."""
    entries: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entries[entry["sha256"]] = entry
    return entries


class CachedEmbeddings(Embeddings):
    """Looks up `sha256(text)` in the committed cache instead of calling any
    embedding API. Never falls back to the network or a fake vector on a
    miss — it raises `KeyError` naming the cache file(s) and the refresh
    command, because a silent fallback here is exactly the failure mode
    lesson 05 warns against: a "green" CI run that was actually scored
    against the wrong (or fake) vectors."""

    def __init__(
        self,
        cache_paths: tuple[Path, ...] = DEFAULT_CACHE_PATHS,
        model: str | None = None,
    ) -> None:
        self.cache_paths = cache_paths
        self.model = model or settings.embedding_model

    def _lookup(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = _load_cache(self.cache_paths).get(key)
        if entry is None or entry["model"] != self.model:
            paths = ", ".join(str(p) for p in self.cache_paths)
            raise KeyError(
                f"No cached embedding for this text (sha256={key}, model={self.model!r}) in "
                f"{paths}. Run `make eval-cache` to (re)populate the cache — CachedEmbeddings "
                "never falls back to a live API call or a fake vector."
            )
        vector = decode_vector(entry["vec"])
        if len(vector) != entry["dim"]:
            raise ValueError(
                f"Cache entry for sha256={key} claims dim={entry['dim']} but decoded to "
                f"{len(vector)} floats — the cache file is corrupt, re-run `make eval-cache`."
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._lookup(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._lookup(text)
