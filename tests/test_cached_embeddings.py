# tests/test_cached_embeddings.py — plumbing tests for CachedEmbeddings
# (ADR-0017). No network, no real cache files: each test writes a tiny JSONL
# fixture into tmp_path so it's self-contained and never depends on the real
# evals/embeddings_cache/ files existing.
import json
import math
import os
from pathlib import Path

import pytest

from compliance_copilot.cached_embeddings import CachedEmbeddings, decode_vector, encode_vector
from compliance_copilot.embeddings import get_embeddings


def _write_cache(path: Path, sha256: str, model: str, vector: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"sha256": sha256, "model": model, "dim": len(vector), "vec": encode_vector(vector)}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_round_trip_precision_is_float32_not_meaningfully_lossy():
    """encode_vector/decode_vector pack at float32 — cosine distance against
    the original float64 vector should barely move (task's own bound:
    |dot(v32, v64) - 1| < 1e-5 after both are unit-normalised)."""
    v64 = [math.sin(i) for i in range(1536)]
    norm = math.sqrt(sum(x * x for x in v64))
    v64 = [x / norm for x in v64]

    v32 = decode_vector(encode_vector(v64))
    norm32 = math.sqrt(sum(x * x for x in v32))
    v32 = [x / norm32 for x in v32]

    dot = sum(a * b for a, b in zip(v64, v32, strict=True))
    assert abs(dot - 1) < 1e-5


def test_embed_query_returns_the_cached_vector(tmp_path):
    text = "What is a high-risk AI system?"
    key = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    vector = [0.1, 0.2, 0.3]
    path = tmp_path / "queries.jsonl"
    _write_cache(path, key, "text-embedding-3-small", vector)

    embeddings = CachedEmbeddings(cache_paths=(path,), model="text-embedding-3-small")
    # float32 round-trip (encode_vector packs float32), not exact float64
    # equality — matches decode_vector's own precision contract.
    got = embeddings.embed_query(text)
    assert got == pytest.approx(vector, abs=1e-6)
    assert embeddings.embed_documents([text]) == [got]


def test_cache_miss_raises_keyerror_naming_the_refresh_command(tmp_path):
    path = tmp_path / "queries.jsonl"
    _write_cache(path, "some-other-hash", "text-embedding-3-small", [0.1])

    embeddings = CachedEmbeddings(cache_paths=(path,), model="text-embedding-3-small")
    try:
        embeddings.embed_query("a question never cached")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        message = str(exc)
        assert "make eval-cache" in message
        assert str(path) in message


def test_model_mismatch_raises(tmp_path):
    text = "some question"
    key = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    path = tmp_path / "queries.jsonl"
    _write_cache(path, key, "an-old-model", [0.1, 0.2])

    embeddings = CachedEmbeddings(cache_paths=(path,), model="text-embedding-3-small")
    try:
        embeddings.embed_query(text)
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "make eval-cache" in str(exc)


def test_get_embeddings_returns_cached_under_env_switch(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "cached")
    assert isinstance(get_embeddings(), CachedEmbeddings)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    assert os.environ.get("EMBEDDINGS_PROVIDER") is None
