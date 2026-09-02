# tests/fake_embeddings.py — deterministic, no-network stand-in for
# get_embeddings() (compliance_copilot/embeddings.py), used by every test
# that needs an Embeddings implementation but must not call OpenAI (cost +
# network flakiness in CI). ADR-0004's LangChain `Embeddings` interface is
# exactly what makes this substitution possible.
#
# langchain_core ships DeterministicFakeEmbedding (same hash->seed->vector
# idea) but it requires numpy, which isn't a dependency of this project —
# not worth adding numpy just for a test double when stdlib `random` does
# the same job in a few lines.
import hashlib
import math
import random

from langchain_core.embeddings import Embeddings

from compliance_copilot.settings import settings


class FakeEmbeddings(Embeddings):
    """Hashes each text to a seed, draws a `size`-dim vector from that seed
    (so the same text always embeds to the same vector), then L2-normalises
    it — real OpenAI embeddings are unit vectors, and normalisation is what
    the "unit-normalised" requirement is checking for, even though
    cosine_distance is scale-invariant and would work without it."""

    def __init__(self, size: int = settings.embedding_dim) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)  # noqa: S311 — deterministic test fixture, not a security use
        vector = [rng.gauss(0, 1) for _ in range(self.size)]
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
