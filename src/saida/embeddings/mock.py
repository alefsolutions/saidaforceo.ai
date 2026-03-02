from __future__ import annotations

from saida.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    name = "mock"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 13), 0.1, 0.2] for t in texts]
