from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
