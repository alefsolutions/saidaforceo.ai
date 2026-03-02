from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    name: str

    @abstractmethod
    def explain(self, prompt: str) -> str:
        raise NotImplementedError
