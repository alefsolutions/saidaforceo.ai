from __future__ import annotations

from saida.llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    name = "mock"

    def explain(self, prompt: str) -> str:
        return f"[mock-llm] {prompt[:500]}"
