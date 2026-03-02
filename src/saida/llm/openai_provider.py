from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from saida.llm.base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self, client: Any | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.getenv("SAIDA_OPENAI_LLM_MODEL", "gpt-4o-mini")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAILLMProvider")
        self._client = OpenAI(api_key=api_key)
        return self._client

    def explain(self, prompt: str) -> str:
        response = self._get_client().responses.create(model=self.model, input=prompt)
        text = getattr(response, "output_text", "")
        return (text or "").strip()
