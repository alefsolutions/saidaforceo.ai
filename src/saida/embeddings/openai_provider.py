from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from saida.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"

    def __init__(self, client: Any | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.getenv("SAIDA_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        self._client = OpenAI(api_key=api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._get_client().embeddings.create(model=self.model, input=texts)
        return [list(row.embedding) for row in response.data]
