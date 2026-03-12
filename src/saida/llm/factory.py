"""Factory helpers for optional LLM providers."""

from __future__ import annotations

from saida.config import LlmConfig
from saida.llm.base import BaseLlmProvider
from saida.llm.openai_provider import OpenAiLlmProvider
from saida.llm.ollama import OllamaLlmProvider


def build_llm_provider(config: LlmConfig) -> BaseLlmProvider | None:
    """Build the configured LLM provider, if one is enabled and supported."""
    if not config.enabled or not config.provider:
        return None
    if config.provider == "ollama":
        return OllamaLlmProvider(config)
    if config.provider == "openai":
        return OpenAiLlmProvider(config)
    return None
