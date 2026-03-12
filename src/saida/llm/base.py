"""Base interface for optional LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saida.llm.models import IntentProposal, ResponseContext, ResponseProposal


class BaseLlmProvider(ABC):
    """Abstract provider used by SAIDA for optional prompt and response handling."""

    provider_name: str = "base"

    @abstractmethod
    def interpret_prompt(
        self,
        question: str,
        dataset_name: str,
        profile_summary: str,
        context_summary: str | None,
    ) -> IntentProposal | None:
        """Return a structured prompt interpretation or None to skip LLM handling."""

    @abstractmethod
    def generate_response(self, response_context: ResponseContext) -> ResponseProposal | None:
        """Return a structured response proposal or None to skip LLM reasoning."""
