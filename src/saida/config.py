"""Configuration objects for SAIDA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NlpConfig:
    """Configuration for request normalization."""

    enable_transformers: bool = False
    zero_shot_model: str = "facebook/bart-large-mnli"
    confidence_threshold: float = 0.55


@dataclass(slots=True)
class ReasoningConfig:
    """Configuration for optional reasoning integrations."""

    enabled: bool = False
    provider: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SaidaConfig:
    """Top-level configuration for the SAIDA engine."""

    nlp: NlpConfig = field(default_factory=NlpConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
