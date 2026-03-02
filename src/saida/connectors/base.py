from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    name: str

    @abstractmethod
    def discover(self) -> list[str]:
        """Return resource identifiers available for ingestion."""

    @abstractmethod
    def load(self, resource_id: str) -> Any:
        """Return raw or structured resource content."""

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return metadata about the connector/source system."""
