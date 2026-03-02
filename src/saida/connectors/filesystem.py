from __future__ import annotations

from pathlib import Path
from typing import Any

from saida.connectors.base import BaseConnector


class FileSystemConnector(BaseConnector):
    name = "filesystem"

    def __init__(self, root: str):
        self.root = Path(root)

    def discover(self) -> list[str]:
        if not self.root.exists():
            return []
        return [str(p) for p in self.root.rglob("*") if p.is_file()]

    def load(self, resource_id: str) -> Any:
        path = Path(resource_id)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(resource_id)
        return path.read_bytes()

    def get_metadata(self) -> dict:
        return {"type": "filesystem", "root": str(self.root)}
