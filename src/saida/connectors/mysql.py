from __future__ import annotations

from typing import Any

from saida.connectors.base import BaseConnector


class MySQLConnector(BaseConnector):
    name = "mysql"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def discover(self) -> list[str]:
        return []

    def load(self, resource_id: str) -> Any:
        raise NotImplementedError("MySQL connector load() requires SQL client wiring.")

    def get_metadata(self) -> dict:
        return {"type": "mysql", "dsn": self.dsn}
