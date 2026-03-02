from __future__ import annotations

from typing import Any

from saida.connectors.base import BaseConnector


class PostgresConnector(BaseConnector):
    name = "postgres"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def discover(self) -> list[str]:
        # Should return table names/views.
        return []

    def load(self, resource_id: str) -> Any:
        raise NotImplementedError("Postgres connector load() requires SQL client wiring.")

    def get_metadata(self) -> dict:
        return {"type": "postgres", "dsn": self.dsn}
