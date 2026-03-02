from __future__ import annotations

from typing import Any

from saida.connectors.base import BaseConnector


class GoogleDriveConnector(BaseConnector):
    name = "gdrive"

    def discover(self) -> list[str]:
        # Integration hook for Google Drive API.
        return []

    def load(self, resource_id: str) -> Any:
        raise NotImplementedError("Google Drive connector is a scaffold. Add API client wiring.")

    def get_metadata(self) -> dict:
        return {"type": "gdrive"}
