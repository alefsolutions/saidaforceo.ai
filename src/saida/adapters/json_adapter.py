"""JSON adapter implementation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from saida.context import SourceContextParser
from saida.exceptions import AdapterError
from saida.schemas import Dataset


class JSONAdapter:
    """Load JSON data into the SAIDA dataset schema."""

    def __init__(self, path: str | Path, *, name: str | None = None, context_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.context_path = Path(context_path) if context_path else None

    def load(self) -> Dataset:
        """Load a JSON file and attach optional semantic context."""
        if not self.path.exists():
            raise AdapterError(f"JSON file not found: {self.path}")

        try:
            dataframe = pd.read_json(self.path)
        except ValueError:
            try:
                dataframe = pd.read_json(self.path, lines=True)
            except Exception as exc:  # pragma: no cover
                raise AdapterError(f"Failed to load JSON file: {self.path}") from exc
        except Exception as exc:  # pragma: no cover
            raise AdapterError(f"Failed to load JSON file: {self.path}") from exc

        context = None
        if self.context_path:
            context = SourceContextParser().parse_file(self.context_path)

        return Dataset(
            name=self.name,
            source_type="json",
            data=dataframe,
            metadata={"path": str(self.path)},
            context=context,
        )
