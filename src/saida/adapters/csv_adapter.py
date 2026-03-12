"""CSV adapter implementation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from saida.context import SourceContextParser
from saida.exceptions import AdapterError
from saida.schemas import Dataset


class CSVAdapter:
    """Load CSV data into the SAIDA dataset schema."""

    def __init__(self, path: str | Path, *, name: str | None = None, context_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.context_path = Path(context_path) if context_path else None

    def load(self) -> Dataset:
        """Load the CSV file and attach optional semantic context."""
        if not self.path.exists():
            raise AdapterError(f"CSV file not found: {self.path}")

        try:
            dataframe = pd.read_csv(self.path)
        except Exception as exc:  # pragma: no cover
            raise AdapterError(f"Failed to load CSV file: {self.path}") from exc

        context = None
        if self.context_path:
            context = SourceContextParser().parse_file(self.context_path)

        return Dataset(
            name=self.name,
            source_type="csv",
            data=dataframe,
            metadata={"path": str(self.path)},
            context=context,
        )
