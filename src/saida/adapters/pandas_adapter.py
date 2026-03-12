"""Pandas adapter implementation."""

from __future__ import annotations

import pandas as pd

from saida.context import SourceContextParser
from saida.schemas import Dataset


class PandasAdapter:
    """Wrap an existing pandas DataFrame in the SAIDA dataset schema."""

    def __init__(self, dataframe: pd.DataFrame, *, name: str = "dataframe", context_markdown: str | None = None) -> None:
        self.dataframe = dataframe.copy()
        self.name = name
        self.context_markdown = context_markdown

    def load(self) -> Dataset:
        """Return the DataFrame as a SAIDA dataset."""
        context = None
        if self.context_markdown:
            context = SourceContextParser().parse(self.context_markdown)

        return Dataset(
            name=self.name,
            source_type="pandas",
            data=self.dataframe,
            metadata={"rows": len(self.dataframe), "columns": list(self.dataframe.columns)},
            context=context,
        )
