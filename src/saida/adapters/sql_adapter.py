"""SQL adapter implementation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from saida.context import SourceContextParser
from saida.exceptions import AdapterError
from saida.schemas import Dataset


class SQLAdapter:
    """Load query results from a SQLite database into the SAIDA dataset schema."""

    def __init__(
        self,
        database_path: str | Path,
        query: str,
        *,
        name: str = "sql_query",
        context_path: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.query = query
        self.name = name
        self.context_path = Path(context_path) if context_path else None

    def load(self) -> Dataset:
        """Execute the SQL query and return a normalized dataset."""
        if not self.database_path.exists():
            raise AdapterError(f"SQLite database not found: {self.database_path}")

        try:
            connection = sqlite3.connect(self.database_path)
            dataframe = pd.read_sql_query(self.query, connection)
            connection.close()
        except Exception as exc:  # pragma: no cover
            raise AdapterError(f"Failed to load SQL query results from: {self.database_path}") from exc

        context = None
        if self.context_path:
            context = SourceContextParser().parse_file(self.context_path)

        return Dataset(
            name=self.name,
            source_type="sql",
            data=dataframe,
            metadata={"database_path": str(self.database_path), "query": self.query},
            context=context,
        )
