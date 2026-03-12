from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from saida.adapters import CSVAdapter, JSONAdapter, PandasAdapter, SQLAdapter
from saida.exceptions import AdapterError


def test_csv_adapter_loads_context_and_adds_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    context_path = tmp_path / "sales.md"
    csv_path.write_text(" revenue ,region\n100,West\n", encoding="utf-8")
    context_path.write_text(
        """
# Dataset: Sales

## Metric Definitions
revenue: total revenue
""".strip(),
        encoding="utf-8",
    )

    dataset = CSVAdapter(csv_path, context_path=context_path).load()

    assert list(dataset.data.columns) == ["revenue", "region"]
    assert dataset.context is not None
    assert dataset.context.metric_definitions["revenue"] == "total revenue"
    assert dataset.metadata["row_count"] == 1
    assert dataset.metadata["column_count"] == 2


def test_csv_adapter_rejects_empty_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("revenue,region\n", encoding="utf-8")

    with pytest.raises(AdapterError, match="Loaded dataset is empty."):
        CSVAdapter(csv_path).load()


def test_json_adapter_rejects_duplicate_columns_after_normalization(tmp_path: Path) -> None:
    json_path = tmp_path / "duplicate.json"
    json_path.write_text('[{" revenue ": 100, "revenue": 90}]', encoding="utf-8")

    with pytest.raises(AdapterError, match="duplicate column names"):
        JSONAdapter(json_path).load()


def test_pandas_adapter_rejects_non_dataframe_input() -> None:
    with pytest.raises(AdapterError, match="requires a pandas DataFrame"):
        PandasAdapter([{"revenue": 100}]).load()  # type: ignore[arg-type]


def test_pandas_adapter_rejects_duplicate_columns_after_normalization() -> None:
    dataframe = pd.DataFrame([[100, 90]], columns=[" revenue ", "revenue"])

    with pytest.raises(AdapterError, match="duplicate column names"):
        PandasAdapter(dataframe).load()


def test_sql_adapter_raises_clean_error_for_bad_query(tmp_path: Path) -> None:
    database_path = tmp_path / "sales.db"
    connection = sqlite3.connect(database_path)
    connection.execute("create table sales (revenue integer)")
    connection.commit()
    connection.close()

    with pytest.raises(AdapterError, match="Failed to load SQL query results"):
        SQLAdapter(database_path, "select missing_column from sales").load()
