from __future__ import annotations

import pandas as pd
import pytest

from saida import Saida
from saida.exceptions import ValidationError
from saida.schemas import Dataset


def test_engine_load_context_parses_markdown() -> None:
    engine = Saida()

    context = engine.load_context(
        """
# Dataset: Sales

## Metric Definitions
revenue: total revenue
""".strip()
    )

    assert context.source_summary == "Sales"
    assert context.metric_definitions["revenue"] == "total revenue"


def test_engine_rejects_empty_dataset() -> None:
    engine = Saida()
    dataset = Dataset(name="empty", source_type="pandas", data=pd.DataFrame({"revenue": []}))

    with pytest.raises(ValidationError, match="Cannot analyze an empty dataset"):
        engine.analyze(dataset, "Show revenue")


def test_engine_rejects_duplicate_columns() -> None:
    engine = Saida()
    dataset = Dataset(name="dup", source_type="pandas", data=pd.DataFrame([[1, 2]], columns=["revenue", "revenue"]))

    with pytest.raises(ValidationError, match="duplicate column names"):
        engine.analyze(dataset, "Show revenue")


def test_engine_rejects_non_dataframe_dataset() -> None:
    engine = Saida()
    dataset = Dataset(name="bad", source_type="pandas", data=[{"revenue": 1}])  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="must be a pandas DataFrame"):
        engine.analyze(dataset, "Show revenue")
