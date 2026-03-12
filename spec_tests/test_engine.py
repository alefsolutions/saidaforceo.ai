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


def test_engine_exposes_current_capabilities() -> None:
    engine = Saida()

    capabilities = engine.capabilities()

    assert capabilities == {
        "analyze": True,
        "profile": True,
        "load_context": True,
        "train": False,
        "predict": False,
        "forecast": False,
    }


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


def test_engine_profile_returns_dataset_profile() -> None:
    engine = Saida()
    dataset = Dataset(
        name="sales",
        source_type="pandas",
        data=pd.DataFrame({"revenue": [100.0, 90.0], "region": ["West", "East"]}),
    )

    profile = engine.profile(dataset)

    assert profile.dataset_name == "sales"
    assert "revenue" in profile.measure_columns


def test_engine_analyze_includes_context_trace_stage() -> None:
    engine = Saida()
    context = engine.load_context(
        """
# Dataset: Sales

## Metric Definitions
revenue: total revenue
""".strip()
    )
    dataset = Dataset(
        name="sales",
        source_type="pandas",
        data=pd.DataFrame(
            {
                "posted_at": ["2026-02-01", "2026-03-01"],
                "revenue": [100.0, 90.0],
                "region": ["West", "East"],
            }
        ),
        context=context,
    )

    result = engine.analyze(dataset, "Why did revenue drop in March?")

    assert any(event.stage == "context" for event in result.trace)


def test_engine_analyze_without_context_has_no_context_trace_stage() -> None:
    engine = Saida()
    dataset = Dataset(
        name="sales",
        source_type="pandas",
        data=pd.DataFrame(
            {
                "posted_at": ["2026-02-01", "2026-03-01"],
                "revenue": [100.0, 90.0],
                "region": ["West", "East"],
            }
        ),
    )

    result = engine.analyze(dataset, "Why did revenue drop in March?")

    assert all(event.stage != "context" for event in result.trace)


_ENGINE_PROFILE_CASES = [
    (
        index,
        pd.DataFrame(
            {
                "posted_at": ["2026-02-01", "2026-03-01", "2026-04-01"],
                "revenue": [float(index), float(index + 5), float(index + 10)],
                "region": [f"Region{index % 4}", f"Region{(index + 1) % 4}", f"Region{(index + 2) % 4}"],
            }
        ),
    )
    for index in range(1, 92)
]


@pytest.mark.parametrize(("case_id", "dataframe"), _ENGINE_PROFILE_CASES)
def test_engine_profiles_many_valid_datasets(case_id: int, dataframe: pd.DataFrame) -> None:
    engine = Saida()
    dataset = Dataset(name=f"sales_{case_id}", source_type="pandas", data=dataframe)

    profile = engine.profile(dataset)

    assert profile.dataset_name == f"sales_{case_id}"
    assert profile.row_count == 3
    assert "posted_at" in profile.time_columns
