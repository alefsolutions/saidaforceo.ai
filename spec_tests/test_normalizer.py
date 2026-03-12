from __future__ import annotations

import pandas as pd

from saida.nlp import RequestNormalizer
from saida.schemas import ColumnProfile, Dataset, DatasetProfile, SourceContext


def build_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="sales",
        row_count=4,
        column_count=4,
        columns=[
            ColumnProfile(
                name="revenue",
                inferred_type="float",
                nullable=False,
                null_ratio=0.0,
                unique_count=4,
                distinct_ratio=1.0,
                sample_values=[100.0, 80.0],
                is_measure_candidate=True,
            ),
            ColumnProfile(
                name="region",
                inferred_type="category",
                nullable=False,
                null_ratio=0.0,
                unique_count=2,
                distinct_ratio=0.5,
                sample_values=["West", "East"],
                is_dimension_candidate=True,
            ),
            ColumnProfile(
                name="segment",
                inferred_type="category",
                nullable=False,
                null_ratio=0.0,
                unique_count=2,
                distinct_ratio=0.5,
                sample_values=["SMB", "Enterprise"],
                is_dimension_candidate=True,
            ),
            ColumnProfile(
                name="posted_at",
                inferred_type="datetime",
                nullable=False,
                null_ratio=0.0,
                unique_count=4,
                distinct_ratio=1.0,
                sample_values=["2026-03-01"],
                is_time_candidate=True,
            ),
        ],
        measure_columns=["revenue"],
        dimension_columns=["region", "segment"],
        time_columns=["posted_at"],
        identifier_columns=[],
    )


def build_dataset() -> Dataset:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 80.0, 60.0, 40.0],
            "region": ["West", "East", "West", "East"],
            "segment": ["SMB", "SMB", "Enterprise", "Enterprise"],
            "posted_at": ["2026-02-01", "2026-02-01", "2026-03-01", "2026-03-01"],
        }
    )
    return Dataset(name="sales", source_type="pandas", data=dataframe)


def test_normalizer_extracts_group_by_filters_and_time_reference() -> None:
    normalizer = RequestNormalizer()
    context = SourceContext(raw_markdown="", metric_definitions={"revenue": "total revenue"})

    request, warnings = normalizer.normalize(
        "Why did revenue drop in March by region for West?",
        build_dataset(),
        build_profile(),
        context,
    )

    assert warnings == []
    assert request.task_type_hint == "diagnostic"
    assert request.target == "revenue"
    assert request.group_by == ["region"]
    assert request.filters == {"region": "West"}
    assert request.time_reference == {"type": "month_name", "value": "march", "month": "3"}


def test_normalizer_extracts_quarter_and_multiple_group_triggers() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize(
        "Show revenue across region and for each segment in Q1",
        build_dataset(),
        build_profile(),
        None,
    )

    assert request.task_type_hint == "descriptive"
    assert request.group_by == ["region", "segment"]
    assert request.time_reference == {"type": "quarter", "value": "q1", "quarter": "1"}
