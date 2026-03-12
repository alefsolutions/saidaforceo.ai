from __future__ import annotations

import pandas as pd
import pytest

from saida.nlp import RequestNormalizer
from saida.exceptions import ValidationError
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


def test_normalizer_rejects_empty_question() -> None:
    normalizer = RequestNormalizer()

    with pytest.raises(ValidationError, match="question cannot be empty"):
        normalizer.normalize("", build_dataset(), build_profile(), None)


def test_normalizer_rejects_missing_target_when_no_measure_columns_exist() -> None:
    normalizer = RequestNormalizer()
    profile = build_profile()
    profile.measure_columns = []

    with pytest.raises(ValidationError, match="No target metric could be resolved"):
        normalizer.normalize("Show something interesting", build_dataset(), profile, None)


def test_normalizer_falls_back_to_first_measure_with_warning() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize("Show data by region", build_dataset(), build_profile(), None)

    assert request.target == "revenue"
    assert any("No explicit metric matched the prompt" in warning for warning in warnings)


def test_normalizer_extracts_relative_time_reference() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show revenue for last month", build_dataset(), build_profile(), None)

    assert request.time_reference == {"type": "relative_period", "value": "last_month"}


def test_normalizer_extracts_horizon_from_prompt() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Forecast revenue for 3 months", build_dataset(), build_profile(), None)

    assert request.horizon == 3
    assert request.task_type_hint == "forecasting"


def test_normalizer_extracts_average_aggregation() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("What is the average revenue?", build_dataset(), build_profile(), None)

    assert request.target == "revenue"
    assert request.aggregation == "mean"


def test_normalizer_extracts_highest_aggregation() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show highest revenue", build_dataset(), build_profile(), None)

    assert request.target == "revenue"
    assert request.aggregation == "max"


def test_normalizer_extracts_lowest_aggregation() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show lowest revenue", build_dataset(), build_profile(), None)

    assert request.target == "revenue"
    assert request.aggregation == "min"


def test_normalizer_uses_context_metric_aliases() -> None:
    normalizer = RequestNormalizer()
    context = SourceContext(raw_markdown="", metric_definitions={"profit": "net profit"})

    request, _ = normalizer.normalize("Show profit by region", build_dataset(), build_profile(), context)

    assert request.target == "profit"


def test_normalizer_extracts_equals_filters() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show revenue where region=West", build_dataset(), build_profile(), None)

    assert request.filters == {"region": "West"}


def test_normalizer_deduplicates_group_by_matches() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show revenue by region across region", build_dataset(), build_profile(), None)

    assert request.group_by == ["region"]


def test_normalizer_sets_options_payload() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize("Show revenue", build_dataset(), build_profile(), None)

    assert request.options["dataset"] == "sales"
    assert request.options["nlp_backend"] == "rules"


def test_normalizer_detects_distinct_value_listing_for_dimension_prompt() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize("Give me a list of all segments", build_dataset(), build_profile(), None)

    assert warnings == []
    assert request.target == "segment"
    assert request.options["distinct_values"] is True


def test_normalizer_detects_row_count_intent() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize("How many data rows do we have?", build_dataset(), build_profile(), None)

    assert warnings == []
    assert request.intent_name == "row_count"
    assert request.target is None


def test_normalizer_detects_representation_ranking_intent() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize(
        "Which segment is the least represented in sales data?",
        build_dataset(),
        build_profile(),
        None,
    )

    assert warnings == []
    assert request.intent_name == "representation_ranking"
    assert request.target == "segment"
    assert request.group_by == ["segment"]
    assert request.aggregation == "count"
    assert request.options["ranking_direction"] == "asc"


def test_normalizer_detects_column_inventory_intent() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize("What are the columns in the sales data?", build_dataset(), build_profile(), None)

    assert warnings == []
    assert request.intent_name == "column_inventory"
    assert request.target is None


def test_normalizer_detects_time_coverage_years_intent() -> None:
    normalizer = RequestNormalizer()

    request, warnings = normalizer.normalize(
        "The data shows revenue for which years?",
        build_dataset(),
        build_profile(),
        None,
    )

    assert warnings == []
    assert request.intent_name == "time_coverage"
    assert request.target is None
    assert request.options["time_coverage_mode"] == "years_present"


def test_normalizer_detects_time_coverage_months_intent() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize(
        "What months are present in the sales data?",
        build_dataset(),
        build_profile(),
        None,
    )

    assert request.intent_name == "time_coverage"
    assert request.options["time_coverage_mode"] == "months_present"


def test_normalizer_detects_time_coverage_date_range_intent() -> None:
    normalizer = RequestNormalizer()

    request, _ = normalizer.normalize(
        "What date range does the sales data cover?",
        build_dataset(),
        build_profile(),
        None,
    )

    assert request.intent_name == "time_coverage"
    assert request.options["time_coverage_mode"] == "date_range"


_NORMALIZER_QUESTION_CASES = [
    (
        f"Show revenue by region for West in march case {index}",
        "descriptive",
        "revenue",
        ["region"],
    )
    for index in range(1, 45)
] + [
    (
        f"Why did revenue drop in March by region case {index}",
        "diagnostic",
        "revenue",
        ["region"],
    )
    for index in range(45, 89)
]


@pytest.mark.parametrize(("question", "task_type", "expected_target", "expected_group_by"), _NORMALIZER_QUESTION_CASES)
def test_normalizer_handles_many_supported_question_shapes(
    question: str,
    task_type: str,
    expected_target: str,
    expected_group_by: list[str],
) -> None:
    normalizer = RequestNormalizer()
    request, _ = normalizer.normalize(question, build_dataset(), build_profile(), None)

    assert request.task_type_hint == task_type
    assert request.target == expected_target
    assert request.group_by == expected_group_by
