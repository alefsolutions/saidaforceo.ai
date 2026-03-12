from __future__ import annotations

import pandas as pd

from saida.results import ResultBuilder
from saida.schemas import (
    AnalysisPlan,
    AnalysisRequest,
    ColumnProfile,
    DatasetProfile,
    ExecutionTraceEvent,
    ForecastResult,
    Metric,
    ModelTrainingResult,
    PlanStep,
    TableArtifact,
)


def build_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="sales",
        row_count=3,
        column_count=3,
        columns=[
            ColumnProfile(
                name="revenue",
                inferred_type="float",
                nullable=False,
                null_ratio=0.0,
                unique_count=3,
                distinct_ratio=1.0,
                sample_values=[100.0],
                is_measure_candidate=True,
            ),
            ColumnProfile(
                name="region",
                inferred_type="category",
                nullable=False,
                null_ratio=0.0,
                unique_count=2,
                distinct_ratio=0.67,
                sample_values=["West"],
                is_dimension_candidate=True,
            ),
            ColumnProfile(
                name="posted_at",
                inferred_type="datetime",
                nullable=False,
                null_ratio=0.0,
                unique_count=3,
                distinct_ratio=1.0,
                sample_values=["2026-03-01"],
                is_time_candidate=True,
            ),
        ],
        measure_columns=["revenue"],
        dimension_columns=["region"],
        time_columns=["posted_at"],
        identifier_columns=[],
        warnings=["profile warning"],
    )


def test_build_analysis_result_populates_artifacts() -> None:
    builder = ResultBuilder()
    plan = AnalysisPlan(
        task_type="diagnostic",
        rationale="Test plan.",
        steps=[
            PlanStep(
                step_id="summary_metrics",
                tool_family="duckdb",
                action="dataset_summary",
                parameters={},
                description="Compute summary metrics.",
            )
        ],
    )
    request = AnalysisRequest(question="Why did revenue drop?", task_type_hint="diagnostic", target="revenue")
    metrics = [Metric(name="row_count", value=3), Metric(name="revenue_sum", value=270.0)]
    tables = [
        TableArtifact(
            name="dataset_preview",
            description="First rows.",
            dataframe=pd.DataFrame({"revenue": [100.0, 90.0]}),
        )
    ]
    trace = [
        ExecutionTraceEvent(stage="adapter", message="loaded"),
        ExecutionTraceEvent(stage="results", message="packaged"),
    ]

    result = builder.build_analysis_result(
        summary="Test summary.",
        metrics=metrics,
        tables=tables,
        warnings=["request warning"],
        plan=plan,
        request=request,
        profile=build_profile(),
        trace=trace,
    )

    assert result.summary == "Test summary."
    assert result.artifacts["request"]["target"] == "revenue"
    assert result.artifacts["metric_lookup"]["revenue_sum"] == 270.0
    assert result.artifacts["table_index"]["dataset_preview"]["rows"] == 2
    assert result.artifacts["profile"]["dataset_name"] == "sales"
    assert result.artifacts["warning_count"] == 1
    assert result.artifacts["trace_stages"] == ["adapter", "results"]
    assert result.artifacts["plan_step_ids"] == ["summary_metrics"]


def test_build_train_and_forecast_results_keep_payloads() -> None:
    builder = ResultBuilder()

    train_result = builder.build_train_result(
        summary="Training placeholder.",
        training=ModelTrainingResult(model_name="baseline", problem_type="regression", target="revenue"),
        trace=[ExecutionTraceEvent(stage="ml", message="placeholder")],
    )
    forecast_result = builder.build_forecast_result(
        summary="Forecast placeholder.",
        forecast=ForecastResult(target="revenue", horizon=3, forecast_values=[1.0, 2.0, 3.0]),
        trace=[ExecutionTraceEvent(stage="ml", message="placeholder")],
    )

    assert train_result.training.model_name == "baseline"
    assert train_result.trace[0].stage == "ml"
    assert forecast_result.forecast.horizon == 3
    assert forecast_result.forecast.forecast_values == [1.0, 2.0, 3.0]


def test_build_analysis_result_handles_empty_metrics_and_tables() -> None:
    builder = ResultBuilder()
    result = builder.build_analysis_result(
        summary="Empty result.",
        metrics=[],
        tables=[],
        warnings=[],
        plan=AnalysisPlan(task_type="descriptive", rationale="Empty."),
        request=AnalysisRequest(question="Show revenue"),
        profile=build_profile(),
        trace=[],
    )

    assert result.artifacts["metric_lookup"] == {}
    assert result.artifacts["table_index"] == {}
    assert result.artifacts["trace_stages"] == []


def test_build_analysis_result_uses_last_metric_value_for_lookup() -> None:
    builder = ResultBuilder()

    result = builder.build_analysis_result(
        summary="Duplicate metric names.",
        metrics=[Metric(name="row_count", value=1), Metric(name="row_count", value=2)],
        tables=[],
        warnings=[],
        plan=AnalysisPlan(task_type="descriptive", rationale="Duplicate metrics."),
        request=AnalysisRequest(question="Show revenue"),
        profile=build_profile(),
        trace=[],
    )

    assert result.artifacts["metric_lookup"]["row_count"] == 2


def test_build_analysis_result_indexes_multiple_tables() -> None:
    builder = ResultBuilder()
    tables = [
        TableArtifact(name="first", description="First table", dataframe=pd.DataFrame({"value": [1]})),
        TableArtifact(name="second", description="Second table", dataframe=pd.DataFrame({"value": [1, 2]})),
    ]

    result = builder.build_analysis_result(
        summary="Multiple tables.",
        metrics=[],
        tables=tables,
        warnings=["warning one", "warning two"],
        plan=AnalysisPlan(task_type="descriptive", rationale="Multiple tables."),
        request=AnalysisRequest(question="Show revenue"),
        profile=build_profile(),
        trace=[ExecutionTraceEvent(stage="results", message="packaged")],
    )

    assert result.artifacts["table_index"]["first"]["rows"] == 1
    assert result.artifacts["table_index"]["second"]["rows"] == 2
    assert result.artifacts["warning_count"] == 2


import pytest


_RESULT_BUILDER_CASES = [
    (
        index,
        [Metric(name="row_count", value=index), Metric(name="revenue_sum", value=float(index * 10))],
        [
            TableArtifact(
                name=f"table_{index}",
                description="Synthetic table.",
                dataframe=pd.DataFrame({"value": list(range(index % 3 + 1))}),
            )
        ],
    )
    for index in range(1, 96)
]


@pytest.mark.parametrize(("case_id", "metrics", "tables"), _RESULT_BUILDER_CASES)
def test_result_builder_handles_many_metric_and_table_shapes(
    case_id: int,
    metrics: list[Metric],
    tables: list[TableArtifact],
) -> None:
    builder = ResultBuilder()
    result = builder.build_analysis_result(
        summary=f"Summary {case_id}",
        metrics=metrics,
        tables=tables,
        warnings=["warning"] if case_id % 2 == 0 else [],
        plan=AnalysisPlan(task_type="descriptive", rationale="Synthetic."),
        request=AnalysisRequest(question=f"Question {case_id}", target="revenue"),
        profile=build_profile(),
        trace=[ExecutionTraceEvent(stage="results", message="packaged")],
    )

    assert result.summary == f"Summary {case_id}"
    assert result.artifacts["metric_lookup"]["row_count"] == case_id
    assert list(result.artifacts["table_index"].keys()) == [f"table_{case_id}"]
