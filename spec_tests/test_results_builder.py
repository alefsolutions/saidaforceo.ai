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
