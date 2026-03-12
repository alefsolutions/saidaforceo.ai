from __future__ import annotations

import pandas as pd

from saida.reasoning import ResultSummarizer
from saida.schemas import (
    AnalysisPlan,
    AnalysisRequest,
    ColumnProfile,
    DatasetProfile,
    Metric,
    SourceContext,
    TableArtifact,
)


def build_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="sales",
        row_count=4,
        column_count=3,
        columns=[
            ColumnProfile(
                name="revenue",
                inferred_type="float",
                nullable=False,
                null_ratio=0.0,
                unique_count=4,
                distinct_ratio=1.0,
                sample_values=[100.0],
                is_measure_candidate=True,
            )
        ],
        measure_columns=["revenue"],
        dimension_columns=["region"],
        time_columns=["posted_at"],
        identifier_columns=[],
    )


def test_summarizer_describes_share_of_total_and_freshness_note() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="descriptive", rationale="Test.")
    request = AnalysisRequest(question="Show revenue by region", task_type_hint="descriptive", target="revenue")
    metrics = [Metric(name="row_count", value=4), Metric(name="revenue_sum", value=400.0)]
    tables = [
        TableArtifact(
            name="time_trend",
            description="Trend.",
            dataframe=pd.DataFrame(
                {
                    "period_month": ["2026-02", "2026-03"],
                    "target_total": [180.0, 220.0],
                    "period_delta": [None, 40.0],
                }
            ),
        ),
        TableArtifact(
            name="ranked_breakdown",
            description="Ranking.",
            dataframe=pd.DataFrame({"rank": [1], "region": ["West"], "target_total": [250.0]}),
        ),
        TableArtifact(
            name="contribution_breakdown",
            description="Contribution.",
            dataframe=pd.DataFrame(
                {"region": ["West", "East"], "target_total": [250.0, 150.0], "share_of_total": [0.625, 0.375]}
            ),
        ),
        TableArtifact(
            name="anomaly_summary",
            description="Anomalies.",
            dataframe=pd.DataFrame(columns=["observation", "target_value", "z_score"]),
        ),
    ]
    context = SourceContext(raw_markdown="", freshness_notes=["source refreshes daily"])

    summary = summarizer.summarize(plan, metrics, tables, [], request, build_profile(), context)

    assert "Completed a descriptive analysis for revenue on sales." in summary
    assert "The latest period is 2026-03 with revenue at 220.00 and a period change of +40.00." in summary
    assert "Top contributor was region=West with revenue total of 250.00." in summary
    assert "Largest share of total revenue came from region=West at 62.5%." in summary
    assert "Detected 0 anomaly candidates." in summary
    assert "Context freshness note: source refreshes daily." in summary


def test_summarizer_includes_warning_text() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="diagnostic", rationale="Test.")
    request = AnalysisRequest(question="Why?", task_type_hint="diagnostic", target="revenue")

    summary = summarizer.summarize(
        plan,
        metrics=[],
        tables=[],
        warnings=["filter was narrow"],
        request=request,
        profile=build_profile(),
        context=None,
    )

    assert "Warnings: filter was narrow." in summary
