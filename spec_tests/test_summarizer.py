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


def test_summarizer_describes_period_comparison_and_top_mover() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="diagnostic", rationale="Test.")
    request = AnalysisRequest(question="Why did revenue drop?", task_type_hint="diagnostic", target="revenue")
    tables = [
        TableArtifact(
            name="period_comparison",
            description="Period comparison.",
            dataframe=pd.DataFrame(
                {
                    "period": ["2026-02", "2026-03"],
                    "target_total": [200.0, 150.0],
                    "delta": [None, -50.0],
                }
            ),
        ),
        TableArtifact(
            name="top_movers",
            description="Movers.",
            dataframe=pd.DataFrame(
                {
                    "rank": [1],
                    "region": ["West"],
                    "previous_total": [120.0],
                    "current_total": [70.0],
                    "delta": [-50.0],
                    "pct_change": [-0.4166666667],
                    "abs_delta": [50.0],
                }
            ),
        ),
    ]

    summary = summarizer.summarize(plan, [], tables, [], request, build_profile(), None)

    assert "Revenue moved from 200.00 in 2026-02 to 150.00 in 2026-03 (-25.0%)." in summary
    assert "Top mover was region=West with a -50.00 change in revenue (-41.7%)." in summary


def test_summarizer_describes_contribution_delta_branch() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="diagnostic", rationale="Test.")
    request = AnalysisRequest(question="Why did revenue drop?", task_type_hint="diagnostic", target="revenue")
    tables = [
        TableArtifact(
            name="contribution_breakdown",
            description="Contribution.",
            dataframe=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "previous_total": [100.0, 120.0],
                    "current_total": [60.0, 100.0],
                    "delta": [-40.0, -20.0],
                }
            ),
        )
    ]

    summary = summarizer.summarize(plan, [], tables, [], request, build_profile(), None)

    assert "Largest contribution change came from region=East at -40.00 revenue." in summary


def test_summarizer_describes_time_series_diagnostics() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="descriptive", rationale="Test.")
    request = AnalysisRequest(question="Show revenue trend", task_type_hint="descriptive", target="revenue")
    tables = [
        TableArtifact(
            name="time_series_diagnostics",
            description="Diagnostics.",
            dataframe=pd.DataFrame(
                {
                    "first_period": ["2026-01"],
                    "last_period": ["2026-03"],
                    "net_change": [30.0],
                    "change_volatility": [12.5],
                }
            ),
        )
    ]

    summary = summarizer.summarize(plan, [], tables, [], request, build_profile(), None)

    assert "Across 2026-01 to 2026-03, revenue changed by +30.00 with period-to-period volatility of 12.50." in summary


def test_summarizer_uses_dataset_label_when_target_missing() -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="descriptive", rationale="Test.")
    request = AnalysisRequest(question="Show dataset", task_type_hint="descriptive", target=None)

    summary = summarizer.summarize(plan, [], [], [], request, build_profile(), None)

    assert "Completed a descriptive analysis for the dataset on sales." in summary


import pytest


_SUMMARIZER_CASES = [
    (
        index,
        [
            TableArtifact(
                name="time_trend",
                description="Trend.",
                dataframe=pd.DataFrame(
                    {
                        "period_month": ["2026-02", "2026-03"],
                        "target_total": [float(index), float(index + 5)],
                        "period_delta": [None, 5.0],
                    }
                ),
            )
        ],
    )
    for index in range(1, 94)
]


@pytest.mark.parametrize(("case_id", "tables"), _SUMMARIZER_CASES)
def test_summarizer_handles_many_trend_only_cases(case_id: int, tables: list[TableArtifact]) -> None:
    summarizer = ResultSummarizer()
    plan = AnalysisPlan(task_type="descriptive", rationale="Synthetic.")
    request = AnalysisRequest(question=f"Show revenue trend {case_id}", task_type_hint="descriptive", target="revenue")

    summary = summarizer.summarize(
        plan,
        metrics=[Metric(name="row_count", value=2), Metric(name="revenue_sum", value=float(case_id + case_id + 5))],
        tables=tables,
        warnings=[],
        request=request,
        profile=build_profile(),
        context=None,
    )

    assert "Completed a descriptive analysis for revenue on sales." in summary
    assert "The latest period is 2026-03 with revenue at" in summary
