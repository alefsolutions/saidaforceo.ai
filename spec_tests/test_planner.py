from __future__ import annotations

import pytest

from saida.planning import AnalysisPlanner
from saida.exceptions import PlanningError
from saida.schemas import AnalysisRequest, ColumnProfile, DatasetProfile


def build_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="sales",
        row_count=10,
        column_count=4,
        columns=[
            ColumnProfile(
                name="revenue",
                inferred_type="float",
                nullable=False,
                null_ratio=0.0,
                unique_count=10,
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
                distinct_ratio=0.2,
                sample_values=["West"],
                is_dimension_candidate=True,
            ),
            ColumnProfile(
                name="posted_at",
                inferred_type="datetime",
                nullable=False,
                null_ratio=0.0,
                unique_count=10,
                distinct_ratio=1.0,
                sample_values=["2026-03-01"],
                is_time_candidate=True,
            ),
        ],
        measure_columns=["revenue"],
        dimension_columns=["region"],
        time_columns=["posted_at"],
        identifier_columns=[],
    )


def test_planner_builds_diagnostic_plan_with_contribution_steps() -> None:
    planner = AnalysisPlanner()
    request = AnalysisRequest(
        question="Why did revenue drop in March?",
        task_type_hint="diagnostic",
        target="revenue",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
    )

    plan = planner.build_plan(request, build_profile())
    actions = [step.action for step in plan.steps]

    assert "dataset_summary" in actions
    assert "period_comparison" in actions
    assert "contribution_breakdown" in actions
    assert "anomaly_summary" in actions


def test_planner_builds_grouped_descriptive_plan() -> None:
    planner = AnalysisPlanner()
    request = AnalysisRequest(
        question="Show revenue by region",
        task_type_hint="descriptive",
        target="revenue",
        group_by=["region"],
    )

    plan = planner.build_plan(request, build_profile())
    actions = [step.action for step in plan.steps]

    assert "group_breakdown" in actions
    assert "ranked_breakdown" in actions


def test_planner_rejects_invalid_filter_columns() -> None:
    planner = AnalysisPlanner()
    request = AnalysisRequest(
        question="Show revenue for missing_region=West",
        task_type_hint="descriptive",
        target="revenue",
        filters={"missing_region": "West"},
    )

    with pytest.raises(PlanningError, match="Filter columns do not exist"):
        planner.build_plan(request, build_profile())


def test_planner_rejects_time_request_without_time_column() -> None:
    planner = AnalysisPlanner()
    profile = build_profile()
    profile.time_columns = []
    request = AnalysisRequest(
        question="Why did revenue drop in March?",
        task_type_hint="diagnostic",
        target="revenue",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
    )

    with pytest.raises(PlanningError, match="Time-based analysis requires a datetime column"):
        planner.build_plan(request, profile)


def test_planner_rejects_non_month_time_references_for_non_ml_analysis() -> None:
    planner = AnalysisPlanner()
    request = AnalysisRequest(
        question="Show revenue in Q1",
        task_type_hint="descriptive",
        target="revenue",
        time_reference={"type": "quarter", "value": "q1", "quarter": "1"},
    )

    with pytest.raises(PlanningError, match="Only month-based time references"):
        planner.build_plan(request, build_profile())
