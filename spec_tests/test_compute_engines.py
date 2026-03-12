from __future__ import annotations

import pandas as pd
import pytest

from saida.compute.duckdb import DuckDBComputeEngine
from saida.compute.stats import StatsComputeEngine
from saida.exceptions import ComputeError


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "posted_at": ["2026-02-01", "2026-02-01", "2026-03-01", "2026-03-01", "2026-04-01"],
            "revenue": [120.0, 80.0, 60.0, 40.0, 300.0],
            "region": ["West", "East", "West", "East", "West"],
            "cost": [70.0, 50.0, 40.0, 20.0, 100.0],
        }
    )


def test_duckdb_period_and_contribution_breakdown() -> None:
    engine = DuckDBComputeEngine()
    dataframe = build_dataframe()

    trend_table = engine.time_trend(
        dataframe,
        target="revenue",
        time_column="posted_at",
    )
    period_table = engine.period_comparison(
        dataframe,
        target="revenue",
        time_column="posted_at",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
    )
    contribution_table = engine.contribution_breakdown(
        dataframe,
        target="revenue",
        group_by=["region"],
        time_column="posted_at",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
    )

    assert "period_delta" in trend_table.dataframe.columns
    assert list(period_table.dataframe["period"]) == ["2026-02", "2026-03"]
    assert "delta" in contribution_table.dataframe.columns


def test_duckdb_ranked_breakdown_respects_limit() -> None:
    engine = DuckDBComputeEngine()
    dataframe = build_dataframe()

    ranked = engine.ranked_breakdown(dataframe, target="revenue", group_by=["region"], limit=1)

    assert len(ranked.dataframe) == 1
    assert ranked.dataframe.iloc[0]["rank"] == 1


def test_duckdb_grouped_period_comparison_and_top_movers() -> None:
    engine = DuckDBComputeEngine()
    dataframe = build_dataframe()

    grouped_comparison = engine.grouped_period_comparison(
        dataframe,
        target="revenue",
        group_by=["region"],
        time_column="posted_at",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
    )
    movers = engine.top_movers(
        dataframe,
        target="revenue",
        group_by=["region"],
        time_column="posted_at",
        time_reference={"type": "month_name", "value": "march", "month": "3"},
        limit=2,
    )

    assert "pct_change" in grouped_comparison.dataframe.columns
    assert "abs_delta" in movers.dataframe.columns
    assert len(movers.dataframe) <= 2


def test_stats_engine_returns_correlation_and_anomalies() -> None:
    engine = StatsComputeEngine()
    dataframe = build_dataframe()

    distribution_table = engine.distribution_summary(dataframe, target="revenue")
    correlation_table = engine.correlation_matrix(dataframe, target="revenue")
    anomaly_table = engine.anomaly_summary(dataframe, target="revenue", time_column="posted_at")
    comparison_table = engine.group_mean_comparison(dataframe, target="revenue", group_column="region")
    diagnostics_table = engine.time_series_diagnostics(dataframe, target="revenue", time_column="posted_at")

    assert distribution_table is not None
    assert "skewness" in distribution_table.dataframe.columns
    assert correlation_table is not None
    assert "correlation" in correlation_table.dataframe.columns
    assert anomaly_table is not None
    assert len(anomaly_table.dataframe) >= 1
    assert comparison_table is not None
    assert "p_value" in comparison_table.dataframe.columns
    assert diagnostics_table is not None
    assert "lag1_autocorrelation" in diagnostics_table.dataframe.columns


def test_duckdb_engine_rejects_invalid_filters() -> None:
    engine = DuckDBComputeEngine()

    with pytest.raises(ComputeError, match="Filter column 'missing' does not exist"):
        engine.dataset_summary(build_dataframe(), target="revenue", filters={"missing": "West"})


def test_duckdb_engine_rejects_filters_that_remove_all_rows() -> None:
    engine = DuckDBComputeEngine()

    with pytest.raises(ComputeError, match="Filters removed all rows"):
        engine.dataset_summary(build_dataframe(), target="revenue", filters={"region": "North"})


def test_stats_engine_rejects_missing_target_columns() -> None:
    engine = StatsComputeEngine()

    with pytest.raises(ComputeError, match="Target column 'profit' does not exist"):
        engine.distribution_summary(build_dataframe(), target="profit")
