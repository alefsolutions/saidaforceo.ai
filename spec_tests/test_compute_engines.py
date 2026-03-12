from __future__ import annotations

import pandas as pd

from saida.compute.duckdb import DuckDBComputeEngine
from saida.compute.stats import StatsComputeEngine


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

    assert list(period_table.dataframe["period"]) == ["2026-02", "2026-03"]
    assert "delta" in contribution_table.dataframe.columns


def test_duckdb_ranked_breakdown_respects_limit() -> None:
    engine = DuckDBComputeEngine()
    dataframe = build_dataframe()

    ranked = engine.ranked_breakdown(dataframe, target="revenue", group_by=["region"], limit=1)

    assert len(ranked.dataframe) == 1
    assert ranked.dataframe.iloc[0]["rank"] == 1


def test_stats_engine_returns_correlation_and_anomalies() -> None:
    engine = StatsComputeEngine()
    dataframe = build_dataframe()

    correlation_table = engine.correlation_matrix(dataframe, target="revenue")
    anomaly_table = engine.anomaly_summary(dataframe, target="revenue", time_column="posted_at")

    assert correlation_table is not None
    assert "correlation" in correlation_table.dataframe.columns
    assert anomaly_table is not None
    assert len(anomaly_table.dataframe) >= 1
