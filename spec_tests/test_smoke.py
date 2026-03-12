from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from saida import Saida
from saida.adapters import CSVAdapter, JSONAdapter, PandasAdapter, SQLAdapter
from saida.context import SourceContextParser
from saida.exceptions import ModelTrainingError
from saida.profiling import DatasetProfiler
from saida.schemas import Dataset


def test_context_parser_extracts_metrics_and_rules() -> None:
    markdown = """
# Dataset: Sales

## Metrics
revenue = total invoice value

## Important Rules
- cancelled orders must be excluded
""".strip()

    context = SourceContextParser().parse(markdown)

    assert context.metric_definitions["revenue"] == "total invoice value"
    assert context.business_rules == ["cancelled orders must be excluded"]


def test_context_parser_supports_all_documented_sections() -> None:
    markdown = """
# Dataset: Sales

## Table Descriptions
orders: order-level sales facts

## Field Descriptions
- posted_at: posting timestamp
- customer_id: customer identifier

## Metric Definitions
revenue: total invoice value after discounts

## Business Rules
- cancelled orders must be excluded

## Caveats
- refunds arrive one day late

## Trusted Date Fields
- posted_at
- settled_at

## Preferred Identifiers
- customer_id

## Freshness Notes
- source refreshes daily
""".strip()

    context = SourceContextParser().parse(markdown)

    assert context.source_summary == "Sales"
    assert context.table_descriptions["orders"] == "order-level sales facts"
    assert context.field_descriptions["posted_at"] == "posting timestamp"
    assert context.metric_definitions["revenue"] == "total invoice value after discounts"
    assert context.business_rules == ["cancelled orders must be excluded"]
    assert context.caveats == ["refunds arrive one day late"]
    assert context.trusted_date_fields == ["posted_at", "settled_at"]
    assert context.preferred_identifiers == ["customer_id"]
    assert context.freshness_notes == ["source refreshes daily"]


def test_context_parser_extracts_field_sections() -> None:
    markdown = """
# Summary
sales dataset

## Field Descriptions
revenue = total revenue
region = sales region

## Table Descriptions
orders = order-level facts
""".strip()

    context = SourceContextParser().parse(markdown)

    assert context.source_summary == "sales dataset"
    assert context.field_descriptions["revenue"] == "total revenue"
    assert context.table_descriptions["orders"] == "order-level facts"


def test_analyze_runs_end_to_end(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "posted_at,revenue,region\n"
        "2026-01-01,100,West\n"
        "2026-02-01,90,West\n"
        "2026-03-01,80,East\n",
        encoding="utf-8",
    )

    dataset = CSVAdapter(csv_path).load()
    result = Saida().analyze(dataset, "Why did revenue drop in March?")

    assert result.summary
    assert any(metric.name == "row_count" for metric in result.metrics)
    assert any(table.name == "time_trend" for table in result.tables)
    assert any(table.name == "numeric_summary" for table in result.tables)
    assert any(table.name == "period_comparison" for table in result.tables)
    assert any(table.name == "contribution_breakdown" for table in result.tables)
    assert any(table.name == "ranked_breakdown" for table in result.tables)


def test_json_adapter_loads_records(tmp_path: Path) -> None:
    json_path = tmp_path / "sales.json"
    json_path.write_text('[{"revenue": 100, "region": "West"}]', encoding="utf-8")

    dataset = JSONAdapter(json_path).load()

    assert dataset.source_type == "json"
    assert list(dataset.data.columns) == ["revenue", "region"]


def test_sql_adapter_loads_query_results(tmp_path: Path) -> None:
    database_path = tmp_path / "sales.db"
    connection = sqlite3.connect(database_path)
    connection.execute("create table sales (revenue integer, region text)")
    connection.execute("insert into sales (revenue, region) values (100, 'West')")
    connection.commit()
    connection.close()

    dataset = SQLAdapter(database_path, "select revenue, region from sales").load()

    assert dataset.source_type == "sql"
    assert dataset.data.iloc[0]["revenue"] == 100


def test_pandas_adapter_keeps_context() -> None:
    dataframe = pd.DataFrame({"revenue": [100, 120], "region": ["West", "East"]})
    adapter = PandasAdapter(
        dataframe,
        name="sales",
        context_markdown="""
# Dataset: Sales

## Metrics
revenue = total invoice value
""".strip(),
    )

    dataset = adapter.load()

    assert dataset.name == "sales"
    assert dataset.context is not None
    assert dataset.context.metric_definitions["revenue"] == "total invoice value"


def test_analyze_applies_group_and_filter_detection() -> None:
    dataframe = pd.DataFrame(
        {
            "posted_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "revenue": [100, 50, 25],
            "region": ["West", "East", "West"],
        }
    )
    dataset = Dataset(name="sales", source_type="pandas", data=dataframe)

    result = Saida().analyze(dataset, "Show revenue by region for West")

    grouped_tables = [table for table in result.tables if table.name == "group_breakdown"]
    assert grouped_tables
    grouped = grouped_tables[0].dataframe
    assert set(grouped["region"]) == {"West"}


def test_analyze_returns_ranked_breakdown_and_contribution_tables() -> None:
    dataframe = pd.DataFrame(
        {
            "posted_at": [
                "2026-02-01",
                "2026-02-01",
                "2026-03-01",
                "2026-03-01",
            ],
            "revenue": [120, 80, 60, 40],
            "region": ["West", "East", "West", "East"],
        }
    )
    dataset = Dataset(name="sales", source_type="pandas", data=dataframe)

    result = Saida().analyze(dataset, "Why did revenue drop in March by region?")

    ranked_table = next(table for table in result.tables if table.name == "ranked_breakdown")
    contribution_table = next(table for table in result.tables if table.name == "contribution_breakdown")
    grouped_period_table = next(table for table in result.tables if table.name == "grouped_period_comparison")
    mover_table = next(table for table in result.tables if table.name == "top_movers")

    assert "rank" in ranked_table.dataframe.columns
    assert "delta" in contribution_table.dataframe.columns
    assert "pct_change" in grouped_period_table.dataframe.columns
    assert "abs_delta" in mover_table.dataframe.columns
    assert not contribution_table.dataframe.empty


def test_analyze_returns_anomaly_summary_for_outlier_series() -> None:
    dataframe = pd.DataFrame(
        {
            "posted_at": [
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "2026-04-01",
                "2026-05-01",
            ],
            "revenue": [100, 105, 98, 102, 300],
            "region": ["West", "West", "West", "West", "West"],
        }
    )
    dataset = Dataset(name="sales", source_type="pandas", data=dataframe)

    result = Saida().analyze(dataset, "Show revenue trend")

    anomaly_table = next(table for table in result.tables if table.name == "anomaly_summary")
    assert len(anomaly_table.dataframe) >= 1


def test_train_forecast_and_predict_raise_not_implemented() -> None:
    dataframe = pd.DataFrame(
        {
            "posted_at": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "sales": [100, 110, 120],
        }
    )
    dataset = Dataset(name="sales", source_type="pandas", data=dataframe)
    engine = Saida()

    with pytest.raises(ModelTrainingError):
        engine.train(dataset, target="sales")

    with pytest.raises(ModelTrainingError):
        engine.predict(dataset, artifact_path="model.json")

    with pytest.raises(ModelTrainingError):
        engine.forecast(dataset, target="sales", horizon=2)


def test_profiler_detects_identifiers_dimensions_and_measures() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "region": ["West", "West", "East", "East"],
            "revenue": [100.0, 120.0, 80.0, 90.0],
            "posted_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    dataset = Dataset(name="sales", source_type="pandas", data=dataframe)

    profile = DatasetProfiler().profile(dataset)

    assert "customer_id" in profile.identifier_columns
    assert "region" in profile.dimension_columns
    assert "revenue" in profile.measure_columns
    assert "posted_at" in profile.time_columns


def test_profiler_marks_low_cardinality_strings_as_category() -> None:
    dataframe = pd.DataFrame(
        {
            "segment": ["SMB", "SMB", "Enterprise", "Enterprise", "SMB"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    dataset = Dataset(name="segments", source_type="pandas", data=dataframe)

    profile = DatasetProfiler().profile(dataset)
    segment_profile = next(column for column in profile.columns if column.name == "segment")

    assert segment_profile.inferred_type == "category"
    assert segment_profile.is_dimension_candidate is True


def test_profiler_warns_on_duplicates_and_limited_readiness() -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "flag": ["yes", "yes", "no"],
        }
    )
    dataset = Dataset(name="small", source_type="pandas", data=dataframe)

    profile = DatasetProfiler().profile(dataset)

    assert profile.duplicate_row_count == 1
    assert "Dataset contains duplicate rows." in profile.warnings
    assert profile.ml_readiness is not None
    assert profile.ml_readiness.forecasting_ready is False
    assert any("No time column" in warning for warning in profile.ml_readiness.readiness_warnings)
