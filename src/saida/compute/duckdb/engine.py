"""DuckDB-backed deterministic analytics."""

from __future__ import annotations

import duckdb
import pandas as pd

from saida.exceptions import ComputeError
from saida.schemas import Metric, TableArtifact


class DuckDBComputeEngine:
    """Run deterministic analytical queries against a pandas DataFrame."""

    def dataset_summary(
        self,
        dataframe: pd.DataFrame,
        target: str | None,
        filters: dict[str, str] | None = None,
    ) -> tuple[list[Metric], list[TableArtifact]]:
        """Compute top-level metrics for the dataset."""
        prepared = self._apply_filters(dataframe, filters)
        metrics = [
            Metric(name="row_count", value=int(len(prepared)), description="Number of rows in the dataset."),
            Metric(name="column_count", value=int(len(prepared.columns)), description="Number of columns in the dataset."),
        ]
        if target and target in prepared.columns:
            try:
                connection = duckdb.connect()
                connection.register("source_df", prepared)
                value = connection.execute(f'select sum("{target}") as total_value from source_df').fetchone()[0]
                connection.close()
            except Exception as exc:  # pragma: no cover
                raise ComputeError(f"Failed to compute summary metric for target '{target}'.") from exc
            metrics.append(Metric(name=f"{target}_sum", value=float(value or 0.0), description=f"Sum of {target}."))

        preview = prepared.head(10).copy()
        tables = [TableArtifact(name="dataset_preview", description="First 10 rows of the dataset.", dataframe=preview)]
        return metrics, tables

    def time_trend(
        self,
        dataframe: pd.DataFrame,
        target: str,
        time_column: str,
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Aggregate a target over monthly time buckets."""
        prepared = self._apply_filters(dataframe, filters).copy()
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared = prepared.dropna(subset=[time_column])
        prepared["period_month"] = prepared[time_column].dt.to_period("M").astype(str)
        query = """
            select
                period_month,
                sum(target_value) as target_total
            from prepared
            group by period_month
            order by period_month
        """
        try:
            connection = duckdb.connect()
            connection.register("prepared", prepared.assign(target_value=prepared[target]))
            trend = connection.execute(query).fetchdf()
            connection.close()
        except Exception as exc:  # pragma: no cover
            raise ComputeError(f"Failed to compute time trend for target '{target}'.") from exc
        return TableArtifact(name="time_trend", description=f"Monthly trend for {target}.", dataframe=trend)

    def group_breakdown(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Aggregate a target by one or more grouping columns."""
        prepared = self._apply_filters(dataframe, filters)
        group_column_sql = ", ".join(group_by)
        query = f"""
            select
                {group_column_sql},
                sum(target_value) as target_total
            from prepared
            group by {group_column_sql}
            order by target_total desc
        """
        try:
            connection = duckdb.connect()
            connection.register("prepared", prepared.assign(target_value=prepared[target]))
            grouped = connection.execute(query).fetchdf()
            connection.close()
        except Exception as exc:  # pragma: no cover
            raise ComputeError(f"Failed to compute grouped breakdown for target '{target}'.") from exc
        return TableArtifact(name="group_breakdown", description=f"Grouped breakdown for {target}.", dataframe=grouped)

    def period_comparison(
        self,
        dataframe: pd.DataFrame,
        target: str,
        time_column: str,
        time_reference: dict[str, str],
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Compare a selected period against the immediately previous comparable period."""
        prepared = self._apply_filters(dataframe, filters).copy()
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared = prepared.dropna(subset=[time_column, target])
        prepared["period_month"] = prepared[time_column].dt.to_period("M")

        if time_reference.get("type") != "month_name":
            empty = pd.DataFrame(columns=["period", "target_total"])
            return TableArtifact(
                name="period_comparison",
                description="No comparable period could be derived from the request.",
                dataframe=empty,
            )

        requested_month = int(time_reference["month"])
        matching_periods = prepared.loc[prepared["period_month"].dt.month == requested_month, "period_month"].sort_values()
        if matching_periods.empty:
            empty = pd.DataFrame(columns=["period", "target_total"])
            return TableArtifact(
                name="period_comparison",
                description="No rows matched the requested period.",
                dataframe=empty,
            )

        current_period = matching_periods.iloc[-1]
        previous_period = current_period - 1

        current_total = prepared.loc[prepared["period_month"] == current_period, target].sum()
        previous_total = prepared.loc[prepared["period_month"] == previous_period, target].sum()

        comparison = pd.DataFrame(
            {
                "period": [str(previous_period), str(current_period)],
                "target_total": [float(previous_total), float(current_total)],
            }
        )
        comparison["delta"] = comparison["target_total"].diff()

        return TableArtifact(
            name="period_comparison",
            description=f"Comparison for {target} across adjacent periods.",
            dataframe=comparison,
        )

    def _apply_filters(self, dataframe: pd.DataFrame, filters: dict[str, str] | None) -> pd.DataFrame:
        if not filters:
            return dataframe

        prepared = dataframe.copy()
        for column_name, expected_value in filters.items():
            if column_name not in prepared.columns:
                continue

            series = prepared[column_name]
            if pd.api.types.is_string_dtype(series):
                prepared = prepared.loc[series.astype(str).str.lower() == expected_value.lower()]
            else:
                prepared = prepared.loc[series.astype(str) == str(expected_value)]

        return prepared
