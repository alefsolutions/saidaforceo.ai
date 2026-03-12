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
        if target is not None:
            self._require_columns(prepared, [target])
        metrics = [
            Metric(name="row_count", value=int(len(prepared)), description="Number of rows in the dataset."),
            Metric(name="column_count", value=int(len(prepared.columns)), description="Number of columns in the dataset."),
        ]
        if target:
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
        self._require_columns(prepared, [target, time_column])
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared = prepared.dropna(subset=[time_column])
        prepared["period_month"] = prepared[time_column].dt.to_period("M").astype(str)
        query = """
            with monthly_totals as (
                select
                    period_month,
                    sum(target_value) as target_total
                from prepared
                group by period_month
            )
            select
                period_month,
                target_total,
                target_total - lag(target_total) over(order by period_month) as period_delta
            from monthly_totals
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

    def grouped_period_comparison(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        time_column: str,
        time_reference: dict[str, str],
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Compare grouped totals between adjacent periods."""
        prepared = self._apply_filters(dataframe, filters).copy()
        self._require_columns(prepared, [target, time_column, *group_by])
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared = prepared.dropna(subset=[time_column, target])
        prepared["period_month"] = prepared[time_column].dt.to_period("M")

        periods = self._resolve_adjacent_periods(prepared, time_reference)
        if periods is None:
            return TableArtifact(
                name="grouped_period_comparison",
                description="No comparable grouped periods could be derived from the request.",
                dataframe=pd.DataFrame(columns=[*group_by, "previous_total", "current_total", "delta", "pct_change"]),
            )

        current_period, previous_period = periods
        current_slice = prepared.loc[prepared["period_month"] == current_period]
        previous_slice = prepared.loc[prepared["period_month"] == previous_period]

        current_grouped = current_slice.groupby(group_by, dropna=False)[target].sum().reset_index(name="current_total")
        previous_grouped = previous_slice.groupby(group_by, dropna=False)[target].sum().reset_index(name="previous_total")

        comparison = current_grouped.merge(previous_grouped, on=group_by, how="outer").fillna(0.0)
        comparison["delta"] = comparison["current_total"] - comparison["previous_total"]
        comparison["pct_change"] = comparison.apply(
            lambda row: float(row["delta"] / row["previous_total"]) if row["previous_total"] else 0.0,
            axis=1,
        )
        comparison = comparison.sort_values("delta")

        return TableArtifact(
            name="grouped_period_comparison",
            description=f"Grouped period comparison for {target}.",
            dataframe=comparison.reset_index(drop=True),
        )

    def top_movers(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        time_column: str,
        time_reference: dict[str, str],
        filters: dict[str, str] | None = None,
        limit: int = 5,
    ) -> TableArtifact:
        """Return the largest grouped movers between adjacent periods."""
        comparison = self.grouped_period_comparison(
            dataframe=dataframe,
            target=target,
            group_by=group_by,
            time_column=time_column,
            time_reference=time_reference,
            filters=filters,
        ).dataframe.copy()

        if comparison.empty:
            return TableArtifact(
                name="top_movers",
                description=f"No movers were available for {target}.",
                dataframe=comparison,
            )

        comparison["abs_delta"] = comparison["delta"].abs()
        comparison = comparison.sort_values("abs_delta", ascending=False).head(limit).copy()
        comparison["rank"] = range(1, len(comparison) + 1)

        ordered_columns = ["rank", *group_by, "previous_total", "current_total", "delta", "pct_change", "abs_delta"]
        comparison = comparison.loc[:, [column for column in ordered_columns if column in comparison.columns]]
        return TableArtifact(
            name="top_movers",
            description=f"Top {limit} movers for {target}.",
            dataframe=comparison.reset_index(drop=True),
        )

    def group_breakdown(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Aggregate a target by one or more grouping columns."""
        prepared = self._apply_filters(dataframe, filters)
        self._require_columns(prepared, [target, *group_by])
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

    def ranked_breakdown(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        filters: dict[str, str] | None = None,
        limit: int = 5,
    ) -> TableArtifact:
        """Return the top grouped contributors by target total."""
        grouped = self.group_breakdown(dataframe, target, group_by, filters).dataframe.head(limit).copy()
        grouped["rank"] = range(1, len(grouped) + 1)
        ordered_columns = ["rank", *group_by, "target_total"]
        ranked = grouped.loc[:, [column for column in ordered_columns if column in grouped.columns]]
        return TableArtifact(
            name="ranked_breakdown",
            description=f"Top {limit} grouped contributors for {target}.",
            dataframe=ranked,
        )

    def contribution_breakdown(
        self,
        dataframe: pd.DataFrame,
        target: str,
        group_by: list[str],
        time_column: str | None = None,
        time_reference: dict[str, str] | None = None,
        filters: dict[str, str] | None = None,
    ) -> TableArtifact:
        """Measure group contribution deltas between adjacent periods when possible."""
        self._require_columns(dataframe, [target, *group_by])
        if time_column is None or time_reference is None:
            grouped = self.group_breakdown(dataframe, target, group_by, filters).dataframe.copy()
            total = float(grouped["target_total"].sum()) if not grouped.empty else 0.0
            grouped["share_of_total"] = grouped["target_total"].apply(lambda value: float(value / total) if total else 0.0)
            return TableArtifact(
                name="contribution_breakdown",
                description=f"Share of total {target} by group.",
                dataframe=grouped,
            )

        prepared = self._apply_filters(dataframe, filters).copy()
        self._require_columns(prepared, [time_column])
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared = prepared.dropna(subset=[time_column, target])
        prepared["period_month"] = prepared[time_column].dt.to_period("M")

        if time_reference.get("type") != "month_name":
            return self.contribution_breakdown(
                dataframe,
                target,
                group_by,
                time_column=None,
                time_reference=None,
                filters=filters,
            )

        requested_month = int(time_reference["month"])
        matching_periods = prepared.loc[prepared["period_month"].dt.month == requested_month, "period_month"].sort_values()
        if matching_periods.empty:
            return TableArtifact(
                name="contribution_breakdown",
                description="No rows matched the requested period for contribution analysis.",
                dataframe=pd.DataFrame(columns=[*group_by, "previous_total", "current_total", "delta"]),
            )

        current_period = matching_periods.iloc[-1]
        previous_period = current_period - 1

        current_slice = prepared.loc[prepared["period_month"] == current_period]
        previous_slice = prepared.loc[prepared["period_month"] == previous_period]

        current_grouped = current_slice.groupby(group_by, dropna=False)[target].sum().reset_index(name="current_total")
        previous_grouped = previous_slice.groupby(group_by, dropna=False)[target].sum().reset_index(name="previous_total")

        merged = current_grouped.merge(previous_grouped, on=group_by, how="outer").fillna(0.0)
        merged["delta"] = merged["current_total"] - merged["previous_total"]
        merged = merged.sort_values("delta")

        return TableArtifact(
            name="contribution_breakdown",
            description=f"Contribution deltas for {target} between adjacent periods.",
            dataframe=merged.reset_index(drop=True),
        )

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
        self._require_columns(prepared, [target, time_column])
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

        periods = self._resolve_adjacent_periods(prepared, time_reference)
        if periods is None:
            empty = pd.DataFrame(columns=["period", "target_total"])
            return TableArtifact(
                name="period_comparison",
                description="No rows matched the requested period.",
                dataframe=empty,
            )
        current_period, previous_period = periods

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

    def _resolve_adjacent_periods(
        self,
        dataframe: pd.DataFrame,
        time_reference: dict[str, str],
    ) -> tuple[pd.Period, pd.Period] | None:
        if time_reference.get("type") != "month_name":
            return None

        requested_month = int(time_reference["month"])
        matching_periods = dataframe.loc[dataframe["period_month"].dt.month == requested_month, "period_month"].sort_values()
        if matching_periods.empty:
            return None

        current_period = matching_periods.iloc[-1]
        previous_period = current_period - 1
        return current_period, previous_period

    def _apply_filters(self, dataframe: pd.DataFrame, filters: dict[str, str] | None) -> pd.DataFrame:
        if not filters:
            return dataframe

        prepared = dataframe.copy()
        for column_name, expected_value in filters.items():
            if column_name not in prepared.columns:
                raise ComputeError(f"Filter column '{column_name}' does not exist in the dataset.")

            series = prepared[column_name]
            if pd.api.types.is_string_dtype(series):
                prepared = prepared.loc[series.astype(str).str.lower() == expected_value.lower()]
            else:
                prepared = prepared.loc[series.astype(str) == str(expected_value)]

        if prepared.empty:
            raise ComputeError("Filters removed all rows from the dataset.")
        return prepared

    def _require_columns(self, dataframe: pd.DataFrame, column_names: list[str]) -> None:
        missing_columns = [column_name for column_name in column_names if column_name not in dataframe.columns]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ComputeError(f"Required columns are missing from the dataset: {joined}")
