"""Deterministic statistical routines."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from saida.exceptions import ComputeError
from saida.schemas import TableArtifact


class StatsComputeEngine:
    """Run simple statistical routines over a pandas DataFrame."""

    def missingness_summary(self, dataframe: pd.DataFrame) -> TableArtifact:
        """Return a null summary for each column."""
        summary = pd.DataFrame(
            {
                "column": dataframe.columns,
                "null_count": [int(dataframe[column].isna().sum()) for column in dataframe.columns],
                "null_ratio": [float(dataframe[column].isna().mean()) for column in dataframe.columns],
            }
        ).sort_values(["null_ratio", "null_count"], ascending=False)

        return TableArtifact(
            name="missingness_summary",
            description="Null counts and ratios by column.",
            dataframe=summary.reset_index(drop=True),
        )

    def numeric_summary(self, dataframe: pd.DataFrame) -> TableArtifact:
        """Return a summary table for numeric columns."""
        numeric_columns = dataframe.select_dtypes(include=["number"])
        if numeric_columns.empty:
            summary = pd.DataFrame(columns=["column", "count", "mean", "std", "min", "max"])
        else:
            summary = numeric_columns.describe().transpose().reset_index().rename(columns={"index": "column"})

        return TableArtifact(
            name="numeric_summary",
            description="Summary statistics for numeric columns.",
            dataframe=summary,
        )

    def distribution_summary(self, dataframe: pd.DataFrame, target: str) -> TableArtifact | None:
        """Return simple distribution diagnostics for a numeric target."""
        if target not in dataframe.columns:
            raise ComputeError(f"Target column '{target}' does not exist in the dataset.")

        series = pd.to_numeric(dataframe[target], errors="coerce").dropna()
        if series.empty:
            raise ComputeError(f"Target column '{target}' has no numeric values for distribution analysis.")

        summary = pd.DataFrame(
            [
                {
                    "target": target,
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75)),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                }
            ]
        )

        return TableArtifact(
            name="distribution_summary",
            description=f"Distribution summary for {target}.",
            dataframe=summary,
        )

    def correlation_matrix(self, dataframe: pd.DataFrame, target: str | None = None) -> TableArtifact | None:
        """Return a correlation table for numeric columns."""
        numeric_columns = dataframe.select_dtypes(include=["number"])
        if numeric_columns.shape[1] < 2:
            return None
        if target is not None and target not in dataframe.columns:
            raise ComputeError(f"Target column '{target}' does not exist in the dataset.")

        correlations = numeric_columns.corr(numeric_only=True)
        if target and target in correlations.columns:
            target_correlations = (
                correlations[target]
                .drop(labels=[target], errors="ignore")
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "column", target: "correlation"})
            )
            description = f"Correlation of numeric columns against {target}."
            table = target_correlations
            name = "target_correlation"
        else:
            table = correlations.reset_index().rename(columns={"index": "column"})
            description = "Correlation matrix for numeric columns."
            name = "correlation_matrix"

        return TableArtifact(name=name, description=description, dataframe=table)

    def group_mean_comparison(self, dataframe: pd.DataFrame, target: str, group_column: str) -> TableArtifact | None:
        """Compare the target mean across the first two groups in a dimension column."""
        if target not in dataframe.columns or group_column not in dataframe.columns:
            raise ComputeError(f"Columns '{target}' and '{group_column}' are required for group mean comparison.")
        if group_column == target:
            raise ComputeError("Group comparison column must be different from the target column.")

        prepared = dataframe[[target, group_column]].copy()
        prepared[target] = pd.to_numeric(prepared[target], errors="coerce")
        prepared = prepared.dropna(subset=[target, group_column])
        if prepared.empty:
            return None

        grouped_values: list[tuple[str, pd.Series]] = []
        for group_name, group_frame in prepared.groupby(group_column, dropna=False):
            values = group_frame[target]
            if len(values) >= 2:
                grouped_values.append((str(group_name), values))

        if len(grouped_values) < 2:
            return None

        left_name, left_values = grouped_values[0]
        right_name, right_values = grouped_values[1]
        t_stat, p_value = stats.ttest_ind(left_values, right_values, equal_var=False)

        comparison = pd.DataFrame(
            [
                {
                    "group_column": group_column,
                    "left_group": left_name,
                    "right_group": right_name,
                    "left_mean": float(left_values.mean()),
                    "right_mean": float(right_values.mean()),
                    "mean_delta": float(left_values.mean() - right_values.mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                }
            ]
        )

        return TableArtifact(
            name="group_mean_comparison",
            description=f"Mean comparison for {target} across {group_column}.",
            dataframe=comparison,
        )

    def time_series_diagnostics(self, dataframe: pd.DataFrame, target: str, time_column: str) -> TableArtifact | None:
        """Return simple diagnostics for a time-ordered target series."""
        if target not in dataframe.columns or time_column not in dataframe.columns:
            raise ComputeError(f"Columns '{target}' and '{time_column}' are required for time-series diagnostics.")

        prepared = dataframe[[time_column, target]].copy()
        prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
        prepared[target] = pd.to_numeric(prepared[target], errors="coerce")
        prepared = prepared.dropna(subset=[time_column, target]).sort_values(time_column)
        if len(prepared) < 3:
            return None

        prepared["period_month"] = prepared[time_column].dt.to_period("M").astype(str)
        series = prepared.groupby("period_month", as_index=False)[target].sum()
        if len(series) < 3:
            return None

        values = series[target].astype(float)
        diffs = values.diff().dropna()
        diagnostics = pd.DataFrame(
            [
                {
                    "target": target,
                    "period_count": int(len(series)),
                    "first_period": series.iloc[0]["period_month"],
                    "last_period": series.iloc[-1]["period_month"],
                    "first_value": float(values.iloc[0]),
                    "last_value": float(values.iloc[-1]),
                    "net_change": float(values.iloc[-1] - values.iloc[0]),
                    "average_change": float(diffs.mean()) if not diffs.empty else 0.0,
                    "change_volatility": float(diffs.std(ddof=0)) if len(diffs) > 1 else 0.0,
                    "lag1_autocorrelation": float(values.autocorr(lag=1)) if len(values) > 2 else 0.0,
                }
            ]
        )

        return TableArtifact(
            name="time_series_diagnostics",
            description=f"Time-series diagnostics for {target}.",
            dataframe=diagnostics,
        )

    def anomaly_summary(self, dataframe: pd.DataFrame, target: str, time_column: str | None = None) -> TableArtifact | None:
        """Flag simple z-score anomalies for a target series."""
        if target not in dataframe.columns:
            raise ComputeError(f"Target column '{target}' does not exist in the dataset.")

        prepared = dataframe[[target]].copy()
        if time_column and time_column in dataframe.columns:
            prepared[time_column] = pd.to_datetime(dataframe[time_column], errors="coerce")
            prepared = prepared.dropna(subset=[time_column, target]).sort_values(time_column)
            prepared["label"] = prepared[time_column].astype(str)
        else:
            prepared = prepared.dropna(subset=[target])
            prepared["label"] = prepared.index.astype(str)

        if len(prepared) < 3:
            return None

        series = prepared[target].astype(float)
        std = float(series.std(ddof=0))
        if std == 0.0:
            return None

        mean = float(series.mean())
        prepared["z_score"] = (series - mean) / std
        anomalies = prepared.loc[prepared["z_score"].abs() >= 1.8, ["label", target, "z_score"]].copy()
        anomalies = anomalies.rename(columns={"label": "observation", target: "target_value"})

        return TableArtifact(
            name="anomaly_summary",
            description=f"Z-score anomaly candidates for {target}.",
            dataframe=anomalies.reset_index(drop=True),
        )
