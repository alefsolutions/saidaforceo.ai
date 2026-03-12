"""Deterministic statistical routines."""

from __future__ import annotations

import pandas as pd

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

    def correlation_matrix(self, dataframe: pd.DataFrame, target: str | None = None) -> TableArtifact | None:
        """Return a correlation table for numeric columns."""
        numeric_columns = dataframe.select_dtypes(include=["number"])
        if numeric_columns.shape[1] < 2:
            return None

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

    def anomaly_summary(self, dataframe: pd.DataFrame, target: str, time_column: str | None = None) -> TableArtifact | None:
        """Flag simple z-score anomalies for a target series."""
        if target not in dataframe.columns:
            return None

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
