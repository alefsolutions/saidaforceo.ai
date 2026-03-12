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
