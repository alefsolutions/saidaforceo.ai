"""Deterministic dataset profiling."""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from saida.exceptions import ProfileError
from saida.schemas import ColumnProfile, Dataset, DatasetProfile, MLReadinessProfile


class DatasetProfiler:
    """Build dataset intelligence from a pandas DataFrame."""

    def profile(self, dataset: Dataset) -> DatasetProfile:
        """Create a deterministic profile for a dataset."""
        dataframe = dataset.data
        if dataframe.empty and len(dataframe.columns) == 0:
            raise ProfileError("Dataset contains no columns to profile.")

        columns = [self._profile_column(dataframe, column_name) for column_name in dataframe.columns]
        measure_columns = [column.name for column in columns if column.is_measure_candidate]
        dimension_columns = [column.name for column in columns if column.is_dimension_candidate]
        time_columns = [column.name for column in columns if column.is_time_candidate]
        identifier_columns = [column.name for column in columns if column.is_identifier_candidate]

        warnings: list[str] = []
        if dataframe.empty:
            warnings.append("Dataset contains no rows.")
        if not time_columns:
            warnings.append("No datetime columns detected.")

        ml_readiness = self._build_ml_readiness(dataframe, measure_columns, time_columns)

        return DatasetProfile(
            dataset_name=dataset.name,
            row_count=int(len(dataframe)),
            column_count=int(len(dataframe.columns)),
            columns=columns,
            measure_columns=measure_columns,
            dimension_columns=dimension_columns,
            time_columns=time_columns,
            identifier_columns=identifier_columns,
            duplicate_row_count=int(dataframe.duplicated().sum()),
            warnings=warnings,
            ml_readiness=ml_readiness,
        )

    def _profile_column(self, dataframe: pd.DataFrame, column_name: str) -> ColumnProfile:
        series = dataframe[column_name]
        row_count = len(series)
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        distinct_ratio = (unique_count / row_count) if row_count else None
        inferred_type = self._infer_type(series)

        is_time_candidate = inferred_type in {"datetime", "date"}
        is_measure_candidate = inferred_type in {"integer", "float", "numeric"}
        is_identifier_candidate = unique_count == row_count and row_count > 0 and inferred_type in {
            "string",
            "integer",
            "numeric",
        }
        is_dimension_candidate = inferred_type in {"string", "category", "boolean"} or is_identifier_candidate

        warnings: list[str] = []
        if null_count == row_count:
            warnings.append("Column contains only null values.")

        return ColumnProfile(
            name=column_name,
            inferred_type=inferred_type,
            nullable=null_count > 0,
            null_ratio=(null_count / row_count) if row_count else 0.0,
            unique_count=unique_count,
            distinct_ratio=distinct_ratio,
            sample_values=series.dropna().head(5).tolist(),
            is_identifier_candidate=is_identifier_candidate,
            is_dimension_candidate=is_dimension_candidate,
            is_measure_candidate=is_measure_candidate,
            is_time_candidate=is_time_candidate,
            warnings=warnings,
        )

    def _infer_type(self, series: pd.Series) -> str:
        if is_datetime64_any_dtype(series):
            return "datetime"
        if is_bool_dtype(series):
            return "boolean"
        if is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            if pd.api.types.is_float_dtype(series):
                return "float"
            return "numeric"
        if is_string_dtype(series):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() > 0 and parsed.notna().mean() > 0.8:
                return "datetime"
            return "string"
        return "unknown"

    def _build_ml_readiness(
        self,
        dataframe: pd.DataFrame,
        measure_columns: list[str],
        time_columns: list[str],
    ) -> MLReadinessProfile:
        candidate_targets = list(measure_columns)
        candidate_features = [column for column in dataframe.columns if column not in candidate_targets]
        readiness_warnings: list[str] = []
        if len(dataframe) < 12:
            readiness_warnings.append("Dataset has fewer than 12 rows; ML results may be weak.")

        return MLReadinessProfile(
            candidate_targets=candidate_targets,
            candidate_features=candidate_features,
            forecasting_ready=bool(time_columns and measure_columns and len(dataframe) >= 3),
            regression_ready=bool(measure_columns and candidate_features),
            classification_ready=bool(candidate_features),
            detected_time_column=time_columns[0] if time_columns else None,
            readiness_warnings=readiness_warnings,
        )
