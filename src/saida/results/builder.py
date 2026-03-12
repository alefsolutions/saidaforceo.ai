"""Build top-level result objects."""

from __future__ import annotations

from dataclasses import asdict

from saida.schemas import (
    AnalysisPlan,
    AnalysisRequest,
    AnalysisResult,
    DatasetProfile,
    ExecutionTraceEvent,
    ForecastAnalysisResult,
    ForecastResult,
    Metric,
    ModelTrainingResult,
    TableArtifact,
    TrainResult,
)


class ResultBuilder:
    """Package structured outputs into result schemas."""

    def build_analysis_result(
        self,
        summary: str,
        metrics: list[Metric],
        tables: list[TableArtifact],
        warnings: list[str],
        plan: AnalysisPlan,
        request: AnalysisRequest,
        profile: DatasetProfile,
        trace: list[ExecutionTraceEvent],
    ) -> AnalysisResult:
        artifacts = self._build_analysis_artifacts(metrics, tables, warnings, plan, request, profile, trace)
        return AnalysisResult(
            summary=summary,
            metrics=metrics,
            tables=tables,
            warnings=warnings,
            plan=plan,
            trace=trace,
            artifacts=artifacts,
        )

    def build_train_result(
        self,
        summary: str,
        training: ModelTrainingResult,
        trace: list[ExecutionTraceEvent],
    ) -> TrainResult:
        return TrainResult(summary=summary, training=training, trace=trace)

    def build_forecast_result(
        self,
        summary: str,
        forecast: ForecastResult,
        trace: list[ExecutionTraceEvent],
    ) -> ForecastAnalysisResult:
        return ForecastAnalysisResult(summary=summary, forecast=forecast, trace=trace)

    def _build_analysis_artifacts(
        self,
        metrics: list[Metric],
        tables: list[TableArtifact],
        warnings: list[str],
        plan: AnalysisPlan,
        request: AnalysisRequest,
        profile: DatasetProfile,
        trace: list[ExecutionTraceEvent],
    ) -> dict[str, object]:
        metric_lookup = {metric.name: metric.value for metric in metrics}
        table_index = {
            table.name: {
                "rows": int(len(table.dataframe)),
                "columns": list(table.dataframe.columns),
                "description": table.description,
            }
            for table in tables
        }
        trace_stages = [event.stage for event in trace]

        return {
            "request": asdict(request),
            "profile": {
                "dataset_name": profile.dataset_name,
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "measure_columns": list(profile.measure_columns),
                "dimension_columns": list(profile.dimension_columns),
                "time_columns": list(profile.time_columns),
                "identifier_columns": list(profile.identifier_columns),
                "warnings": list(profile.warnings),
            },
            "metric_lookup": metric_lookup,
            "table_index": table_index,
            "warning_count": len(warnings),
            "trace_stages": trace_stages,
            "plan_step_ids": [step.step_id for step in plan.steps],
        }
