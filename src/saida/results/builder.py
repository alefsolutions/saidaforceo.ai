"""Build top-level result objects."""

from __future__ import annotations

from saida.schemas import (
    AnalysisPlan,
    AnalysisResult,
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
        trace: list[ExecutionTraceEvent],
    ) -> AnalysisResult:
        return AnalysisResult(
            summary=summary,
            metrics=metrics,
            tables=tables,
            warnings=warnings,
            plan=plan,
            trace=trace,
            artifacts={},
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
