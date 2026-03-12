"""Main orchestration engine for SAIDA."""

from __future__ import annotations

from saida.compute import BaselineMlEngine, DuckDBComputeEngine, StatsComputeEngine
from saida.config import SaidaConfig
from saida.context import SourceContextParser
from saida.nlp import RequestNormalizer
from saida.planning import AnalysisPlanner
from saida.profiling import DatasetProfiler
from saida.reasoning import ResultSummarizer
from saida.results import ResultBuilder
from saida.schemas import (
    AnalysisResult,
    Dataset,
    DatasetProfile,
    ExecutionTraceEvent,
    ForecastAnalysisResult,
    ModelSpec,
    PredictionResult,
    SourceContext,
    TrainResult,
)


class Saida:
    """Coordinate SAIDA modules through a simple Python API."""

    def __init__(self, config: SaidaConfig | None = None) -> None:
        self.config = config or SaidaConfig()
        self.profiler = DatasetProfiler()
        self.normalizer = RequestNormalizer(self.config.nlp)
        self.planner = AnalysisPlanner()
        self.duckdb = DuckDBComputeEngine()
        self.stats = StatsComputeEngine()
        self.ml = BaselineMlEngine()
        self.summarizer = ResultSummarizer()
        self.results = ResultBuilder()

    def profile(self, dataset: Dataset) -> DatasetProfile:
        """Profile a dataset deterministically."""
        return self.profiler.profile(dataset)

    def analyze(self, dataset: Dataset, question: str) -> AnalysisResult:
        """Run an end-to-end deterministic analysis workflow."""
        trace = [self._trace("adapter", "dataset loaded", {"dataset": dataset.name})]
        if dataset.context is not None:
            trace.append(self._trace("context", "context attached", {"metric_count": len(dataset.context.metric_definitions)}))

        profile = self.profile(dataset)
        trace.append(self._trace("profiling", "profile generated", {"row_count": profile.row_count}))

        request, request_warnings = self.normalizer.normalize(question, dataset, profile, dataset.context)
        trace.append(self._trace("nlp", "request normalized", {"task_type": request.task_type_hint, "target": request.target}))

        plan = self.planner.build_plan(request, profile, dataset.context)
        self.planner.validate(plan)
        trace.append(self._trace("planning", "plan validated", {"task_type": plan.task_type, "step_count": len(plan.steps)}))

        metrics = []
        tables = []
        warnings = self._merge_warnings(profile.warnings, request_warnings, plan.warnings)

        for step in plan.steps:
            if step.tool_family == "duckdb":
                if step.action == "dataset_summary":
                    step_metrics, step_tables = self.duckdb.dataset_summary(
                        dataset.data,
                        step.parameters.get("target"),
                        step.parameters.get("filters"),
                    )
                    metrics.extend(step_metrics)
                    tables.extend(step_tables)
                elif step.action == "time_trend":
                    tables.append(
                        self.duckdb.time_trend(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["time_column"],
                            step.parameters.get("filters"),
                        )
                    )
                elif step.action == "group_breakdown":
                    tables.append(
                        self.duckdb.group_breakdown(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["group_by"],
                            step.parameters.get("filters"),
                        )
                    )
                elif step.action == "ranked_breakdown":
                    tables.append(
                        self.duckdb.ranked_breakdown(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["group_by"],
                            step.parameters.get("filters"),
                            step.parameters.get("limit", 5),
                        )
                    )
                elif step.action == "grouped_period_comparison":
                    tables.append(
                        self.duckdb.grouped_period_comparison(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["group_by"],
                            step.parameters["time_column"],
                            step.parameters["time_reference"],
                            step.parameters.get("filters"),
                        )
                    )
                elif step.action == "top_movers":
                    tables.append(
                        self.duckdb.top_movers(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["group_by"],
                            step.parameters["time_column"],
                            step.parameters["time_reference"],
                            step.parameters.get("filters"),
                            step.parameters.get("limit", 5),
                        )
                    )
                elif step.action == "contribution_breakdown":
                    tables.append(
                        self.duckdb.contribution_breakdown(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["group_by"],
                            step.parameters.get("time_column"),
                            step.parameters.get("time_reference"),
                            step.parameters.get("filters"),
                        )
                    )
                elif step.action == "period_comparison":
                    tables.append(
                        self.duckdb.period_comparison(
                            dataset.data,
                            step.parameters["target"],
                            step.parameters["time_column"],
                            step.parameters["time_reference"],
                            step.parameters.get("filters"),
                        )
                    )
            elif step.tool_family == "stats":
                if step.action == "missingness_summary":
                    tables.append(self.stats.missingness_summary(dataset.data))
                elif step.action == "numeric_summary":
                    tables.append(self.stats.numeric_summary(dataset.data))
                elif step.action == "distribution_summary":
                    distribution_table = self.stats.distribution_summary(dataset.data, step.parameters["target"])
                    if distribution_table is not None:
                        tables.append(distribution_table)
                elif step.action == "target_correlation":
                    correlation_table = self.stats.correlation_matrix(dataset.data, step.parameters.get("target"))
                    if correlation_table is not None:
                        tables.append(correlation_table)
                elif step.action == "anomaly_summary":
                    anomaly_table = self.stats.anomaly_summary(
                        dataset.data,
                        step.parameters["target"],
                        step.parameters.get("time_column"),
                    )
                    if anomaly_table is not None:
                        tables.append(anomaly_table)
                elif step.action == "time_series_diagnostics":
                    diagnostics_table = self.stats.time_series_diagnostics(
                        dataset.data,
                        step.parameters["target"],
                        step.parameters["time_column"],
                    )
                    if diagnostics_table is not None:
                        tables.append(diagnostics_table)
                elif step.action == "group_mean_comparison":
                    comparison_table = self.stats.group_mean_comparison(
                        dataset.data,
                        step.parameters["target"],
                        step.parameters["group_column"],
                    )
                    if comparison_table is not None:
                        tables.append(comparison_table)
            trace.append(self._trace("compute", f"executed {step.action}", step.parameters))

        summary = self.summarizer.summarize(plan, metrics, tables, warnings, request, profile, dataset.context)
        trace.append(self._trace("results", "analysis result packaged", {"summary_length": len(summary)}))
        return self.results.build_analysis_result(summary, metrics, tables, warnings, plan, request, profile, trace)

    def train(
        self,
        dataset: Dataset,
        target: str,
        problem_type: str = "regression",
        feature_columns: list[str] | None = None,
    ) -> TrainResult:
        """Reserve the training API surface for a later ML implementation."""
        spec = ModelSpec(problem_type=problem_type, target=target, feature_columns=feature_columns, model_name=None)
        return self.ml.train(spec)

    def predict(self, dataset: Dataset, artifact_path: str) -> PredictionResult:
        """Reserve the prediction API surface for a later ML implementation."""
        _ = dataset
        _ = artifact_path
        return self.ml.predict()

    def forecast(self, dataset: Dataset, target: str, horizon: int = 3) -> ForecastAnalysisResult:
        """Reserve the forecast API surface for a later ML implementation."""
        _ = dataset
        return self.ml.forecast(target, horizon)

    def load_context(self, markdown: str) -> SourceContext:
        """Parse markdown context through the context layer."""
        return SourceContextParser().parse(markdown)

    def _trace(self, stage: str, message: str, payload: dict[str, object] | None = None) -> ExecutionTraceEvent:
        return ExecutionTraceEvent(stage=stage, message=message, payload=payload)

    def _merge_warnings(self, *warning_groups: list[str]) -> list[str]:
        merged: list[str] = []
        for warning_group in warning_groups:
            for warning in warning_group:
                if warning not in merged:
                    merged.append(warning)
        return merged
