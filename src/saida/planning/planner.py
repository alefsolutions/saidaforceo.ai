"""Structured planning from normalized requests."""

from __future__ import annotations

from saida.exceptions import PlanningError
from saida.schemas import AnalysisPlan, AnalysisRequest, DatasetProfile, PlanStep, SourceContext


class AnalysisPlanner:
    """Create and validate deterministic analysis plans."""

    def build_plan(
        self,
        request: AnalysisRequest,
        profile: DatasetProfile,
        context: SourceContext | None = None,
    ) -> AnalysisPlan:
        """Build an executable plan from the request and profile."""
        task_type = request.task_type_hint or "descriptive"
        warnings: list[str] = []
        steps: list[PlanStep] = []

        if request.target is None and profile.measure_columns:
            request.target = profile.measure_columns[0]
            warnings.append("No target was provided; using the first measure column.")

        if task_type in {"descriptive", "diagnostic", "statistical"}:
            steps.append(
                PlanStep(
                    step_id="summary_metrics",
                    tool_family="duckdb",
                    action="dataset_summary",
                    parameters={"target": request.target, "filters": request.filters},
                    description="Compute top-level dataset metrics.",
                )
            )
            if request.target and profile.time_columns:
                steps.append(
                    PlanStep(
                        step_id="time_trend",
                        tool_family="duckdb",
                        action="time_trend",
                        parameters={
                            "target": request.target,
                            "time_column": profile.time_columns[0],
                            "filters": request.filters,
                        },
                        description="Compute the target trend over time.",
                    )
                )
            if request.target and request.time_reference and profile.time_columns:
                steps.append(
                    PlanStep(
                        step_id="period_comparison",
                        tool_family="duckdb",
                        action="period_comparison",
                        parameters={
                            "target": request.target,
                            "time_column": profile.time_columns[0],
                            "time_reference": request.time_reference,
                            "filters": request.filters,
                        },
                        description="Compare the requested period against the previous comparable period.",
                    )
                )
                if request.group_by:
                    steps.append(
                        PlanStep(
                            step_id="grouped_period_comparison",
                            tool_family="duckdb",
                            action="grouped_period_comparison",
                            parameters={
                                "target": request.target,
                                "group_by": request.group_by,
                                "time_column": profile.time_columns[0],
                                "time_reference": request.time_reference,
                                "filters": request.filters,
                            },
                            description="Compare grouped totals between adjacent periods.",
                        )
                    )
            if request.group_by:
                steps.append(
                    PlanStep(
                        step_id="group_breakdown",
                        tool_family="duckdb",
                        action="group_breakdown",
                        parameters={"target": request.target, "group_by": request.group_by, "filters": request.filters},
                        description="Break down the target metric by requested dimensions.",
                    )
                )
                steps.append(
                    PlanStep(
                        step_id="ranked_breakdown",
                        tool_family="duckdb",
                        action="ranked_breakdown",
                        parameters={
                            "target": request.target,
                            "group_by": request.group_by,
                            "filters": request.filters,
                            "limit": 5,
                        },
                        description="Rank the largest grouped contributors.",
                    )
                )
                if request.time_reference and profile.time_columns:
                    steps.append(
                        PlanStep(
                            step_id="top_movers",
                            tool_family="duckdb",
                            action="top_movers",
                            parameters={
                                "target": request.target,
                                "group_by": request.group_by,
                                "time_column": profile.time_columns[0],
                                "time_reference": request.time_reference,
                                "filters": request.filters,
                                "limit": 5,
                            },
                            description="Identify the largest grouped movers between adjacent periods.",
                        )
                    )
            elif task_type == "diagnostic" and request.target and profile.dimension_columns:
                steps.append(
                    PlanStep(
                        step_id="top_dimension_breakdown",
                        tool_family="duckdb",
                        action="group_breakdown",
                        parameters={
                            "target": request.target,
                            "group_by": [profile.dimension_columns[0]],
                            "filters": request.filters,
                        },
                        description="Break down the target metric by the leading dimension candidate.",
                    )
                )
                steps.append(
                    PlanStep(
                        step_id="top_dimension_ranking",
                        tool_family="duckdb",
                        action="ranked_breakdown",
                        parameters={
                            "target": request.target,
                            "group_by": [profile.dimension_columns[0]],
                            "filters": request.filters,
                            "limit": 5,
                        },
                        description="Rank the leading grouped contributors for the diagnostic workflow.",
                    )
                )
                if request.time_reference and profile.time_columns:
                    steps.append(
                        PlanStep(
                            step_id="top_dimension_movers",
                            tool_family="duckdb",
                            action="top_movers",
                            parameters={
                                "target": request.target,
                                "group_by": [profile.dimension_columns[0]],
                                "time_column": profile.time_columns[0],
                                "time_reference": request.time_reference,
                                "filters": request.filters,
                                "limit": 5,
                            },
                            description="Identify the largest movers for the leading dimension candidate.",
                        )
                    )
            if task_type == "diagnostic" and request.target and profile.dimension_columns:
                steps.append(
                    PlanStep(
                        step_id="contribution_breakdown",
                        tool_family="duckdb",
                        action="contribution_breakdown",
                        parameters={
                            "target": request.target,
                            "group_by": request.group_by or [profile.dimension_columns[0]],
                            "time_column": profile.time_columns[0] if profile.time_columns else None,
                            "time_reference": request.time_reference,
                            "filters": request.filters,
                        },
                        description="Estimate group-level contribution changes for the diagnostic workflow.",
                    )
                )
            steps.append(
                PlanStep(
                    step_id="missingness_summary",
                    tool_family="stats",
                    action="missingness_summary",
                    parameters={},
                    description="Summarize missing values by column.",
                )
            )
            steps.append(
                PlanStep(
                    step_id="numeric_summary",
                    tool_family="stats",
                    action="numeric_summary",
                    parameters={},
                    description="Summarize numeric columns with deterministic statistics.",
                )
            )
            if request.target:
                steps.append(
                    PlanStep(
                        step_id="distribution_summary",
                        tool_family="stats",
                        action="distribution_summary",
                        parameters={"target": request.target},
                        description="Summarize the target distribution.",
                    )
                )
                steps.append(
                    PlanStep(
                        step_id="target_correlation",
                        tool_family="stats",
                        action="target_correlation",
                        parameters={"target": request.target},
                        description="Measure correlations between the target and other numeric columns.",
                    )
                )
                steps.append(
                    PlanStep(
                        step_id="anomaly_summary",
                        tool_family="stats",
                        action="anomaly_summary",
                        parameters={
                            "target": request.target,
                            "time_column": profile.time_columns[0] if profile.time_columns else None,
                        },
                        description="Flag simple anomaly candidates for the target.",
                    )
                )
                if profile.time_columns:
                    steps.append(
                        PlanStep(
                            step_id="time_series_diagnostics",
                            tool_family="stats",
                            action="time_series_diagnostics",
                            parameters={"target": request.target, "time_column": profile.time_columns[0]},
                            description="Compute simple time-series diagnostics for the target.",
                        )
                    )
                candidate_dimensions = request.group_by or profile.dimension_columns
                comparison_dimension = [column for column in candidate_dimensions if column != request.target][:1]
                if comparison_dimension:
                    steps.append(
                        PlanStep(
                            step_id="group_mean_comparison",
                            tool_family="stats",
                            action="group_mean_comparison",
                            parameters={"target": request.target, "group_column": comparison_dimension[0]},
                            description="Compare the target mean across the first available grouping dimension.",
                        )
                    )

        if task_type == "forecasting":
            if not profile.time_columns:
                raise PlanningError("Forecasting requires a datetime column.")
            if request.target is None:
                raise PlanningError("Forecasting requires a target metric.")
            steps.append(
                PlanStep(
                    step_id="forecast",
                    tool_family="ml",
                    action="forecast",
                    parameters={"target": request.target, "time_column": profile.time_columns[0], "horizon": request.horizon or 3},
                    description="Generate a forecast for the requested target.",
                )
            )

        if task_type == "predictive":
            warnings.append("Predictive model training is not implemented yet.")

        rationale = self._build_rationale(task_type, request, context)
        return AnalysisPlan(task_type=task_type, rationale=rationale, steps=steps, warnings=warnings)

    def validate(self, plan: AnalysisPlan) -> None:
        """Validate a plan before execution."""
        if not plan.steps:
            raise PlanningError("Analysis plan contains no executable steps.")

    def _build_rationale(self, task_type: str, request: AnalysisRequest, context: SourceContext | None) -> str:
        rationale = f"Selected a {task_type} workflow based on the normalized request."
        if request.target:
            rationale += f" Target metric: {request.target}."
        if context and context.metric_definitions:
            rationale += " Semantic metric definitions were available."
        if request.filters:
            rationale += f" Filters were detected for: {', '.join(request.filters)}."
        return rationale
