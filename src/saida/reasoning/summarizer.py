"""Reasoning-friendly summary generation."""

from __future__ import annotations

from saida.schemas import AnalysisPlan, Metric, TableArtifact


class ResultSummarizer:
    """Build a grounded summary from computed outputs."""

    def summarize(
        self,
        plan: AnalysisPlan,
        metrics: list[Metric],
        tables: list[TableArtifact],
        warnings: list[str],
    ) -> str:
        """Generate a deterministic summary grounded in computed outputs."""
        parts = [f"Completed a {plan.task_type} analysis."]
        row_count = next((metric.value for metric in metrics if metric.name == "row_count"), None)
        if row_count is not None:
            parts.append(f"The dataset contains {row_count} rows.")

        target_metric = next((metric for metric in metrics if metric.name.endswith("_sum")), None)
        if target_metric is not None:
            label = target_metric.name.replace("_sum", "").replace("_", " ").title()
            parts.append(f"{label} total is {target_metric.value:.2f}.")

        if any(table.name == "time_trend" for table in tables):
            parts.append("A time trend table was generated.")

        if warnings:
            parts.append(f"Warnings: {'; '.join(warnings)}.")

        return " ".join(parts)
