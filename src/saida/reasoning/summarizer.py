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

        ranked_table = next((table for table in tables if table.name == "ranked_breakdown"), None)
        if ranked_table is not None and not ranked_table.dataframe.empty:
            first_row = ranked_table.dataframe.iloc[0].to_dict()
            top_fields = [str(value) for key, value in first_row.items() if key not in {"rank", "target_total"}]
            if top_fields:
                parts.append(f"Top contributor: {' / '.join(top_fields)}.")

        contribution_table = next((table for table in tables if table.name == "contribution_breakdown"), None)
        if contribution_table is not None and not contribution_table.dataframe.empty:
            strongest_drop = contribution_table.dataframe.iloc[0].to_dict()
            if "delta" in strongest_drop:
                parts.append(f"Largest contribution delta was {float(strongest_drop['delta']):.2f}.")

        mover_table = next((table for table in tables if table.name == "top_movers"), None)
        if mover_table is not None and not mover_table.dataframe.empty:
            first_mover = mover_table.dataframe.iloc[0].to_dict()
            if "delta" in first_mover:
                parts.append(f"Top mover delta was {float(first_mover['delta']):.2f}.")

        anomaly_table = next((table for table in tables if table.name == "anomaly_summary"), None)
        if anomaly_table is not None:
            parts.append(f"Anomaly candidates found: {len(anomaly_table.dataframe)}.")

        if warnings:
            parts.append(f"Warnings: {'; '.join(warnings)}.")

        return " ".join(parts)
