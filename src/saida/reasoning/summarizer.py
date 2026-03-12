"""Reasoning-friendly summary generation."""

from __future__ import annotations

import pandas as pd

from saida.schemas import AnalysisPlan, AnalysisRequest, DatasetProfile, Metric, SourceContext, TableArtifact


class ResultSummarizer:
    """Build a grounded summary from computed outputs."""

    def summarize(
        self,
        plan: AnalysisPlan,
        metrics: list[Metric],
        tables: list[TableArtifact],
        warnings: list[str],
        request: AnalysisRequest,
        profile: DatasetProfile,
        context: SourceContext | None = None,
    ) -> str:
        """Generate a deterministic summary grounded in computed outputs."""
        target_label = request.target.replace("_", " ") if request.target else "the dataset"
        parts = [f"Completed a {plan.task_type} analysis for {target_label} on {profile.dataset_name}."]

        row_count = self._metric_value(metrics, "row_count")
        if row_count is not None:
            parts.append(f"The dataset contains {row_count} rows.")

        target_metric = next((metric for metric in metrics if metric.name.endswith("_sum")), None)
        if target_metric is not None:
            label = target_metric.name.replace("_sum", "").replace("_", " ").title()
            parts.append(f"{label} total is {target_metric.value:.2f}.")

        period_table = self._table(tables, "period_comparison")
        if period_table is not None and len(period_table.dataframe) >= 2:
            parts.append(self._describe_period_comparison(period_table.dataframe, request.target))
        else:
            trend_table = self._table(tables, "time_trend")
            if trend_table is not None and not trend_table.dataframe.empty:
                parts.append(self._describe_latest_trend_point(trend_table.dataframe, request.target))

        ranked_table = self._table(tables, "ranked_breakdown")
        if ranked_table is not None and not ranked_table.dataframe.empty:
            parts.append(self._describe_ranked_contributor(ranked_table.dataframe.iloc[0], request.target))

        contribution_table = self._table(tables, "contribution_breakdown")
        if contribution_table is not None and not contribution_table.dataframe.empty:
            contribution_part = self._describe_contribution_breakdown(contribution_table.dataframe, request.target)
            if contribution_part:
                parts.append(contribution_part)

        mover_table = self._table(tables, "top_movers")
        if mover_table is not None and not mover_table.dataframe.empty:
            parts.append(self._describe_top_mover(mover_table.dataframe.iloc[0], request.target))

        diagnostics_table = self._table(tables, "time_series_diagnostics")
        if diagnostics_table is not None and not diagnostics_table.dataframe.empty:
            parts.append(self._describe_time_series_diagnostics(diagnostics_table.dataframe.iloc[0], request.target))

        anomaly_table = self._table(tables, "anomaly_summary")
        if anomaly_table is not None:
            anomaly_count = len(anomaly_table.dataframe)
            parts.append(f"Detected {anomaly_count} anomaly candidate{'s' if anomaly_count != 1 else ''}.")

        context_note = self._describe_context_note(context)
        if context_note:
            parts.append(context_note)

        if warnings:
            parts.append(f"Warnings: {'; '.join(warnings)}.")

        return " ".join(parts)

    def _metric_value(self, metrics: list[Metric], name: str) -> object | None:
        metric = next((item for item in metrics if item.name == name), None)
        if metric is None:
            return None
        return metric.value

    def _table(self, tables: list[TableArtifact], name: str) -> TableArtifact | None:
        return next((table for table in tables if table.name == name), None)

    def _describe_period_comparison(self, dataframe: pd.DataFrame, target: str | None) -> str:
        previous_row = dataframe.iloc[0]
        current_row = dataframe.iloc[-1]
        delta = float(current_row.get("delta", 0.0) or 0.0)
        previous_total = float(previous_row.get("target_total", 0.0) or 0.0)
        current_total = float(current_row.get("target_total", 0.0) or 0.0)
        pct_change = (delta / previous_total) if previous_total else 0.0
        label = target.replace("_", " ") if target else "value"

        return (
            f"{label.title()} moved from {previous_total:.2f} in {previous_row['period']} "
            f"to {current_total:.2f} in {current_row['period']} ({pct_change:+.1%})."
        )

    def _describe_latest_trend_point(self, dataframe: pd.DataFrame, target: str | None) -> str:
        latest_row = dataframe.iloc[-1]
        target_total = float(latest_row.get("target_total", 0.0) or 0.0)
        period = latest_row.get("period_month", "the latest period")
        delta = latest_row.get("period_delta")
        label = target.replace("_", " ") if target else "value"
        if pd.notna(delta):
            return f"The latest period is {period} with {label} at {target_total:.2f} and a period change of {float(delta):+.2f}."
        return f"The latest period is {period} with {label} at {target_total:.2f}."

    def _describe_ranked_contributor(self, row: pd.Series, target: str | None) -> str:
        group_label = self._row_label(row, exclude={"rank", "target_total"})
        target_total = float(row.get("target_total", 0.0) or 0.0)
        label = target.replace("_", " ") if target else "value"
        return f"Top contributor was {group_label} with {label} total of {target_total:.2f}."

    def _describe_contribution_breakdown(self, dataframe: pd.DataFrame, target: str | None) -> str | None:
        label = target.replace("_", " ") if target else "value"
        if "delta" in dataframe.columns:
            strongest_drop = dataframe.sort_values("delta").iloc[0]
            group_label = self._row_label(strongest_drop, exclude={"previous_total", "current_total", "delta", "share_of_total"})
            delta = float(strongest_drop.get("delta", 0.0) or 0.0)
            return f"Largest contribution change came from {group_label} at {delta:+.2f} {label}."
        if "share_of_total" in dataframe.columns:
            top_share = dataframe.sort_values("share_of_total", ascending=False).iloc[0]
            group_label = self._row_label(top_share, exclude={"target_total", "share_of_total"})
            share = float(top_share.get("share_of_total", 0.0) or 0.0)
            return f"Largest share of total {label} came from {group_label} at {share:.1%}."
        return None

    def _describe_top_mover(self, row: pd.Series, target: str | None) -> str:
        group_label = self._row_label(row, exclude={"rank", "previous_total", "current_total", "delta", "pct_change", "abs_delta"})
        delta = float(row.get("delta", 0.0) or 0.0)
        pct_change = float(row.get("pct_change", 0.0) or 0.0)
        label = target.replace("_", " ") if target else "value"
        return f"Top mover was {group_label} with a {delta:+.2f} change in {label} ({pct_change:+.1%})."

    def _describe_time_series_diagnostics(self, row: pd.Series, target: str | None) -> str:
        label = target.replace("_", " ") if target else "value"
        first_period = row.get("first_period")
        last_period = row.get("last_period")
        net_change = float(row.get("net_change", 0.0) or 0.0)
        volatility = float(row.get("change_volatility", 0.0) or 0.0)
        return (
            f"Across {first_period} to {last_period}, {label} changed by {net_change:+.2f} "
            f"with period-to-period volatility of {volatility:.2f}."
        )

    def _describe_context_note(self, context: SourceContext | None) -> str | None:
        if context is None:
            return None
        if context.caveats:
            return f"Context caveat: {context.caveats[0]}."
        if context.freshness_notes:
            return f"Context freshness note: {context.freshness_notes[0]}."
        return None

    def _row_label(self, row: pd.Series, exclude: set[str]) -> str:
        parts: list[str] = []
        for key, value in row.items():
            if key in exclude:
                continue
            if pd.isna(value):
                continue
            parts.append(f"{key}={value}")
        return ", ".join(parts) if parts else "the leading group"
