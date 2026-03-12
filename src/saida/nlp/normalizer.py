"""Structured request normalization with a transformer hook."""

from __future__ import annotations

import re
from calendar import month_name
from calendar import month_abbr

from saida.config import NlpConfig
from saida.schemas import AnalysisRequest, Dataset, DatasetProfile, SourceContext

TASK_LABELS = ["descriptive", "diagnostic", "statistical", "predictive", "forecasting"]
TASK_KEYWORDS = {
    "forecasting": {"forecast", "predict next", "projection", "future"},
    "predictive": {"train", "predict", "classification", "regression", "model"},
    "diagnostic": {"why", "drop", "decrease", "decline", "driver", "cause"},
    "statistical": {"correlation", "significant", "hypothesis", "anomaly", "distribution"},
    "descriptive": {"show", "summarize", "overview", "trend", "list"},
}


class RequestNormalizer:
    """Convert raw text into a structured AnalysisRequest."""

    def __init__(self, config: NlpConfig | None = None) -> None:
        self.config = config or NlpConfig()

    def normalize(
        self,
        question: str,
        dataset: Dataset,
        profile: DatasetProfile,
        context: SourceContext | None = None,
    ) -> tuple[AnalysisRequest, list[str]]:
        """Normalize a user question into an AnalysisRequest."""
        warnings: list[str] = []
        task_type_hint = self._classify_task(question)
        if self.config.enable_transformers and self.config.zero_shot_model:
            task_type_hint = self._maybe_refine_task_with_transformers(question, task_type_hint, warnings)

        target = self._resolve_target(question, profile, context)
        time_reference = self._extract_time_reference(question)
        horizon = self._extract_horizon(question)
        group_by = self._extract_group_by(question, profile)
        filters = self._extract_filters(question, profile, context)

        if target is None and profile.measure_columns:
            warnings.append("No explicit metric matched the prompt; using the first measure candidate.")
            target = profile.measure_columns[0]

        request = AnalysisRequest(
            question=question,
            task_type_hint=task_type_hint,
            target=target,
            horizon=horizon,
            filters=filters,
            group_by=group_by,
            time_reference=time_reference,
            options={"dataset": dataset.name, "nlp_backend": "transformer+rules" if self.config.enable_transformers else "rules"},
        )
        return request, warnings

    def _classify_task(self, question: str) -> str:
        lowered = question.lower()
        for task_name, keywords in TASK_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return task_name
        return "descriptive"

    def _maybe_refine_task_with_transformers(self, question: str, current_label: str, warnings: list[str]) -> str:
        try:
            from transformers import pipeline
        except Exception:
            warnings.append("Transformers pipeline not available; falling back to deterministic request rules.")
            return current_label

        try:
            classifier = pipeline("zero-shot-classification", model=self.config.zero_shot_model)
            result = classifier(question, TASK_LABELS, multi_label=False)
        except Exception:
            warnings.append("Transformer classification failed; falling back to deterministic request rules.")
            return current_label

        label = result["labels"][0]
        score = float(result["scores"][0])
        if score < self.config.confidence_threshold:
            warnings.append("Transformer NLP confidence was low; retaining rule-based intent classification.")
            return current_label
        return label

    def _resolve_target(self, question: str, profile: DatasetProfile, context: SourceContext | None) -> str | None:
        lowered = question.lower()
        aliases: dict[str, str] = {}
        if context:
            for metric_name in context.metric_definitions:
                aliases[metric_name.lower()] = metric_name
        for column_name in profile.measure_columns + profile.dimension_columns:
            aliases[column_name.lower()] = column_name
        for alias, resolved_name in aliases.items():
            if alias in lowered:
                return resolved_name
        return None

    def _extract_time_reference(self, question: str) -> dict[str, str] | None:
        lowered = question.lower()
        for month_index in range(1, 13):
            month = month_name[month_index].lower()
            abbreviation = month_abbr[month_index].lower()
            if re.search(rf"\b{month}\b", lowered) or re.search(rf"\b{abbreviation}\b", lowered):
                return {"type": "month_name", "value": month, "month": str(month_index)}
        quarter_match = re.search(r"\bq([1-4])\b", lowered)
        if quarter_match:
            return {"type": "quarter", "value": quarter_match.group(0), "quarter": quarter_match.group(1)}
        if "last quarter" in lowered:
            return {"type": "relative_period", "value": "last_quarter"}
        if "last month" in lowered:
            return {"type": "relative_period", "value": "last_month"}
        if "this month" in lowered:
            return {"type": "relative_period", "value": "this_month"}
        return None

    def _extract_horizon(self, question: str) -> int | None:
        match = re.search(r"\b(\d+)\s+(?:months|month|periods|steps)\b", question.lower())
        return int(match.group(1)) if match else None

    def _extract_group_by(self, question: str, profile: DatasetProfile) -> list[str] | None:
        lowered = question.lower()
        matches: list[str] = []

        if " by " in lowered:
            _, suffix = lowered.split(" by ", 1)
            matches.extend(column for column in profile.dimension_columns if column.lower() in suffix)

        for trigger in ("per ", "across ", "for each "):
            if trigger in lowered:
                _, suffix = lowered.split(trigger, 1)
                matches.extend(column for column in profile.dimension_columns if column.lower() in suffix)

        matches = list(dict.fromkeys(matches))
        return matches or None

    def _extract_filters(
        self,
        question: str,
        profile: DatasetProfile,
        context: SourceContext | None,
    ) -> dict[str, str] | None:
        lowered = question.lower()
        filters: dict[str, str] = {}

        for dimension in profile.dimension_columns:
            pattern = rf"\b{re.escape(dimension.lower())}\s*=\s*([a-z0-9_\- ]+)"
            match = re.search(pattern, lowered)
            if match:
                filters[dimension] = match.group(1).strip()

        candidate_values = self._candidate_filter_values(profile, context)
        for column_name, values in candidate_values.items():
            for value in values:
                if value and re.search(rf"\b{re.escape(value.lower())}\b", lowered):
                    filters[column_name] = value

        return filters or None

    def _candidate_filter_values(
        self,
        profile: DatasetProfile,
        context: SourceContext | None,
    ) -> dict[str, list[str]]:
        candidate_values: dict[str, list[str]] = {}

        for column in profile.columns:
            if not column.is_dimension_candidate:
                continue
            values = []
            for sample in column.sample_values:
                if isinstance(sample, str):
                    values.append(sample)
            candidate_values[column.name] = values

        if context:
            for field_name in context.field_descriptions:
                candidate_values.setdefault(field_name, [])

        return candidate_values
