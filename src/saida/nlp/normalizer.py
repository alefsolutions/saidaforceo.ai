"""Structured request normalization with a transformer hook."""

from __future__ import annotations

import re
from calendar import month_name
from calendar import month_abbr

from saida.config import NlpConfig
from saida.exceptions import ValidationError
from saida.llm import IntentProposal
from saida.schemas import AnalysisRequest, Dataset, DatasetProfile, SourceContext

TASK_LABELS = ["descriptive", "diagnostic", "statistical", "predictive", "forecasting"]
DISTINCT_VALUE_KEYWORDS = {
    "list",
    "list of all",
    "list all",
    "all values",
    "available values",
    "give me all",
}
AGGREGATION_KEYWORDS = {
    "mean": {"average", "mean", "avg"},
    "max": {"highest", "maximum", "max", "top", "largest", "best"},
    "min": {"lowest", "minimum", "min", "smallest", "worst"},
    "sum": {"total", "sum"},
    "count": {"count", "how many", "number of"},
}
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
        if not question or not question.strip():
            raise ValidationError("Analysis question cannot be empty.")
        if dataset.data.empty:
            raise ValidationError("Cannot analyze an empty dataset.")
        if profile.column_count == 0:
            raise ValidationError("Dataset profile contains no columns.")

        warnings: list[str] = []
        task_type_hint = self._classify_task(question)
        if self.config.enable_transformers and self.config.zero_shot_model:
            task_type_hint = self._maybe_refine_task_with_transformers(question, task_type_hint, warnings)

        target = self._resolve_target(question, profile, context)
        aggregation = self._extract_aggregation(question)
        time_reference = self._extract_time_reference(question)
        horizon = self._extract_horizon(question)
        group_by = self._extract_group_by(question, profile)
        filters = self._extract_filters(question, profile, context)

        if target is None and profile.measure_columns:
            warnings.append("No explicit metric matched the prompt; using the first measure candidate.")
            target = profile.measure_columns[0]
        if target is None and not profile.measure_columns:
            raise ValidationError("No target metric could be resolved from the question or dataset profile.")
        distinct_values = self._should_list_distinct_values(question, target, profile)

        request = AnalysisRequest(
            question=question,
            task_type_hint=task_type_hint,
            target=target,
            aggregation=aggregation,
            horizon=horizon,
            filters=filters,
            group_by=group_by,
            time_reference=time_reference,
            options={
                "dataset": dataset.name,
                "nlp_backend": "transformer+rules" if self.config.enable_transformers else "rules",
                "distinct_values": distinct_values,
            },
        )
        return request, warnings

    def normalize_with_proposal(
        self,
        question: str,
        dataset: Dataset,
        profile: DatasetProfile,
        proposal: IntentProposal,
        context: SourceContext | None = None,
    ) -> tuple[AnalysisRequest, list[str]]:
        """Normalize a prompt using a validated LLM proposal plus deterministic fallbacks."""
        self._validate_inputs(question, dataset, profile)
        warnings = list(proposal.warnings)

        rule_task_type = self._classify_task(question)
        rule_target = self._resolve_target(question, profile, context)
        rule_aggregation = self._extract_aggregation(question)
        rule_time_reference = self._extract_time_reference(question)
        rule_horizon = self._extract_horizon(question)
        rule_group_by = self._extract_group_by(question, profile)
        rule_filters = self._extract_filters(question, profile, context)

        task_type_hint = self._validate_task_type(proposal.task_type_hint) or rule_task_type
        target = self._resolve_candidate_column(proposal.target, profile, context)
        aggregation = self._validate_aggregation(proposal.aggregation) or rule_aggregation
        if target is None:
            target = rule_target
        group_by = self._resolve_candidate_group_by(proposal.group_by, profile)
        if group_by is None:
            group_by = rule_group_by
        filters = self._resolve_candidate_filters(proposal.filters, profile)
        if filters is None:
            filters = rule_filters
        time_reference = self._resolve_candidate_time_reference(proposal.time_reference)
        if time_reference is None:
            time_reference = rule_time_reference
        horizon = proposal.horizon if proposal.horizon and proposal.horizon > 0 else rule_horizon

        if target is None and profile.measure_columns:
            warnings.append("No explicit metric matched the prompt; using the first measure candidate.")
            target = profile.measure_columns[0]
        if target is None and not profile.measure_columns:
            raise ValidationError("No target metric could be resolved from the question or dataset profile.")
        distinct_values = self._should_list_distinct_values(question, target, profile)

        request = AnalysisRequest(
            question=question,
            task_type_hint=task_type_hint,
            target=target,
            aggregation=aggregation,
            horizon=horizon,
            filters=filters,
            group_by=group_by,
            time_reference=time_reference,
            options={
                "dataset": dataset.name,
                "nlp_backend": "llm+validation",
                "llm_status": proposal.status,
                "distinct_values": distinct_values,
            },
        )
        return request, warnings

    def _validate_inputs(self, question: str, dataset: Dataset, profile: DatasetProfile) -> None:
        if not question or not question.strip():
            raise ValidationError("Analysis question cannot be empty.")
        if dataset.data.empty:
            raise ValidationError("Cannot analyze an empty dataset.")
        if profile.column_count == 0:
            raise ValidationError("Dataset profile contains no columns.")

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
        measure_aliases: dict[str, str] = {}
        dimension_aliases: dict[str, str] = {}
        if context:
            for metric_name in context.metric_definitions:
                measure_aliases[metric_name.lower()] = metric_name
        for column_name in profile.measure_columns:
            measure_aliases[column_name.lower()] = column_name
        for column_name in profile.dimension_columns:
            dimension_aliases[column_name.lower()] = column_name

        for alias, resolved_name in measure_aliases.items():
            if alias in lowered:
                return resolved_name
        if self._looks_like_distinct_values_request(question):
            for alias, resolved_name in dimension_aliases.items():
                if alias in lowered:
                    return resolved_name
        return None

    def _extract_aggregation(self, question: str) -> str | None:
        lowered = question.lower()
        for aggregation, keywords in AGGREGATION_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return aggregation
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
        if not match:
            return None
        value = int(match.group(1))
        return value if value > 0 else None

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

    def _validate_task_type(self, task_type_hint: str | None) -> str | None:
        if task_type_hint in TASK_LABELS:
            return task_type_hint
        return None

    def _validate_aggregation(self, aggregation: str | None) -> str | None:
        if aggregation in AGGREGATION_KEYWORDS:
            return aggregation
        return None

    def _looks_like_distinct_values_request(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in DISTINCT_VALUE_KEYWORDS)

    def _should_list_distinct_values(
        self,
        question: str,
        target: str | None,
        profile: DatasetProfile,
    ) -> bool:
        if not target:
            return False
        if target not in profile.dimension_columns:
            return False
        if self._extract_aggregation(question):
            return False
        return self._looks_like_distinct_values_request(question)

    def _resolve_candidate_column(
        self,
        candidate_name: str | None,
        profile: DatasetProfile,
        context: SourceContext | None,
    ) -> str | None:
        if not candidate_name:
            return None
        lowered_candidate = candidate_name.lower().strip()
        measure_map = {column.lower(): column for column in profile.measure_columns}
        if lowered_candidate in measure_map:
            return measure_map[lowered_candidate]
        profile_columns = {column.name.lower(): column.name for column in profile.columns}
        if lowered_candidate in profile_columns:
            return profile_columns[lowered_candidate]
        if context:
            metric_map = {metric_name.lower(): metric_name for metric_name in context.metric_definitions}
            if lowered_candidate in metric_map:
                resolved = metric_map[lowered_candidate]
                return profile_columns.get(resolved.lower(), resolved)
        return None

    def _resolve_candidate_group_by(
        self,
        group_by: list[str] | None,
        profile: DatasetProfile,
    ) -> list[str] | None:
        if not group_by:
            return None
        profile_columns = {column.name.lower(): column.name for column in profile.columns}
        resolved: list[str] = []
        for candidate_name in group_by:
            lowered_candidate = candidate_name.lower().strip()
            if lowered_candidate in profile_columns:
                resolved.append(profile_columns[lowered_candidate])
        return list(dict.fromkeys(resolved)) or None

    def _resolve_candidate_filters(
        self,
        filters: dict[str, str] | None,
        profile: DatasetProfile,
    ) -> dict[str, str] | None:
        if not filters:
            return None
        profile_columns = {column.name.lower(): column.name for column in profile.columns}
        resolved: dict[str, str] = {}
        for column_name, value in filters.items():
            lowered_column = column_name.lower().strip()
            if lowered_column in profile_columns and isinstance(value, str) and value.strip():
                resolved[profile_columns[lowered_column]] = value.strip()
        return resolved or None

    def _resolve_candidate_time_reference(self, time_reference: dict[str, str] | None) -> dict[str, str] | None:
        if not time_reference:
            return None
        if not isinstance(time_reference, dict):
            return None
        time_type = time_reference.get("type")
        if time_type not in {"month_name", "quarter", "relative_period"}:
            return None
        return {str(key): str(value) for key, value in time_reference.items() if value is not None}
