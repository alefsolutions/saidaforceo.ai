# SAIDA Schema Spec

## Purpose

This document defines the core internal schemas for SAIDA.

Schemas should remain:
- small
- explicit
- serializable where practical
- easy to use across modules

Use either dataclasses or Pydantic consistently in implementation.

---

## 1. Dataset Schema

Represents a normalized data source inside SAIDA.

```python
class Dataset:
    name: str
    source_type: str
    data: Any
    metadata: dict[str, Any]
    context: SourceContext | None
```

### Field meanings

- `name`: logical source name
- `source_type`: csv, excel, pandas, parquet, sql, json
- `data`: normalized underlying dataset handle, usually DataFrame or queryable object
- `metadata`: source metadata such as file path, table name, connection hints
- `context`: optional semantic markdown context

---

## 2. SourceContext Schema

Represents parsed semantic markdown documentation for a data source.

```python
class SourceContext:
    raw_markdown: str
    source_summary: str | None
    table_descriptions: dict[str, str]
    field_descriptions: dict[str, str]
    metric_definitions: dict[str, str]
    business_rules: list[str]
    caveats: list[str]
    trusted_date_fields: list[str]
    preferred_identifiers: list[str]
    freshness_notes: list[str]
```

### Notes
This schema should be tolerant of missing fields.
Not every context file must populate every section.

---

## 3. ColumnProfile Schema

Represents profile information about one column.

```python
class ColumnProfile:
    name: str
    inferred_type: str
    nullable: bool
    null_ratio: float
    unique_count: int | None
    distinct_ratio: float | None
    sample_values: list[Any]
    is_identifier_candidate: bool
    is_dimension_candidate: bool
    is_measure_candidate: bool
    is_time_candidate: bool
    warnings: list[str]
```

### `inferred_type` values
Suggested controlled values:
- integer
- float
- numeric
- string
- category
- boolean
- datetime
- date
- unknown

---

## 4. DatasetProfile Schema

Represents machine-readable understanding of the full dataset.

```python
class DatasetProfile:
    dataset_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    measure_columns: list[str]
    dimension_columns: list[str]
    time_columns: list[str]
    identifier_columns: list[str]
    duplicate_row_count: int | None
    warnings: list[str]
    ml_readiness: MLReadinessProfile | None
```

---

## 5. MLReadinessProfile Schema

Represents whether the dataset appears suitable for ML tasks.

```python
class MLReadinessProfile:
    candidate_targets: list[str]
    candidate_features: list[str]
    forecasting_ready: bool
    regression_ready: bool
    classification_ready: bool
    detected_time_column: str | None
    readiness_warnings: list[str]
```

---

## 6. AnalysisRequest Schema

Represents a normalized user request into the engine.

```python
class AnalysisRequest:
    question: str
    task_type_hint: str | None
    target: str | None
    aggregation: str | None
    horizon: int | None
    filters: dict[str, Any] | None
    group_by: list[str] | None
    time_reference: dict[str, Any] | None
    options: dict[str, Any]
```

### Notes
This should capture user intent at the library boundary.
It is expected to be produced by the NLP/request-normalization layer before planning.
Supported aggregation values in the current non-ML build are:
- sum
- mean
- max
- min
- count

---

## 7. PlanStep Schema

Represents one step in an analysis plan.

```python
class PlanStep:
    step_id: str
    tool_family: str
    action: str
    parameters: dict[str, Any]
    description: str
```

### `tool_family` values
Suggested values:
- duckdb
- stats
- ml
- reasoning

---

## 8. AnalysisPlan Schema

Represents an executable workflow.

```python
class AnalysisPlan:
    task_type: str
    rationale: str
    steps: list[PlanStep]
    warnings: list[str]
```

### `task_type` values
Suggested values:
- descriptive
- diagnostic
- statistical
- predictive
- forecasting

---

## 9. Metric Schema

Represents a scalar computed metric.

```python
class Metric:
    name: str
    value: Any
    unit: str | None
    description: str | None
```

---

## 10. TableArtifact Schema

Represents a tabular result.

```python
class TableArtifact:
    name: str
    description: str | None
    dataframe: Any
```

Use a pandas DataFrame in V1 where practical.

---

## 11. ExecutionTraceEvent Schema

Represents one event during execution.

```python
class ExecutionTraceEvent:
    stage: str
    message: str
    payload: dict[str, Any] | None
```

### Example stages
- adapter
- context
- nlp
- profiling
- planning
- compute
- reasoning
- results

---

## 12. ModelSpec Schema

Represents requested ML model configuration.

```python
class ModelSpec:
    problem_type: str
    target: str
    feature_columns: list[str] | None
    model_name: str | None
    options: dict[str, Any]
```

### `problem_type` values
- regression
- classification
- forecasting

---

## 13. ModelTrainingResult Schema

Represents the output of training.

```python
class ModelTrainingResult:
    model_name: str
    problem_type: str
    target: str
    feature_columns: list[str]
    metrics: dict[str, float]
    artifact_path: str | None
    warnings: list[str]
```

---

## 14. PredictionResult Schema

Represents prediction output.

```python
class PredictionResult:
    model_name: str
    problem_type: str
    predictions: list[Any]
    confidence: list[float] | None
    warnings: list[str]
```

---

## 15. ForecastResult Schema

Represents forecasting output.

```python
class ForecastResult:
    target: str
    horizon: int
    forecast_values: list[float]
    lower_bounds: list[float] | None
    upper_bounds: list[float] | None
    metrics: dict[str, float] | None
    warnings: list[str]
```

---

## 16. AnalysisResult Schema

Primary top-level output returned by `engine.analyze()`.

```python
class AnalysisResult:
    summary: str
    metrics: list[Metric]
    tables: list[TableArtifact]
    warnings: list[str]
    plan: AnalysisPlan
    trace: list[ExecutionTraceEvent]
    artifacts: dict[str, Any]
    response: dict[str, Any]
```

### Notes
This must be rich enough for:
- console output
- programmatic inspection
- later API exposure

The `summary` field may be generated deterministically or via an optional LLM reasoning layer, but it must remain grounded in computed outputs.
The `response` field should be JSON-safe and describe:
- the original question
- resolved intent
- plan and operations
- computed outputs
- warnings
- trace events

---

## 17. TrainResult Schema

Top-level output returned by `engine.train()`.

```python
class TrainResult:
    summary: str
    training: ModelTrainingResult
    trace: list[ExecutionTraceEvent]
```

---

## 18. ForecastAnalysisResult Schema

Top-level output returned by `engine.forecast()`.

```python
class ForecastAnalysisResult:
    summary: str
    forecast: ForecastResult
    trace: list[ExecutionTraceEvent]
```

---

## Validation Guidance

All schemas should:
- validate required fields
- default lists/dicts safely
- remain serializable when possible
- avoid storing huge raw data blobs unless necessary

---

## Final Schema Principle

Schemas should describe:
- what entered the system
- what the system understood
- what the system planned
- what the system computed
- what the system returned
![SAIDA Banner](assets/github-banner.png)
