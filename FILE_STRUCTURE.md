# SAIDA File Structure

```text
saida/
|-- __init__.py
|-- engine.py
|-- config.py
|-- exceptions.py
|-- schemas/
|-- adapters/
|-- context/
|-- nlp/
|-- profiling/
|-- planning/
|-- compute/
|   |-- duckdb/
|   |-- stats/
|   `-- ml/
|-- reasoning/
`-- results/
```

---

## engine.py

Main orchestration engine.

Coordinates:

- adapters
- context
- nlp
- profiling
- planning
- compute
- reasoning
- results

---

## schemas/

Defines core data models.

Examples:

- Dataset
- Profile
- AnalysisRequest
- AnalysisPlan
- Result
- ModelMetadata

---

## adapters/

Responsible for loading datasets.

Adapters normalize data sources.

Examples:

- CSVAdapter
- ExcelAdapter
- SQLAdapter
- PandasAdapter

---

## context/

Handles Markdown semantic context.

Allows developers to attach documentation to datasets.

---

## nlp/

Handles prompt understanding before planning.

Typical responsibilities:

- transformer-based request understanding
- intent classification
- metric extraction
- target extraction
- date and period extraction
- filter and grouping hint extraction
- request normalization

This layer should return structured request objects, not analytical conclusions.

---

## profiling/

Dataset inspection.

Produces dataset intelligence.

---

## planning/

Creates analysis plans.

Determines workflow from a normalized request plus dataset/profile context.

---

## compute/

Deterministic computation layer.

Contains:

DuckDB analytics

Statistical routines

Machine learning pipelines

---

## reasoning/

Optional LLM integration.

Used for interpretation and explanation after compute.
This layer should remain LLM-provider agnostic.

---

## results/

Defines standardized result objects.
