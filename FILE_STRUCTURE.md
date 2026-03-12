# SAIDA File Structure

```
saida/
├── __init__.py
├── engine.py
├── config.py
├── exceptions.py
├── schemas/
├── adapters/
├── context/
├── profiling/
├── planning/
├── compute/
│   ├── duckdb/
│   ├── stats/
│   └── ml/
├── reasoning/
└── results/
```

---

## engine.py

Main orchestration engine.

Coordinates:

- adapters
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

## profiling/

Dataset inspection.

Produces dataset intelligence.

---

## planning/

Creates analysis plans.

Determines workflow.

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

Used for interpretation and explanation.

---

## results/

Defines standardized result objects.