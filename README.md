# SAIDA

SAIDA is a **lightweight Python analytical reasoning library**.

Core philosophy:

- **Data analysis first**
- **Reasoning second**
- **Deterministic computation before AI**
- **Structured data first**
- **Modular architecture**
- **Library-first (not SaaS, not API)**
- **Readable, boring code over clever code**

SAIDA helps developers:

- Convert natural-language questions into structured analysis requests using transformer-based NLP
- Run deterministic analytics
- Perform statistical analysis
- Train predictive models
- Generate forecasts
- Attach semantic context to datasets
- Optionally use LLM reasoning for interpretation and responses.

SAIDA is designed to be:

- lightweight
- modular
- extensible
- transparent
- deterministic
- readable

The library focuses on **structured analytics first**, with optional semantic reasoning layers.

Prompt handling follows a three-stage design:

- transformer-based NLP extracts structured intent from the user question
- deterministic compute produces facts, metrics, and model outputs
- optional LLM reasoning explains computed outputs without changing them

---

# Installation

```bash
pip install -e .
```

---

# Quick Example

```python
from saida import Saida
from saida.adapters import CSVAdapter

engine = Saida()

dataset = CSVAdapter("sales.csv").load()

result = engine.analyze(
    dataset=dataset,
    question="Why did revenue drop in March?"
)

print(result.summary)
```

For local development in this repo, use:

```bash
python -m pytest -q
```

or try the CLI with the bundled sample files:

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --context examples/sales_context.md --question "Why did revenue drop in March by region?"
```

---

# Key Capabilities

### Prompt Understanding

- transformer-based modern NLP for request understanding
- intent classification
- metric and target extraction
- date and period extraction
- filter and grouping hint extraction
- structured `AnalysisRequest` generation

### Data Analytics

- descriptive analytics
- segmentation
- cohort analysis
- anomaly detection
- correlation analysis
- time-series analysis

### Machine Learning

Not implemented yet in the current repo build.

- regression
- classification
- forecasting
- model evaluation

### Semantic Context

Attach Markdown documentation to data sources to provide business meaning. This significantly improves responses especially if optional LLM is used for reasoning.

### Reasoning

SAIDA keeps reasoning model-agnostic.

- Any compatible LLM provider may be used
- **LLMs interpret computed results, not generate facts**
- Core analytics workflows remain usable without an LLM

---

# Current Status

The current implementation is focused on the non-ML deterministic core:

- CSV, Excel, JSON, Pandas, and SQLite-backed SQL adapters
- semantic markdown context parsing
- dataset profiling
- request normalization
- deterministic planning
- DuckDB analytics
- deterministic statistical summaries and anomaly checks

`train(...)`, `predict(...)`, and `forecast(...)` are intentionally reserved for a later ML implementation pass.

---

# Project Goals

SAIDA aims to become a **data-agnostic deterministic analytics engine** capable of reasoning across structured and unstructured data.
