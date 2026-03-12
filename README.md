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

- Convert natural-language questions into structured analysis requests using rule-based normalization with an optional transformer hook
- Run deterministic analytics
- Perform statistical analysis
- Attach semantic context to datasets
- Optionally use deterministic or LLM-backed reasoning for interpretation and responses.

SAIDA is designed to be:

- lightweight
- modular
- extensible
- transparent
- deterministic
- readable

The library focuses on **structured analytics first**, with optional semantic reasoning layers.

Prompt handling follows a three-stage design:

- request normalization extracts structured intent from the user question
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

For richer local inspection, the CLI also supports JSON output and optional plan/trace printing:

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --context examples/sales_context.md --question "Why did revenue drop in March?" --show-plan --show-trace
python -m saida.cli.main profile --csv examples/sales.csv --json
```

The repo also includes runnable examples:

```bash
$env:PYTHONPATH="src"
python examples/run_profile.py
python examples/run_analysis.py
```

---

# Key Capabilities

### Prompt Understanding

- rule-based request normalization with an optional transformer classifier hook
- intent classification
- metric and target extraction
- date and period extraction
- filter and grouping hint extraction
- structured `AnalysisRequest` generation

### Data Analytics

- descriptive analytics
- segmentation
- anomaly detection
- correlation analysis
- time-series analysis
- grouped period comparison
- top-mover diagnostics

### Machine Learning

Not implemented yet in the current repo build.

- `train(...)`
- `predict(...)`
- `forecast(...)`

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
- dataset profiling and readiness hints
- request normalization with rules and an optional transformer hook
- deterministic planning
- DuckDB analytics for summaries, trends, grouped comparisons, contribution analysis, and top movers
- deterministic statistical summaries, correlations, anomaly checks, and time-series diagnostics
- working CLI commands for `version`, `profile`, and `analyze` against CSV input

`train(...)`, `predict(...)`, and `forecast(...)` are intentionally reserved for a later ML implementation pass.

---

# Project Goals

SAIDA aims to become a **data-agnostic deterministic analytics engine** capable of reasoning across structured and unstructured data.
