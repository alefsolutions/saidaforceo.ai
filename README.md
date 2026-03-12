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

- Convert natural-language questions into structured analysis requests using strict NLP validation with an optional model-agnostic LLM interpretation layer
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

- optional LLM interpretation proposes intent from the user question
- strict NLP validation converts accepted intent into a structured `AnalysisRequest`
- deterministic compute produces facts, metrics, and model outputs
- optional LLM reasoning explains computed outputs without changing them

The final analysis result also includes a standardized JSON-safe response contract so callers can inspect what SAIDA understood, planned, executed, and returned.

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
print(result.to_response_dict())
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

The `analyze --json` path emits the standardized analytical response contract directly.

You can also enable an optional local or hosted LLM provider for prompt interpretation and response wording:

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --question "Why did revenue drop in March?" --llm-provider ollama --llm-model llama3.1
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

- strict rule-based validation against discovered schema and semantic context
- optional model-agnostic LLM interpretation before validation
- intent classification
- metric and target extraction
- aggregation intent extraction for requests like average, highest, lowest, total, and count
- direct dimension-value listing for prompts like "list all segments"
- row-count questions like "how many rows do we have"
- representation ranking questions like "which segment is least represented"
- metadata inventory questions like "what columns are available"
- date and period extraction
- filter and grouping hint extraction
- structured `AnalysisRequest` generation
- clarification or refusal when no valid mapping can be established safely

### Data Analytics

- descriptive analytics
- deterministic aggregation queries such as sum, average, maximum, minimum, and count
- direct listing of distinct dimension values
- row counts and grouped row-count ranking
- dataset metadata inventory such as columns, measures, dimensions, and time columns
- segmentation
- anomaly detection
- correlation analysis
- time-series analysis
- grouped period comparison
- top-mover diagnostics

Aggregation responses are intent-aware:
- scalar aggregate prompts lead with the scalar answer
- grouped aggregate prompts lead with grouped totals or grouped averages
- diagnostic prompts still prioritize explanatory narrative

### Machine Learning

Not implemented yet in the current repo build.

- `train(...)`
- `predict(...)`
- `forecast(...)`

You can inspect the current public surface directly:

```python
engine.capabilities()
# {
#   'analyze': True,
#   'profile': True,
#   'load_context': True,
#   'train': False,
#   'predict': False,
#   'forecast': False,
#   'llm_prompting': False,
#   'llm_reasoning': False,
# }
```

### Semantic Context

Attach Markdown documentation to data sources to provide business meaning. This significantly improves responses especially if optional LLM is used for reasoning.

### Reasoning

SAIDA keeps reasoning model-agnostic.

- Any compatible LLM provider may be used
- LLM use is optional and falls back to deterministic behavior when unavailable
- **LLMs interpret computed results, not generate facts**
- **LLM prompt proposals must pass strict validation before SAIDA creates an `AnalysisRequest`**
- Core analytics workflows remain usable without an LLM

---

# Current Status

The current implementation is focused on the non-ML deterministic core:

- CSV, Excel, JSON, Pandas, and SQLite-backed SQL adapters
- semantic markdown context parsing
- dataset profiling and readiness hints
- request normalization with strict rules and an optional model-agnostic LLM interpretation layer
- deterministic planning
- DuckDB analytics for summaries, trends, grouped comparisons, contribution analysis, and top movers
- deterministic statistical summaries, correlations, anomaly checks, and time-series diagnostics
- working CLI commands for `version`, `profile`, and `analyze` against CSV input
- optional model-agnostic LLM support for prompt interpretation and response wording, including an Ollama provider

`train(...)`, `predict(...)`, and `forecast(...)` are intentionally reserved for a later ML implementation pass.

---

# Project Goals

SAIDA aims to become a **data-agnostic deterministic analytics engine** capable of reasoning across structured and unstructured data.
