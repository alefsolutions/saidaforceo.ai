# SAIDA Coding Guidelines

## Purpose

These guidelines define how SAIDA code should be written so the library remains:

- lightweight
- deterministic-first
- modular
- easy to test
- easy for LLMs and humans to extend

SAIDA is a Python library, not a SaaS app, not an API platform, and not a heavy framework.

---

## Core Engineering Principles

### 1. Deterministic-first
Any numerical fact, metric, statistical output, model score, or forecast must come from deterministic code.

Prompt understanding should use modern transformer-based NLP, optionally combined with deterministic rules.
That stage is allowed to extract structured request data, but it is not the source of analytical truth.

LLMs may:
- interpret
- summarize
- explain
- plan

LLMs must not:
- invent statistics
- fabricate metrics
- override computed facts

### 2. Boring code is better than academic code
Choose human-readable code over clever, compressed, or overly abstract code every time.

SAIDA should prefer implementation styles that are easy to read, explain, debug, and extend.

This follows the Zen of Python:

- explicit is better than implicit
- simple is better than complex
- flat is better than nested
- sparse is better than dense
- readability counts
- in the face of ambiguity, refuse the temptation to guess
- if the implementation is hard to explain, it is a bad idea

Prefer:
- obvious control flow
- descriptive names
- straightforward branching
- small helper functions
- plain data structures when they are enough

Avoid:
- clever compression
- academic abstractions without a practical payoff
- dense one-liners that hide intent
- indirection that makes debugging harder

### 3. Lightweight by default
Prefer small, focused modules.

Avoid:
- unnecessary abstractions
- framework-heavy patterns
- deep inheritance trees
- hidden magic

Prefer:
- simple classes
- small pure functions
- explicit inputs and outputs
- composition over inheritance

### 4. Library-first design
The public interface must work cleanly from Python scripts.

Good:
```python
engine = Saida()
result = engine.analyze(dataset=dataset, question="Why did sales drop?")
```

Avoid designs that assume:
- web servers
- async queues
- background workers
- API-first workflows

### 5. Data analysis first, reasoning second
Core analytical workflows must function without LLMs.

Reasoning should be optional and additive.

Prompt understanding should be separated from reasoning.
Use transformer-based NLP or transformer-assisted request normalization to produce structured intent before planning.

### 6. Strong typing and explicit schemas
Use dataclasses or Pydantic consistently for shared domain objects.

Every major boundary should have typed objects for:
- dataset
- profile
- plan
- result
- model metadata

### 7. Testability
Every deterministic compute function should be easy to unit test.

Prefer:
- pure functions where possible
- isolated compute modules
- minimal hidden state

---

## Repository Conventions

## Directory intent

- `schemas/` stores shared typed data contracts
- `adapters/` ingests and normalizes sources
- `context/` parses semantic markdown context
- `nlp/` converts natural-language questions into structured request objects
- `profiling/` inspects datasets
- `planning/` generates deterministic or LLM-assisted plans
- `compute/` executes analytics, stats, and ML
- `reasoning/` handles optional LLM interpretation
- `results/` packages outputs
- `engine.py` orchestrates everything

---

## Python Style

### Version target
Use Python 3.11+ unless a lower version is explicitly needed.

### Naming
Use:
- `snake_case` for functions and modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

### Function design
Functions should:
- do one thing well
- have explicit parameters
- return typed objects where practical
- avoid mutating external state unless necessary
- be easy for another engineer to understand quickly

Bad:
```python
def process_data(x):
    ...
```

Better:
```python
def build_dataset_profile(dataset: Dataset) -> DatasetProfile:
    ...
```

### Class design
Prefer small service classes with explicit responsibilities.

Good examples:
- `DatasetProfiler`
- `AnalysisPlanner`
- `DuckDBComputeEngine`
- `ResultBuilder`

Avoid giant god classes.
Avoid clever object models when a simple function or plain class is clearer.

---

## Error Handling

Use library-specific exceptions.

Examples:
- `SaidaError`
- `AdapterError`
- `ContextError`
- `ProfileError`
- `PlanningError`
- `ComputeError`
- `ModelTrainingError`
- `ReasoningError`

Raise meaningful exceptions with actionable messages.

Bad:
```python
raise Exception("failed")
```

Better:
```python
raise PlanningError("Forecasting requires a datetime column and at least 12 observations.")
```

---

## Logging

Logging should be:
- minimal
- optional
- useful for debugging

Do not spam logs.

Log:
- source loading start/end
- plan generation
- compute stage transitions
- model training start/end
- important warnings

Do not log:
- full datasets
- secrets
- API keys
- excessive row-level data

---

## Dependencies

Keep dependencies minimal.

### Preferred core dependencies
- duckdb
- pandas
- numpy
- scipy
- sentence-transformers, transformers, or equivalent modern NLP tooling
- statsmodels
- scikit-learn
- xgboost
- pydantic or dataclasses
- optional langchain / ollama integrations

### Avoid unless clearly justified
- large orchestration frameworks
- distributed systems dependencies
- heavyweight deep learning stacks in V1

---

## Adapter Rules

Adapters must:
- normalize external data into a common internal dataset object
- attach metadata
- optionally attach semantic markdown context
- fail clearly when data is invalid

Adapters should not:
- perform analytics
- train models
- interpret prompts

---

## Context Rules

Markdown context should be treated as structured semantic hints.

Supported context categories may include:
- source summary
- table descriptions
- metric definitions
- business rules
- freshness expectations
- caveats
- trusted date fields
- preferred identifiers

Context parsers should:
- preserve raw markdown
- extract structured sections
- validate known sections
- tolerate partial context files

---

## Profiling Rules

Profiling must be deterministic.

It should inspect:
- column types
- null rates
- unique counts
- likely IDs
- dimensions
- measures
- datetime columns
- ML readiness hints

Profiling should not:
- guess business meaning beyond evidence
- perform full forecasting
- perform heavy ML training

---

## Planning Rules

Planning must prefer deterministic rule-based planning first.

Optional LLM planning may be layered on top.

Planning should consume a normalized `AnalysisRequest`, not raw prompt text scattered across the engine.

Plans should always be represented as structured objects.

Each plan must define:
- task type
- ordered steps
- compute family used
- rationale
- validation warnings

A plan must be validated before execution.

---

## Compute Rules

### DuckDB compute
Use DuckDB for:
- aggregations
- joins
- filtering
- grouping
- window functions
- time bucketing
- feature table preparation

### Stats compute
Use stats routines for:
- correlations
- hypothesis tests
- anomaly detection
- distribution checks
- time-series diagnostics

### ML compute
ML training should be explicit.

Do not auto-train on ingestion by default.

Support:
- regression
- classification
- forecasting
- evaluation
- model persistence

---

## Reasoning Rules

Reasoning is optional.

Reasoning may:
- explain computed outputs
- summarize results
- suggest next questions
- help resolve ambiguity when explicitly enabled

Reasoning must not:
- invent facts
- override computed metrics
- bypass validations

Reasoning integrations should remain LLM-provider agnostic.

---

## NLP Rules

NLP is responsible for structured signal extraction at the request boundary.

The default expectation for SAIDA is modern transformer-based NLP rather than legacy rule-only parsing.

NLP may:
- classify user intent
- extract metrics, targets, and dimensions
- extract dates, periods, filters, and grouping hints
- normalize raw prompt text into an `AnalysisRequest`

NLP must not:
- compute metrics
- explain results as if they were computed facts
- bypass plan validation

---

## Results Rules

Every result object should include enough structure for downstream use.

Typical result fields:
- `summary`
- `metrics`
- `tables`
- `warnings`
- `plan`
- `trace`
- `artifacts`

Results should be useful both to:
- humans
- calling Python code

---

## Testing Strategy

Minimum testing expectations:

### Unit tests
For:
- adapters
- context parsing
- profiling
- planning rules
- compute routines
- ML evaluation helpers
- result builders

### Integration tests
For:
- end-to-end analysis flow
- markdown context + dataset flow
- prompt to plan to compute flow
- forecasting workflow

### Golden tests
Useful for:
- stable summaries
- execution traces
- plan generation outputs

---

## Documentation Standards

Every public class and function should have:
- a clear docstring
- typed parameters
- typed returns where practical

Important modules should also include:
- short usage examples
- edge cases
- assumptions

---

## Codex Guidance

When generating code for SAIDA:

- choose boring, readable code over clever code
- keep files small
- keep responsibilities narrow
- prefer explicit schemas
- avoid speculative abstractions
- avoid premature optimization
- implement deterministic analytics first
- treat LLM integration as optional

---

## Final Rule

If there is a choice between:
- cleverness
- clarity

choose clarity.
![SAIDA Banner](assets/github-banner.png)
