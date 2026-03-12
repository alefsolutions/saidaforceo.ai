# SAIDA Engine Workflow

## Purpose

This document defines the execution workflow of the SAIDA library.

SAIDA is a lightweight Python library with the following guiding principle:

**Data analysis first. Reasoning second.**

Reasoning may assist with ambiguity handling and explanation, but deterministic computation is the source of truth.

---

## High-Level Flow

```text
Input Data
-> Adapter Load
-> Optional Markdown Context Attach
-> Dataset Profiling
-> Optional LLM Prompt Interpretation
-> NLP Request Normalization
-> Plan Generation
-> Plan Validation
-> Deterministic Compute Execution
-> Optional LLM Reasoning / Explanation
-> Result Packaging
```

---

## 1. Input Stage

The user provides one or more of the following:

- a dataset source
- an optional semantic markdown context file
- a question or analysis request
- optional ML target and forecasting options for future versions

Example:

```python
engine.analyze(
    dataset=dataset,
    question="Why did sales drop in March?"
)
```

---

## 2. Adapter Load Stage

The adapter is responsible for turning an external source into a normalized `Dataset`.

Examples:
- CSV adapter reads a file
- SQL adapter executes a SQLite query
- pandas adapter wraps a DataFrame

Outputs:
- normalized `Dataset`
- basic source metadata

---

## 3. Context Attach Stage

If a markdown context file is provided, SAIDA loads and parses it into `SourceContext`.

This context supplements technical schema with business meaning.

Examples of context:
- metric definitions
- caveats
- trusted time fields
- preferred identifiers
- business rules

This stage should not mutate the dataset itself.
It only enriches metadata.

---

## 4. Profiling Stage

SAIDA profiles the dataset deterministically.

Outputs:
- `DatasetProfile`
- `MLReadinessProfile` where applicable

Typical profiling tasks:
- infer types
- count rows and columns
- inspect nulls
- detect measures and dimensions
- identify time fields
- identify duplicate rows
- identify ML suitability hints for later phases

This stage should happen before planning.

---

## 5. NLP Request Normalization Stage

SAIDA converts raw user input into a normalized `AnalysisRequest`.

In the current repo build, this stage validates prompt interpretation against discovered dataset structure and semantic context.

If an LLM provider is enabled, the model may propose intent first.
That proposal must be validated before SAIDA accepts it.

This stage extracts:
- question
- task hint
- target
- forecast horizon
- filters
- grouping hints
- execution options

This normalized request is the planning input.

The NLP layer should extract structured signal, not analytical conclusions.

If no valid mapping can be established safely, SAIDA should either:
- ask for clarification, or
- decline cleanly

---

## 6. Planning Stage

The planning stage determines the workflow.

The planner uses:
- request
- dataset profile
- semantic context
- configured capabilities

The planner produces an `AnalysisPlan`.

Example:
- Step 1: DuckDB time trend summary
- Step 2: DuckDB segment comparison
- Step 3: stats correlation check
- Step 4: reasoning summary

### Planning design
Prefer deterministic rules first.
Optional LLM-assisted planning can be layered on top, but it should remain optional.
Current non-ML planning supports month-based time references. Quarter and broader relative-period execution are not implemented yet.

---

## 7. Plan Validation Stage

Before execution, SAIDA validates the plan.

Examples:
- forecasting requires a time column
- classification requires a target
- insufficient rows should trigger warnings
- missing values may limit model training
- invalid filters and missing target columns should fail clearly

Invalid plans should fail early with clear errors.

---

## 8. Compute Execution Stage

The engine executes plan steps in order.

### DuckDB compute
Used for:
- filtering
- aggregations
- grouped period comparisons
- top movers
- time bucketing
- contribution analysis

### Stats compute
Used for:
- descriptive statistics
- correlation
- anomaly detection
- simple group mean comparison
- time-series diagnostics

### ML compute
Used for:
- regression
- classification
- forecasting
- model evaluation

Important:
- ML training should be explicit
- training should not run on ingestion by default
- reuse saved models where possible
- the current repo build keeps ML methods as placeholders and does not execute these stages yet

---

## 9. Optional Reasoning Stage

If reasoning is enabled, SAIDA may:
- summarize results
- explain analytical findings
- generate next-step suggestions
- translate results into more natural language
- help resolve ambiguity when the NLP layer has low confidence

Reasoning uses:
- computed outputs
- plan details
- warnings
- context metadata

Reasoning must not override computed facts.

The reasoning layer should be LLM-agnostic so it can support direct SDK integrations, LangChain-wrapped models, or local models.
The current repo build ships a deterministic summarizer and also supports optional model-agnostic LLM response generation with deterministic fallback.

---

## 10. Result Packaging Stage

The final output is packaged into structured result objects.

Typical output:
- summary
- metrics
- tables
- warnings
- execution trace
- analysis plan
- analysis artifacts

This output should be useful for:
- direct printing
- notebook usage
- downstream application code

---

## Dedicated Workflows

## A. Standard Analytics Workflow

Use when the request is:
- descriptive
- diagnostic
- statistical

Flow:
1. load dataset
2. attach context
3. profile
4. normalize request with transformer-based NLP/rules
5. plan
6. validate
7. run duckdb/stats steps
8. optionally summarize
9. return `AnalysisResult`

---

## B. Training Workflow

Use when the user explicitly requests model training.

Example:
```python
engine.train(dataset=dataset, target="revenue", problem_type="regression")
```

Current status:
- this API exists
- the current repo build raises a clear not-implemented error for ML methods

Flow:
1. load dataset
2. attach context
3. profile
4. assess ML readiness
5. build model spec
6. train model
7. evaluate model
8. optionally persist model
9. return `TrainResult`

Training should not be silently triggered during ordinary descriptive analysis.

---

## C. Prediction Workflow

Use when a trained model exists or training is explicitly allowed.

Flow:
1. validate model availability
2. prepare features
3. run prediction
4. package `PredictionResult`

Current status:
- reserved for later implementation

---

## D. Forecast Workflow

Use when the user asks for future projections.

Example:
```python
engine.forecast(dataset=dataset, target="sales", horizon=3)
```

Flow:
1. load dataset
2. attach context
3. profile
4. normalize request if a natural-language forecast prompt was provided
5. validate time-series suitability
6. train or load forecast model
7. generate forecast
8. package `ForecastAnalysisResult`

Current status:
- reserved for later implementation

---

## Engine Responsibilities

The main engine should:
- coordinate modules
- enforce stage order
- build traces
- manage errors
- keep the public API simple

The engine should not:
- embed heavy business logic inline
- duplicate compute logic from submodules
- hide side effects

---

## Recommended Public API

V1 should remain small.

Suggested public methods:
- `analyze(...)`
- `train(...)`
- `predict(...)`
- `forecast(...)`
- `profile(...)`

Optional:
- `load_context(...)`

---

## Trace Strategy

Each stage should append trace events.

Example events:
- dataset loaded
- context parsed
- profile generated
- request normalized
- plan validated
- duckdb query executed
- reasoning summary generated
- analysis result packaged

This improves debuggability and transparency.

---

## Failure Strategy

Fail early on:
- unsupported source formats
- invalid plans
- missing targets for ML
- insufficient history for forecasting
- invalid filters
- empty datasets

Warn, but do not necessarily fail, on:
- missing values
- weak signal
- short time windows
- context file gaps
- ambiguous prompt interpretation

---

## Final Workflow Principle

SAIDA should always follow this order:

**understand the request -> understand data -> plan analysis -> compute deterministically -> explain optionally**
