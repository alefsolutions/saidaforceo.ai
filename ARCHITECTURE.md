# SAIDA Architecture

SAIDA follows a layered architecture:

User Prompt
-> Optional LLM Interface
-> NLP Understanding / Validation
-> Adapters (data ingestion)
-> Context (semantic markdown)
-> Profiling (dataset intelligence)
-> Planning (analysis plan generation)
-> Compute (analytics / statistics / ML)
-> Reasoning (optional LLM interpretation)
-> Results (structured outputs)

---

## Core Principle

Facts come from computation.

Insights may come from reasoning.

Prompt understanding should be converted into structured request data before planning begins.

---

## Core Layers

### NLP Understanding Layer

Responsible for converting a natural-language question into a structured request.

Typical responsibilities:

- intent classification
- metric and target extraction
- aggregation intent extraction
- date and period extraction
- filter and grouping hint extraction
- ambiguity detection

Outputs a typed `AnalysisRequest` used by the planning layer.

This layer validates prompt interpretation against discovered schema and semantic context.
It should not compute analytical facts.

If an optional LLM provider is enabled, the model may propose intent first.
The validation layer decides whether that proposal becomes an `AnalysisRequest`, needs clarification, or must be refused.

---

### Optional LLM Interface Layer

Responsible for:

- interpreting natural prompts in a more flexible way
- proposing candidate intent fields
- producing optional post-compute response wording

This layer must remain provider and model agnostic.

The LLM may propose.
The validation and compute layers decide.

---

### Adapters

Responsible for loading data sources.

Examples:

- CSV
- Excel
- SQL
- Pandas DataFrame
- JSON

Adapters normalize external data into a **Dataset object**.

---

### Context Layer

Provides semantic business meaning using Markdown.

Example:

- metric definitions
- business rules
- trusted columns
- caveats

---

### Profiling Layer

Understands dataset structure.

Produces:

- column types
- cardinality
- missingness
- candidate measures
- candidate dimensions
- time columns

---

### Planning Layer

Determines what analysis to perform.

The planner should operate on a normalized structured request, not raw prompt text alone.

Example plans:

- descriptive analysis
- diagnostic analysis
- statistical analysis
- aggregation-oriented analysis such as average, max, min, sum, and count
- predictive modeling

---

### Compute Layer

Three deterministic compute modules:

DuckDB - analytical SQL engine

Stats - statistical routines

ML - machine learning and forecasting

---

### Reasoning Layer

Optional LLM layer.

Responsibilities:

- interpret computed outputs
- explain analysis
- summarize insights
- resolve ambiguity when explicitly enabled

---

### Results Layer

Standardized output objects.

Contains:

- tables
- metrics
- summary text
- execution traces
- a JSON-safe analytical response contract describing resolved intent, plan, operations, outputs, and warnings

---

## LLM Positioning

The optional LLM layer should remain LLM-provider agnostic.

Examples:

- OpenAI-compatible chat models
- local models
- Ollama-hosted local models
- LangChain-integrated models
- direct SDK integrations

LLM choice should not change SAIDA's compute contracts or typed result schemas.
