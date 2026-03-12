# SAIDA Architecture

SAIDA follows a layered architecture:

User Prompt
↓
Adapters (data ingestion)
↓
Context (semantic markdown)
↓
Profiling (dataset intelligence)
↓
Planning (analysis plan generation)
↓
Compute (analytics / statistics / ML)
↓
Reasoning (optional LLM interpretation)
↓
Results (structured outputs)

---

## Core Principle

Facts come from computation.

Insights may come from reasoning.

---

## Core Layers

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

Example plans:

- descriptive analysis
- diagnostic analysis
- statistical analysis
- predictive modeling

---

### Compute Layer

Three deterministic compute modules:

DuckDB — analytical SQL engine

Stats — statistical routines

ML — machine learning and forecasting

---

### Reasoning Layer

Optional LLM layer.

Responsibilities:

- interpret prompts
- explain analysis
- summarize insights

---

### Results Layer

Standardized output objects.

Contains:

- tables
- metrics
- summary text
- execution traces