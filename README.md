# SAIDA

SAIDA is a **lightweight Python analytical reasoning library**.

Core philosophy:

- **Data analysis first**
- **Reasoning second**
- **Deterministic computation before AI**
- **Structured data first**
- **Modular architecture**
- **Library-first (not SaaS, not API)**

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

The library focuses on **structured analytics first**, with optional semantic reasoning layers.

Prompt handling follows a three-stage design:

- transformer-based NLP extracts structured intent from the user question
- deterministic compute produces facts, metrics, and model outputs
- optional LLM reasoning explains computed outputs without changing them

---

# Installation

```bash
pip install saida
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

# Project Goals

SAIDA aims to become a **data-agnostic deterministic analytics engine** capable of reasoning across structured and unstructured data.