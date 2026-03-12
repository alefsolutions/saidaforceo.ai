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

- Run deterministic analytics
- Perform statistical analysis
- Train predictive models
- Generate forecasts
- Attach semantic context to datasets
- Optionally use LLM reasoning

SAIDA is designed to be:

- lightweight
- modular
- extensible
- transparent
- deterministic

The library focuses on **structured analytics first**, with optional semantic reasoning layers.

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

Attach Markdown documentation to data sources to provide business meaning.

---

# Project Goals

SAIDA aims to become a **data-agnostic deterministic analytics engine** capable of reasoning across structured and unstructured data.