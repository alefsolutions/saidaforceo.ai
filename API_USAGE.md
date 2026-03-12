# SAIDA Library Usage

SAIDA is used as a Python library.

Example:

```python
from saida import Saida
from saida.adapters import CSVAdapter

engine = Saida()

dataset = CSVAdapter("data.csv").load()

result = engine.analyze(
    dataset=dataset,
    question="Why did revenue drop in March?"
)

print(result.summary)
print(result.to_response_dict())
```

The current non-ML build supports month-based time references reliably. Quarter-style prompts are documented as future work.

When an optional LLM provider is configured, SAIDA lets the model interpret the prompt first and then validates that proposal against the dataset profile and context before creating an `AnalysisRequest`.
The normalized request can now carry supported aggregation intents such as `sum`, `mean`, `max`, `min`, and `count`.

Aggregation-style prompts are supported by the deterministic core as well:

```python
result = engine.analyze(
    dataset=dataset,
    question="What is the average revenue?"
)

print(result.summary)
```

Distinct-value prompts are supported for dimension columns as well:

```python
result = engine.analyze(
    dataset=dataset,
    question="Give me a list of all segments"
)

print(result.summary)
# Available segment values: Enterprise, SMB.
```

Grouped aggregation prompts are summarized directly from grouped results, for example:

```python
result = engine.analyze(
    dataset=dataset,
    question="Give me the total revenue by region"
)

print(result.summary)
# Total revenue by region: region=West = 397.00; region=East = 350.00.
```

`result.to_response_dict()` returns the standardized analytical response contract. It includes:
- original question
- resolved intent
- plan and operations
- computed metric lookup
- output table metadata
- warnings and trace events

---

# Profile Example

```python
profile = engine.profile(dataset)

print(profile.measure_columns)
print(profile.time_columns)
```

---

# Capabilities Example

```python
engine.capabilities()
```

Current output in this repo build:

```python
{
    "analyze": True,
    "profile": True,
    "load_context": True,
    "train": False,
    "predict": False,
    "forecast": False,
    "llm_prompting": False,
    "llm_reasoning": False,
}
```

---

# Local CLI Example

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --context examples/sales_context.md --question "Why did revenue drop in March?"
```

Optional LLM-enhanced CLI example:

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --question "Why did revenue drop in March?" --llm-provider ollama --llm-model llama3.1
```

---

# ML Methods

```python
engine.train(...)
engine.predict(...)
engine.forecast(...)
```

These methods are intentionally not implemented yet in the current repo build.
