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
