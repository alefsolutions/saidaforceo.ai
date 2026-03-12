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

---

# Profile Example

```python
profile = engine.profile(dataset)

print(profile.measure_columns)
print(profile.time_columns)
```

---

# Local CLI Example

```bash
$env:PYTHONPATH="src"
python -m saida.cli.main analyze --csv examples/sales.csv --context examples/sales_context.md --question "Why did revenue drop in March?"
```

---

# ML Methods

```python
engine.train(...)
engine.predict(...)
engine.forecast(...)
```

These methods are intentionally not implemented yet in the current repo build.
