# SAIDA Library Usage

SAIDA is used as a Python library.

Example:

```
from saida import Saida
from saida.adapters import CSVAdapter

engine = Saida()

dataset = CSVAdapter("data.csv").load()

result = engine.analyze(
    dataset=dataset,
    question="Why did sales drop last quarter?"
)

print(result.summary)
```

---

# Forecast Example

```
forecast = engine.forecast(
    dataset=dataset,
    target="sales",
    horizon=3
)
```

---

# Train Model

```
model = engine.train(
    dataset=dataset,
    target="revenue"
)
```