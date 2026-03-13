![SAIDA Banner](assets/github-banner.png)

# Machine Learning Pipeline

SAIDA defines a future predictive modeling and forecasting surface.

Machine learning is **separate from deterministic analytics**.

Current repo status:

- `train(...)`, `predict(...)`, and `forecast(...)` exist in the public API
- the current build raises clear not-implemented errors for these methods
- deterministic analytics is the active implementation focus right now
- `engine.capabilities()` reports these ML methods as unavailable
- the current repo should be treated as a non-ML analytical engine first, with ML reserved for a later implementation pass

---

## ML Pipeline Stages

1. ML Readiness Check
2. Feature Preparation
3. Training
4. Evaluation
5. Prediction
6. Forecasting

---

## Training Philosophy

Training should NOT occur automatically during ingestion.

Instead:

- train on explicit request
- reuse existing models
- persist trained models

---

## Supported Tasks

### Regression

Predict continuous values.

Example:

- revenue prediction

---

### Classification

Predict categories.

Example:

- customer churn prediction

---

### Forecasting

Predict future time-series values.

Example:

- monthly sales forecast

---

## Libraries

Planned ML dependencies remain lightweight:

- scikit-learn
- XGBoost
- statsmodels

Deep learning frameworks are intentionally avoided for V1.

