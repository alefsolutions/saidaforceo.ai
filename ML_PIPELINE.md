# Machine Learning Pipeline

SAIDA supports predictive modeling and forecasting.

Machine learning is **separate from deterministic analytics**.

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

SAIDA uses lightweight ML libraries:

- scikit-learn
- XGBoost
- statsmodels

Deep learning frameworks are intentionally avoided for V1.