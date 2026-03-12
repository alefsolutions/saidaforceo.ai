"""ML placeholders for future implementation."""

from __future__ import annotations

from saida.exceptions import ModelTrainingError
from saida.schemas import ForecastResult, ModelSpec, ModelTrainingResult, PredictionResult


class BaselineMlEngine:
    """Reserve the ML surface for a later implementation pass."""

    def train(self, spec: ModelSpec) -> ModelTrainingResult:
        """Raise a clear error until the ML layer is implemented."""
        raise ModelTrainingError(
            f"Model training for target '{spec.target}' is not implemented yet. "
            "The ML layer will be added in a later pass."
        )

    def predict(self) -> PredictionResult:
        """Raise a clear error until the prediction layer is implemented."""
        raise ModelTrainingError("Prediction is not implemented yet. The ML layer will be added in a later pass.")

    def forecast(self, target: str, horizon: int) -> ForecastResult:
        """Raise a clear error until the forecasting layer is implemented."""
        raise ModelTrainingError(
            f"Forecasting for target '{target}' with horizon {horizon} is not implemented yet. "
            "The ML layer will be added in a later pass."
        )
