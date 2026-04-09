"""Model abstractions for probability prediction models.

All models produce calibrated probabilities for binary prediction markets.
The :class:`ProbabilityModel` ABC defines the contract that the execution
engine, training pipeline, and backtesting framework depend on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ModelPrediction:
    """Output of a probability model prediction.

    Attributes:
        probability: Calibrated P(yes) in [0, 1].
        raw_probability: Pre-calibration model output.
        confidence: Model confidence score in [0, 1].
        model_name: Name of the model that produced this prediction.
        model_version: Version string of the model.
        features_used: Feature name -> value mapping used for this prediction.
        prediction_time: When this prediction was generated.
    """

    probability: float
    raw_probability: float
    confidence: float
    model_name: str
    model_version: str
    features_used: dict[str, float]
    prediction_time: datetime


class ProbabilityModel(ABC):
    """Abstract base class for all probability prediction models.

    Subclasses must implement prediction, fitting, and serialisation
    methods.

    Attributes:
        name: Unique model identifier.
        version: Model version string (git hash, date, or semver).
        trained_at: Timestamp of when the model was last trained.
    """

    name: str
    version: str
    trained_at: datetime | None = None

    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        """Generate a single probability prediction.

        Args:
            features: Dict mapping feature names to values.

        Returns:
            A :class:`ModelPrediction` with calibrated and raw probabilities.
        """

    @abstractmethod
    def predict_proba_batch(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Generate predictions for a batch of observations.

        Args:
            features: DataFrame with one row per observation and feature
                columns.

        Returns:
            List of :class:`ModelPrediction`, one per row.
        """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> None:
        """Train the model on labelled data.

        Args:
            X: Feature matrix.
            y: Binary outcome labels (0/1).
            sample_weights: Optional per-sample weights.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk.

        Args:
            path: File path for the serialised model.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ProbabilityModel:
        """Load a previously saved model from disk.

        Args:
            path: File path of the serialised model.

        Returns:
            A fully initialised model instance.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} version={self.version}>"
