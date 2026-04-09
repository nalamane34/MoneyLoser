"""Bayesian logistic regression model for probability prediction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

log = structlog.get_logger()


class BayesianModel(ProbabilityModel):
    """Bayesian logistic regression using BayesianRidge on logit features.

    Uses scikit-learn's :class:`BayesianRidge` as a Bayesian linear
    regression on the feature space, with a sigmoid link to produce
    probabilities.  The predictive variance from BayesianRidge is used
    to compute a confidence score: wider uncertainty intervals yield
    lower confidence.

    Args:
        version: Version string.
        alpha_1: Shape parameter for the Gamma prior on alpha.
        alpha_2: Inverse scale parameter for the Gamma prior on alpha.
        lambda_1: Shape parameter for the Gamma prior on lambda.
        lambda_2: Inverse scale parameter for the Gamma prior on lambda.
    """

    name = "bayesian"

    def __init__(
        self,
        version: str = "0.1.0",
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
        self.version = version
        self.trained_at: datetime | None = None
        self._model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=True,
        )
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> None:
        """Train the Bayesian regression model.

        Transforms labels from {0,1} to a continuous target via the logit
        link (with clipping to avoid infinities), then fits BayesianRidge.
        """
        self._feature_names = list(X.columns)

        # Scale features
        X_scaled = self._scaler.fit_transform(X.values)

        # Transform binary labels to logit space for regression
        # Clip to avoid log(0)
        eps = 0.01
        y_vals = np.asarray(y, dtype=float)
        y_clipped = np.clip(y_vals, eps, 1.0 - eps)
        y_logit = np.log(y_clipped / (1.0 - y_clipped))

        weights = sample_weights.values if sample_weights is not None else None
        self._model.fit(X_scaled, y_logit, sample_weight=weights)

        self._fitted = True
        self.trained_at = datetime.now(timezone.utc)
        log.info(
            "bayesian_model_trained",
            n_samples=len(X),
            n_features=len(self._feature_names),
            alpha=float(self._model.alpha_),
            lambda_=float(self._model.lambda_),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        """Generate a single probability prediction with uncertainty."""
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        X_scaled = self._scaler.transform(X)

        # BayesianRidge returns mean and std of the predictive distribution
        y_mean, y_std = self._model.predict(X_scaled, return_std=True)
        logit_pred = float(y_mean[0])
        logit_std = float(y_std[0])

        # Sigmoid to get probability
        raw_prob = self._sigmoid(logit_pred)
        prob = raw_prob  # no separate calibration for Bayesian model

        # Confidence: inversely related to predictive std
        confidence = self._std_to_confidence(logit_std)

        return ModelPrediction(
            probability=float(np.clip(prob, 0.0, 1.0)),
            raw_probability=raw_prob,
            confidence=confidence,
            model_name=self.name,
            model_version=self.version,
            features_used=features,
            prediction_time=datetime.now(timezone.utc),
        )

    def predict_proba_batch(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Generate predictions for a batch of observations."""
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X = features.reindex(columns=self._feature_names, fill_value=0.0).values
        X_scaled = self._scaler.transform(X)

        y_means, y_stds = self._model.predict(X_scaled, return_std=True)

        now = datetime.now(timezone.utc)
        predictions = []
        for i in range(len(features)):
            logit_pred = float(y_means[i])
            logit_std = float(y_stds[i])
            raw_prob = self._sigmoid(logit_pred)
            confidence = self._std_to_confidence(logit_std)

            row_features = {col: float(features.iloc[i][col]) for col in features.columns}
            predictions.append(
                ModelPrediction(
                    probability=float(np.clip(raw_prob, 0.0, 1.0)),
                    raw_probability=raw_prob,
                    confidence=confidence,
                    model_name=self.name,
                    model_version=self.version,
                    features_used=row_features,
                    prediction_time=now,
                )
            )
        return predictions

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "version": self.version,
                "trained_at": self.trained_at,
            },
            path,
        )
        log.info("bayesian_model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> BayesianModel:
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(version=data["version"])
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        instance._feature_names = data["feature_names"]
        instance._fitted = True
        instance.trained_at = data.get("trained_at")
        log.info("bayesian_model_loaded", path=str(path))
        return instance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    @staticmethod
    def _std_to_confidence(std: float) -> float:
        """Convert predictive standard deviation to a confidence score.

        Uses an exponential decay: confidence = exp(-std).
        High std -> low confidence, low std -> high confidence.
        """
        return float(np.clip(np.exp(-std), 0.0, 1.0))
