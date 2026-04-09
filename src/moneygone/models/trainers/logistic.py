"""Logistic regression model for probability prediction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.models.calibration import Calibrator

log = structlog.get_logger()


class LogisticModel(ProbabilityModel):
    """Logistic regression wrapper implementing :class:`ProbabilityModel`.

    Wraps scikit-learn's :class:`LogisticRegression` and optionally
    applies post-hoc calibration via :class:`Calibrator`.

    Args:
        version: Version string for this model instance.
        calibration_method: Calibration method (None to skip).
        C: Regularisation strength (inverse).
        max_iter: Maximum solver iterations.
    """

    name = "logistic"

    def __init__(
        self,
        version: str = "0.1.0",
        calibration_method: str | None = "isotonic",
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.version = version
        self.trained_at: datetime | None = None
        self._calibration_method = calibration_method
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self._calibrator: Calibrator | None = None
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
        """Train the logistic regression model.

        If a calibration method is set, splits the data 80/20 to fit
        the calibrator on a held-out set.
        """
        self._feature_names = list(X.columns)
        weights = sample_weights.values if sample_weights is not None else None

        if self._calibration_method is not None and len(X) >= 50:
            # Split for calibration
            n_train = int(len(X) * 0.8)
            X_train, X_cal = X.iloc[:n_train], X.iloc[n_train:]
            y_train, y_cal = y.iloc[:n_train], y.iloc[n_train:]
            w_train = weights[:n_train] if weights is not None else None

            self._model.fit(X_train.values, y_train.values, sample_weight=w_train)

            # Calibrate on held-out set
            raw_probs = self._model.predict_proba(X_cal.values)[:, 1]
            self._calibrator = Calibrator(method=self._calibration_method)
            self._calibrator.fit(raw_probs, y_cal.values)
        else:
            self._model.fit(X.values, y.values, sample_weight=weights)

        self._fitted = True
        self.trained_at = datetime.now(timezone.utc)
        log.info(
            "logistic_model_trained",
            n_samples=len(X),
            n_features=len(self._feature_names),
            calibrated=self._calibrator is not None,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        """Generate a single probability prediction."""
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        raw_prob = float(self._model.predict_proba(X)[0, 1])

        if self._calibrator is not None:
            prob = self._calibrator.calibrate(raw_prob)
        else:
            prob = raw_prob

        return ModelPrediction(
            probability=float(np.clip(prob, 0.0, 1.0)),
            raw_probability=raw_prob,
            confidence=self._compute_confidence(raw_prob),
            model_name=self.name,
            model_version=self.version,
            features_used=features,
            prediction_time=datetime.now(timezone.utc),
        )

    def predict_proba_batch(self, features: pd.DataFrame) -> list[ModelPrediction]:
        """Generate predictions for a batch of observations."""
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        # Align columns to training feature order
        X = features.reindex(columns=self._feature_names, fill_value=0.0).values
        raw_probs = self._model.predict_proba(X)[:, 1]

        if self._calibrator is not None:
            probs = self._calibrator.calibrate_batch(raw_probs)
        else:
            probs = raw_probs

        now = datetime.now(timezone.utc)
        predictions = []
        for i in range(len(features)):
            row_features = {col: float(features.iloc[i][col]) for col in features.columns}
            predictions.append(
                ModelPrediction(
                    probability=float(np.clip(probs[i], 0.0, 1.0)),
                    raw_probability=float(raw_probs[i]),
                    confidence=self._compute_confidence(float(raw_probs[i])),
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
                "calibrator": self._calibrator,
                "feature_names": self._feature_names,
                "version": self.version,
                "trained_at": self.trained_at,
                "calibration_method": self._calibration_method,
            },
            path,
        )
        log.info("logistic_model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> LogisticModel:
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(
            version=data["version"],
            calibration_method=data.get("calibration_method"),
        )
        instance._model = data["model"]
        instance._calibrator = data.get("calibrator")
        instance._feature_names = data["feature_names"]
        instance._fitted = True
        instance.trained_at = data.get("trained_at")
        log.info("logistic_model_loaded", path=str(path))
        return instance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(raw_prob: float) -> float:
        """Compute a confidence score based on distance from 0.5.

        Predictions closer to 0 or 1 get higher confidence.
        """
        return 2.0 * abs(raw_prob - 0.5)
