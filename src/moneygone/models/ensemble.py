"""Ensemble model that aggregates predictions from multiple sub-models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.models.evaluation import ModelEvaluator

log = structlog.get_logger()


class EnsembleModel(ProbabilityModel):
    """Ensemble model that combines predictions from multiple sub-models.

    Supports three aggregation methods:

    - **simple_average**: Equal-weight average of all sub-model probabilities.
    - **inverse_variance**: Weight each model by 1/variance of its recent
      Brier scores (better recent performance = higher weight).
    - **stacking**: Learns optimal combination weights via logistic
      regression on sub-model predictions.

    Args:
        models: List of trained :class:`ProbabilityModel` instances.
        method: Aggregation method.
        version: Version string for the ensemble.
    """

    name = "ensemble"

    def __init__(
        self,
        models: list[ProbabilityModel],
        method: str = "inverse_variance",
        version: str = "0.1.0",
    ) -> None:
        if not models:
            raise ValueError("Ensemble requires at least one sub-model.")
        if method not in ("simple_average", "inverse_variance", "stacking"):
            raise ValueError(
                f"Invalid ensemble method '{method}'. "
                "Choose from: simple_average, inverse_variance, stacking."
            )

        self.version = version
        self.trained_at: datetime | None = None
        self._models = models
        self._method = method
        self._weights: np.ndarray = np.ones(len(models)) / len(models)
        self._stacking_model: object | None = None  # LogisticRegression for stacking
        self._fitted = method == "simple_average"  # simple_average needs no fitting

    @property
    def sub_models(self) -> list[ProbabilityModel]:
        """The constituent sub-models."""
        return list(self._models)

    @property
    def weights(self) -> np.ndarray:
        """Current model weights."""
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Training / weight updates
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> None:
        """Fit the ensemble combination weights.

        For ``simple_average`` this is a no-op.
        For ``inverse_variance`` this computes weights from Brier scores.
        For ``stacking`` this trains a logistic regression on sub-model
        predictions.

        The sub-models must already be trained before calling this.
        """
        if self._method == "simple_average":
            self._weights = np.ones(len(self._models)) / len(self._models)
            self._fitted = True
            return

        # Get predictions from each sub-model
        sub_predictions = np.zeros((len(X), len(self._models)))
        for j, model in enumerate(self._models):
            preds = model.predict_proba_batch(X)
            sub_predictions[:, j] = [p.probability for p in preds]

        y_arr = np.asarray(y, dtype=float)

        if self._method == "inverse_variance":
            self._fit_inverse_variance(sub_predictions, y_arr)
        elif self._method == "stacking":
            self._fit_stacking(sub_predictions, y_arr)

        self._fitted = True
        self.trained_at = datetime.now(timezone.utc)
        log.info(
            "ensemble_fitted",
            method=self._method,
            weights=self._weights.tolist(),
            n_models=len(self._models),
        )

    def _fit_inverse_variance(
        self, sub_predictions: np.ndarray, y: np.ndarray
    ) -> None:
        """Compute inverse-variance weights from per-model Brier scores."""
        n_models = sub_predictions.shape[1]
        brier_scores = np.zeros(n_models)

        for j in range(n_models):
            brier_scores[j] = ModelEvaluator.brier_score(sub_predictions[:, j], y)

        # Inverse of Brier score (lower Brier = better = higher weight)
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        inv_brier = 1.0 / (brier_scores + eps)
        self._weights = inv_brier / inv_brier.sum()

    def _fit_stacking(self, sub_predictions: np.ndarray, y: np.ndarray) -> None:
        """Fit a stacking meta-learner on sub-model predictions."""
        from sklearn.linear_model import LogisticRegression

        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        meta.fit(sub_predictions, y)
        self._stacking_model = meta

        # Set weights from the stacking coefficients (for interpretability)
        coefs = np.abs(meta.coef_[0])
        if coefs.sum() > 0:
            self._weights = coefs / coefs.sum()
        else:
            self._weights = np.ones(len(self._models)) / len(self._models)

    def update_weights(self, recent_outcomes: pd.DataFrame) -> None:
        """Recalibrate ensemble weights using recent prediction outcomes.

        Args:
            recent_outcomes: DataFrame with columns ``probability_<model_name>``
                for each sub-model's predictions and ``outcome`` for the
                true binary result.
        """
        y = recent_outcomes["outcome"].values
        n_models = len(self._models)
        sub_preds = np.zeros((len(recent_outcomes), n_models))

        for j, model in enumerate(self._models):
            col = f"probability_{model.name}"
            if col in recent_outcomes.columns:
                sub_preds[:, j] = recent_outcomes[col].values
            else:
                # Fallback: equal weight if column missing
                sub_preds[:, j] = 0.5

        if self._method == "inverse_variance":
            self._fit_inverse_variance(sub_preds, y)
        elif self._method == "stacking":
            self._fit_stacking(sub_preds, y)
        else:
            self._weights = np.ones(n_models) / n_models

        log.info(
            "ensemble_weights_updated",
            weights=self._weights.tolist(),
            n_outcomes=len(recent_outcomes),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        """Aggregate predictions from all sub-models."""
        sub_preds = []
        sub_raw = []
        sub_conf = []

        for model in self._models:
            pred = model.predict_proba(features)
            sub_preds.append(pred.probability)
            sub_raw.append(pred.raw_probability)
            sub_conf.append(pred.confidence)

        probs_arr = np.array(sub_preds)
        raw_arr = np.array(sub_raw)

        if self._method == "stacking" and self._stacking_model is not None:
            prob = float(
                self._stacking_model.predict_proba(probs_arr.reshape(1, -1))[0, 1]
            )
        else:
            prob = float(np.dot(self._weights, probs_arr))

        raw_prob = float(np.dot(self._weights, raw_arr))
        confidence = float(np.dot(self._weights, np.array(sub_conf)))

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
        """Generate ensemble predictions for a batch."""
        n = len(features)
        n_models = len(self._models)

        # Collect sub-model predictions
        all_probs = np.zeros((n, n_models))
        all_raw = np.zeros((n, n_models))
        all_conf = np.zeros((n, n_models))

        for j, model in enumerate(self._models):
            preds = model.predict_proba_batch(features)
            for i, pred in enumerate(preds):
                all_probs[i, j] = pred.probability
                all_raw[i, j] = pred.raw_probability
                all_conf[i, j] = pred.confidence

        # Aggregate
        if self._method == "stacking" and self._stacking_model is not None:
            probs = self._stacking_model.predict_proba(all_probs)[:, 1]
        else:
            probs = all_probs @ self._weights

        raw_probs = all_raw @ self._weights
        confidences = all_conf @ self._weights

        now = datetime.now(timezone.utc)
        predictions = []
        for i in range(n):
            row_features = {col: float(features.iloc[i][col]) for col in features.columns}
            predictions.append(
                ModelPrediction(
                    probability=float(np.clip(probs[i], 0.0, 1.0)),
                    raw_probability=float(raw_probs[i]),
                    confidence=float(confidences[i]),
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
        """Save the ensemble (weights and stacking model) to disk.

        Sub-models must be saved separately.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self._method,
                "weights": self._weights,
                "stacking_model": self._stacking_model,
                "model_names": [m.name for m in self._models],
                "version": self.version,
                "trained_at": self.trained_at,
            },
            path,
        )
        log.info("ensemble_saved", path=str(path))

    @classmethod
    def load(cls, path: Path, models: list[ProbabilityModel] | None = None) -> EnsembleModel:
        """Load ensemble weights from disk.

        Args:
            path: File path of the serialised ensemble.
            models: List of already-loaded sub-models.  Must match the
                model names stored at save time.

        Returns:
            A restored :class:`EnsembleModel`.
        """
        data = joblib.load(path)
        if models is None:
            raise ValueError(
                "Sub-models must be provided when loading an ensemble. "
                "Pass the same models used during training."
            )

        instance = cls(
            models=models,
            method=data["method"],
            version=data["version"],
        )
        instance._weights = data["weights"]
        instance._stacking_model = data.get("stacking_model")
        instance._fitted = True
        instance.trained_at = data.get("trained_at")
        log.info("ensemble_loaded", path=str(path), method=data["method"])
        return instance
