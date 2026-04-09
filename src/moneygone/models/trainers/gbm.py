"""Gradient boosted model for probability prediction using LightGBM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.models.calibration import Calibrator

log = structlog.get_logger()


class GBMModel(ProbabilityModel):
    """LightGBM gradient boosted model implementing :class:`ProbabilityModel`.

    Wraps :class:`LGBMClassifier` with configurable hyperparameters,
    early stopping on a validation split, and optional post-hoc
    calibration.

    Args:
        version: Version string.
        calibration_method: Calibration method (None to skip).
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth (-1 for no limit).
        num_leaves: Maximum number of leaves per tree.
        min_child_samples: Minimum samples per leaf.
        subsample: Row sampling ratio per tree.
        colsample_bytree: Feature sampling ratio per tree.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        early_stopping_rounds: Patience for early stopping.
        val_fraction: Fraction of training data held out for validation.
    """

    name = "gbm"

    def __init__(
        self,
        version: str = "0.1.0",
        calibration_method: str | None = "isotonic",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.15,
    ) -> None:
        self.version = version
        self.trained_at: datetime | None = None
        self._calibration_method = calibration_method
        self._early_stopping_rounds = early_stopping_rounds
        self._val_fraction = val_fraction

        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=-1,
        )
        self._calibrator: Calibrator | None = None
        self._feature_names: list[str] = []
        self._feature_importances: dict[str, float] = {}
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
        """Train the GBM with early stopping on a validation split."""
        self._feature_names = list(X.columns)
        weights = sample_weights.values if sample_weights is not None else None

        # Split into train / validation / calibration
        n = len(X)
        n_val = max(int(n * self._val_fraction), 10)
        n_cal = max(int(n * 0.10), 10) if self._calibration_method else 0
        n_train = n - n_val - n_cal

        if n_train < 20:
            # Not enough data to split -- train on everything
            self._model.fit(X.values, y.values, sample_weight=weights)
        else:
            X_train = X.iloc[:n_train]
            y_train = y.iloc[:n_train]
            X_val = X.iloc[n_train : n_train + n_val]
            y_val = y.iloc[n_train : n_train + n_val]
            w_train = weights[:n_train] if weights is not None else None

            callbacks = [
                _lgbm_early_stopping(self._early_stopping_rounds),
                _lgbm_log_evaluation(50),
            ]

            self._model.fit(
                X_train.values,
                y_train.values,
                sample_weight=w_train,
                eval_set=[(X_val.values, y_val.values)],
                callbacks=callbacks,
            )

            # Calibration on held-out set
            if self._calibration_method and n_cal > 0:
                X_cal = X.iloc[n_train + n_val :]
                y_cal = y.iloc[n_train + n_val :]
                raw_probs = self._model.predict_proba(X_cal.values)[:, 1]
                self._calibrator = Calibrator(method=self._calibration_method)
                self._calibrator.fit(raw_probs, y_cal.values)

        # Record feature importances
        importances = self._model.feature_importances_
        self._feature_importances = {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

        self._fitted = True
        self.trained_at = datetime.now(timezone.utc)
        log.info(
            "gbm_model_trained",
            n_samples=n,
            n_features=len(self._feature_names),
            best_iteration=getattr(self._model, "best_iteration_", None),
            calibrated=self._calibrator is not None,
        )

    @property
    def feature_importances(self) -> dict[str, float]:
        """Feature importance scores from the trained GBM."""
        return dict(self._feature_importances)

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
                "feature_importances": self._feature_importances,
                "version": self.version,
                "trained_at": self.trained_at,
                "calibration_method": self._calibration_method,
            },
            path,
        )
        log.info("gbm_model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> GBMModel:
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(
            version=data["version"],
            calibration_method=data.get("calibration_method"),
        )
        instance._model = data["model"]
        instance._calibrator = data.get("calibrator")
        instance._feature_names = data["feature_names"]
        instance._feature_importances = data.get("feature_importances", {})
        instance._fitted = True
        instance.trained_at = data.get("trained_at")
        log.info("gbm_model_loaded", path=str(path))
        return instance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(raw_prob: float) -> float:
        """Confidence based on distance from 0.5."""
        return 2.0 * abs(raw_prob - 0.5)


# ---------------------------------------------------------------------------
# LightGBM callback helpers
# ---------------------------------------------------------------------------

def _lgbm_early_stopping(stopping_rounds: int):
    """Return a LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)


def _lgbm_log_evaluation(period: int):
    """Return a LightGBM log evaluation callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)
