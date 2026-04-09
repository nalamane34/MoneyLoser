"""Probability calibration methods for post-hoc model calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import structlog

log = structlog.get_logger()


class Calibrator:
    """Post-hoc probability calibrator.

    Supports three calibration methods:

    - **isotonic**: Non-parametric isotonic regression.  Best when you
      have enough calibration data (>1000 samples).
    - **platt**: Logistic regression on the log-odds (Platt scaling).
      Works well with fewer samples.
    - **beta**: Beta calibration -- fits a logistic regression on
      log(p) and log(1-p) as two separate features, allowing asymmetric
      corrections.

    Usage::

        cal = Calibrator(method="isotonic")
        cal.fit(raw_probs, outcomes)
        calibrated = cal.calibrate(0.7)
        batch = cal.calibrate_batch(np.array([0.3, 0.5, 0.9]))
    """

    VALID_METHODS = ("isotonic", "platt", "beta")

    def __init__(self, method: str = "isotonic") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid calibration method '{method}'. "
                f"Choose from {self.VALID_METHODS}."
            )
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, raw_probs: npt.ArrayLike, outcomes: npt.ArrayLike) -> None:
        """Fit the calibrator on raw probabilities and true outcomes.

        Args:
            raw_probs: Model output probabilities in [0, 1].
            outcomes: True binary labels (0 or 1).
        """
        p = np.asarray(raw_probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)

        if len(p) != len(y):
            raise ValueError("raw_probs and outcomes must have the same length.")
        if len(p) < 2:
            raise ValueError("Need at least 2 samples for calibration.")

        if self.method == "isotonic":
            self._fit_isotonic(p, y)
        elif self.method == "platt":
            self._fit_platt(p, y)
        elif self.method == "beta":
            self._fit_beta(p, y)

        self._fitted = True
        log.info("calibrator_fitted", method=self.method, n_samples=len(p))

    def _fit_isotonic(self, p: np.ndarray, y: np.ndarray) -> None:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p, y)
        self._model = model

    def _fit_platt(self, p: np.ndarray, y: np.ndarray) -> None:
        eps = 1e-12
        p_clipped = np.clip(p, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        model.fit(logits, y)
        self._model = model

    def _fit_beta(self, p: np.ndarray, y: np.ndarray) -> None:
        eps = 1e-12
        p_clipped = np.clip(p, eps, 1.0 - eps)
        # Beta calibration uses log(p) and log(1-p) as features
        features = np.column_stack([np.log(p_clipped), np.log(1.0 - p_clipped)])
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        model.fit(features, y)
        self._model = model

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, raw_prob: float) -> float:
        """Calibrate a single raw probability.

        Args:
            raw_prob: Uncalibrated probability in [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        result = self.calibrate_batch(np.array([raw_prob]))
        return float(result[0])

    def calibrate_batch(self, raw_probs: npt.ArrayLike) -> np.ndarray:
        """Calibrate a batch of raw probabilities.

        Args:
            raw_probs: Array of uncalibrated probabilities.

        Returns:
            Array of calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        p = np.asarray(raw_probs, dtype=float)
        eps = 1e-12

        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)

        elif self.method == "platt":
            p_clipped = np.clip(p, eps, 1.0 - eps)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            return self._model.predict_proba(logits)[:, 1]

        elif self.method == "beta":
            p_clipped = np.clip(p, eps, 1.0 - eps)
            features = np.column_stack([np.log(p_clipped), np.log(1.0 - p_clipped)])
            return self._model.predict_proba(features)[:, 1]

        raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the calibrator to disk via joblib.

        Args:
            path: File path for the serialised calibrator.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"method": self.method, "model": self._model, "fitted": self._fitted},
            path,
        )
        log.info("calibrator_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        """Load a previously saved calibrator from disk.

        Args:
            path: File path of the serialised calibrator.

        Returns:
            A fitted :class:`Calibrator` instance.
        """
        data = joblib.load(path)
        cal = cls(method=data["method"])
        cal._model = data["model"]
        cal._fitted = data["fitted"]
        log.info("calibrator_loaded", path=str(path), method=cal.method)
        return cal
