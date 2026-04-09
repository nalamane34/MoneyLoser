"""Model evaluation metrics for probability calibration and accuracy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import structlog

log = structlog.get_logger()


class ModelEvaluator:
    """Static methods for evaluating probabilistic predictions.

    All methods accept arrays of predicted probabilities and binary
    outcomes (0 or 1).
    """

    @staticmethod
    def brier_score(
        probs: npt.ArrayLike,
        outcomes: npt.ArrayLike,
    ) -> float:
        """Compute the Brier score (mean squared error of probabilities).

        Lower is better.  Range [0, 1].  A perfect model scores 0.
        """
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        return float(np.mean((p - y) ** 2))

    @staticmethod
    def log_loss(
        probs: npt.ArrayLike,
        outcomes: npt.ArrayLike,
        eps: float = 1e-15,
    ) -> float:
        """Compute the log loss (cross-entropy) with probability clipping.

        Clipping prevents infinite loss from extreme predictions.

        Args:
            probs: Predicted probabilities.
            outcomes: Binary outcomes (0/1).
            eps: Clipping bound for probabilities.

        Returns:
            Mean log loss (lower is better).
        """
        p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
        y = np.asarray(outcomes, dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    @staticmethod
    def expected_calibration_error(
        probs: npt.ArrayLike,
        outcomes: npt.ArrayLike,
        n_bins: int = 10,
    ) -> float:
        """Compute the Expected Calibration Error (ECE).

        ECE measures the weighted average absolute difference between
        predicted confidence and observed accuracy across bins.

        Args:
            probs: Predicted probabilities.
            outcomes: Binary outcomes (0/1).
            n_bins: Number of equally-spaced bins in [0, 1].

        Returns:
            ECE value (lower is better).
        """
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        n = len(p)

        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
            # Include lower bound for first bin
            if i == 0:
                mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])

            bin_count = mask.sum()
            if bin_count == 0:
                continue

            bin_accuracy = y[mask].mean()
            bin_confidence = p[mask].mean()
            ece += (bin_count / n) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    @staticmethod
    def reliability_diagram(
        probs: npt.ArrayLike,
        outcomes: npt.ArrayLike,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute data for a reliability (calibration) diagram.

        Args:
            probs: Predicted probabilities.
            outcomes: Binary outcomes (0/1).
            n_bins: Number of bins.

        Returns:
            Tuple of (bin_centers, bin_true_fractions, bin_counts).
            Empty bins are included as NaN in bin_true_fractions with
            count 0.
        """
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_true_fractions = np.full(n_bins, np.nan)
        bin_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            if i == 0:
                mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
            else:
                mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])

            count = mask.sum()
            bin_counts[i] = count
            if count > 0:
                bin_true_fractions[i] = y[mask].mean()

        return bin_centers, bin_true_fractions, bin_counts

    @staticmethod
    def sharpness(probs: npt.ArrayLike) -> float:
        """Compute the sharpness (variance) of predictions.

        Higher sharpness means the model is making more decisive
        predictions rather than clustering around 0.5.
        """
        p = np.asarray(probs, dtype=float)
        if len(p) == 0:
            return 0.0
        return float(np.var(p))

    @classmethod
    def evaluate_all(
        cls,
        probs: npt.ArrayLike,
        outcomes: npt.ArrayLike,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Run all evaluation metrics and return as a dict.

        Args:
            probs: Predicted probabilities.
            outcomes: Binary outcomes (0/1).
            n_bins: Number of bins for ECE.

        Returns:
            Dict with keys: brier_score, log_loss, ece, sharpness.
        """
        return {
            "brier_score": cls.brier_score(probs, outcomes),
            "log_loss": cls.log_loss(probs, outcomes),
            "ece": cls.expected_calibration_error(probs, outcomes, n_bins=n_bins),
            "sharpness": cls.sharpness(probs),
        }
