"""Rolling calibration quality monitor.

Tracks prediction vs. outcome correspondence over a sliding window and
computes calibration metrics (Brier score, ECE, log-loss) so the system
can detect when model calibration degrades.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Snapshot of rolling calibration quality."""

    brier_score: float
    """Mean squared error of predicted probabilities vs. outcomes."""

    ece: float
    """Expected Calibration Error -- average absolute gap between
    predicted probability and observed frequency across bins."""

    log_loss: float
    """Mean negative log-likelihood."""

    n_resolved: int
    """Number of predictions with known outcomes in the window."""


class CalibrationMonitor:
    """Tracks calibration quality in a rolling window.

    Predictions and outcomes are matched by ticker.  The monitor
    accumulates ``(predicted_prob, outcome)`` pairs and computes
    calibration metrics on demand.

    Parameters
    ----------
    window_size:
        Maximum number of resolved predictions to retain.
    n_bins:
        Number of bins for ECE / reliability diagram.
    ece_threshold:
        ECE value above which :meth:`is_degraded` returns ``True``.
    """

    def __init__(
        self,
        window_size: int = 500,
        n_bins: int = 10,
        ece_threshold: float = 0.05,
    ) -> None:
        self._window_size = window_size
        self._n_bins = n_bins
        self._ece_threshold = ece_threshold

        # Pending predictions awaiting outcomes: ticker -> predicted_prob
        self._pending: dict[str, float] = {}

        # Resolved pairs: ordered dict ticker -> (predicted_prob, outcome)
        self._resolved: OrderedDict[str, tuple[float, bool]] = OrderedDict()

        log.info(
            "calibration_monitor.initialized",
            window_size=window_size,
            ece_threshold=ece_threshold,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(self, predicted_prob: float, ticker: str) -> None:
        """Record a model prediction for a market.

        Parameters
        ----------
        predicted_prob:
            Calibrated probability of YES outcome (0-1).
        ticker:
            Market ticker (used to match with outcome later).
        """
        self._pending[ticker] = predicted_prob

    def record_outcome(self, ticker: str, outcome: bool) -> None:
        """Record the actual outcome for a market.

        Parameters
        ----------
        ticker:
            Market ticker previously recorded with :meth:`record_prediction`.
        outcome:
            ``True`` if the market settled YES, ``False`` for NO.
        """
        if ticker not in self._pending:
            log.debug("calibration.outcome_without_prediction", ticker=ticker)
            return

        predicted = self._pending.pop(ticker)
        self._resolved[ticker] = (predicted, outcome)

        # Evict oldest if over capacity
        while len(self._resolved) > self._window_size:
            self._resolved.popitem(last=False)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_rolling_metrics(self) -> CalibrationMetrics:
        """Compute calibration metrics over the resolved window.

        Returns default (zero) metrics if no predictions have been resolved.
        """
        if not self._resolved:
            return CalibrationMetrics(
                brier_score=0.0, ece=0.0, log_loss=0.0, n_resolved=0
            )

        preds = np.array([p for p, _ in self._resolved.values()])
        outcomes = np.array(
            [float(o) for _, o in self._resolved.values()]
        )

        brier = float(np.mean((preds - outcomes) ** 2))
        ece = self._compute_ece(preds, outcomes)
        ll = self._compute_log_loss(preds, outcomes)

        return CalibrationMetrics(
            brier_score=round(brier, 6),
            ece=round(ece, 6),
            log_loss=round(ll, 6),
            n_resolved=len(self._resolved),
        )

    def is_degraded(self) -> bool:
        """Return ``True`` if ECE exceeds the configured threshold."""
        metrics = self.get_rolling_metrics()
        if metrics.n_resolved < 20:
            # Not enough data to judge
            return False
        degraded = metrics.ece > self._ece_threshold
        if degraded:
            log.warning(
                "calibration.degraded",
                ece=metrics.ece,
                threshold=self._ece_threshold,
                n_resolved=metrics.n_resolved,
            )
        return degraded

    def get_reliability_data(
        self,
    ) -> tuple[list[float], list[float], list[int]]:
        """Return data for a reliability diagram.

        Returns
        -------
        tuple
            ``(bin_centers, bin_fractions, bin_counts)`` where:
            - *bin_centers*: midpoint of each probability bin
            - *bin_fractions*: observed frequency of YES in each bin
            - *bin_counts*: number of predictions in each bin
        """
        if not self._resolved:
            return [], [], []

        preds = np.array([p for p, _ in self._resolved.values()])
        outcomes = np.array(
            [float(o) for _, o in self._resolved.values()]
        )

        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        bin_centers: list[float] = []
        bin_fractions: list[float] = []
        bin_counts: list[int] = []

        for i in range(self._n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == self._n_bins - 1:
                mask = (preds >= low) & (preds <= high)
            else:
                mask = (preds >= low) & (preds < high)

            count = int(mask.sum())
            bin_counts.append(count)
            bin_centers.append(float((low + high) / 2))

            if count > 0:
                bin_fractions.append(float(outcomes[mask].mean()))
            else:
                bin_fractions.append(0.0)

        return bin_centers, bin_fractions, bin_counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ece(self, preds: np.ndarray, outcomes: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        ece = 0.0
        n_total = len(preds)

        for i in range(self._n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == self._n_bins - 1:
                mask = (preds >= low) & (preds <= high)
            else:
                mask = (preds >= low) & (preds < high)

            count = mask.sum()
            if count == 0:
                continue

            avg_pred = preds[mask].mean()
            avg_outcome = outcomes[mask].mean()
            ece += (count / n_total) * abs(avg_pred - avg_outcome)

        return float(ece)

    @staticmethod
    def _compute_log_loss(preds: np.ndarray, outcomes: np.ndarray) -> float:
        """Compute mean log-loss, clipping predictions to avoid log(0)."""
        eps = 1e-15
        clipped = np.clip(preds, eps, 1.0 - eps)
        ll = -(
            outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)
        )
        return float(np.mean(ll))
