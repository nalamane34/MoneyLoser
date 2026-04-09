"""Model drift detection via PSI (Population Stability Index) and KS test.

Monitors whether the distribution of model predictions has shifted
relative to a reference distribution captured during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import structlog
from scipy import stats

log = structlog.get_logger(__name__)

_EPS = 1e-8  # Prevent log(0) in PSI calculation


@dataclass(frozen=True)
class DriftResult:
    """Result of a drift check."""

    is_drifted: bool
    metric_name: str
    metric_value: float
    threshold: float
    severity: Literal["none", "warning", "critical"]


class DriftDetector:
    """Detects model drift by comparing recent predictions against a
    reference distribution using PSI and the Kolmogorov-Smirnov test.

    Parameters
    ----------
    reference_distribution:
        Array of model predictions from the training / validation period.
    window_size:
        Number of recent predictions to retain for comparison.
    n_bins:
        Number of bins for the PSI histogram.
    psi_warning:
        PSI threshold for a ``"warning"`` severity.
    psi_critical:
        PSI threshold for a ``"critical"`` severity.
    ks_critical_alpha:
        KS test p-value below which drift is flagged.
    """

    def __init__(
        self,
        reference_distribution: np.ndarray,
        window_size: int = 100,
        n_bins: int = 10,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        ks_critical_alpha: float = 0.01,
    ) -> None:
        self._reference = np.asarray(reference_distribution, dtype=float)
        self._window_size = window_size
        self._n_bins = n_bins
        self._psi_warning = psi_warning
        self._psi_critical = psi_critical
        self._ks_alpha = ks_critical_alpha
        self._reference_seeded = len(self._reference) > 0

        # Pre-compute reference histogram proportions
        self._bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        if self._reference_seeded:
            self._ref_proportions = self._histogram_proportions(self._reference)
        else:
            self._ref_proportions = np.ones(n_bins) / n_bins

        # Rolling buffer of recent predictions
        self._recent: deque[float] = deque(maxlen=window_size)

        # Feature-level buffers
        self._feature_buffers: dict[str, deque[float]] = {}

        log.info(
            "drift_detector.initialized",
            reference_size=len(self._reference),
            reference_seeded=self._reference_seeded,
            window_size=window_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_prediction(self, prediction: float) -> None:
        """Buffer a single model prediction (probability 0-1)."""
        self._recent.append(float(prediction))

    def set_reference(self, distribution: np.ndarray) -> None:
        """Update the reference distribution from real model predictions."""
        self._reference = np.asarray(distribution, dtype=float)
        self._ref_proportions = self._histogram_proportions(self._reference)
        self._reference_seeded = True
        log.info("drift_detector.reference_set", reference_size=len(self._reference))

    def check_drift(self) -> DriftResult:
        """Run PSI and KS drift checks on the buffered predictions.

        Returns the *worst* (highest severity) of the two tests.
        """
        if not self._reference_seeded or len(self._recent) < self._window_size:
            return DriftResult(
                is_drifted=False,
                metric_name="psi",
                metric_value=0.0,
                threshold=self._psi_warning,
                severity="none",
            )

        recent_arr = np.array(self._recent)

        # --- PSI check ---
        psi_value = self._compute_psi(recent_arr)
        psi_result = self._classify_psi(psi_value)

        # --- KS check ---
        ks_result = self._compute_ks(recent_arr)

        # Return whichever is worse
        severity_rank = {"none": 0, "warning": 1, "critical": 2}
        if severity_rank[ks_result.severity] > severity_rank[psi_result.severity]:
            return ks_result
        return psi_result

    def check_feature_drift(
        self, feature_name: str, values: list[float]
    ) -> DriftResult:
        """Check drift for a specific input feature.

        Unlike prediction drift, feature drift uses only the KS test
        because feature distributions are unbounded.

        Parameters
        ----------
        feature_name:
            Name of the feature being checked.
        values:
            Recent observed values for this feature.
        """
        if feature_name not in self._feature_buffers:
            self._feature_buffers[feature_name] = deque(maxlen=self._window_size)

        buf = self._feature_buffers[feature_name]
        buf.extend(values)

        if len(buf) < self._window_size:
            return DriftResult(
                is_drifted=False,
                metric_name=f"ks_{feature_name}",
                metric_value=0.0,
                threshold=self._ks_alpha,
                severity="none",
            )

        # Compare the latest window against the first window stored
        arr = np.array(buf)
        half = len(arr) // 2
        first_half = arr[:half]
        second_half = arr[half:]

        ks_stat, p_value = stats.ks_2samp(first_half, second_half)
        is_drifted = p_value < self._ks_alpha

        severity: Literal["none", "warning", "critical"]
        if not is_drifted:
            severity = "none"
        elif p_value < self._ks_alpha / 10:
            severity = "critical"
        else:
            severity = "warning"

        if is_drifted:
            log.warning(
                "feature_drift_detected",
                feature=feature_name,
                ks_stat=round(ks_stat, 4),
                p_value=round(p_value, 6),
                severity=severity,
            )

        return DriftResult(
            is_drifted=is_drifted,
            metric_name=f"ks_{feature_name}",
            metric_value=round(ks_stat, 6),
            threshold=self._ks_alpha,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _histogram_proportions(self, values: np.ndarray) -> np.ndarray:
        """Compute normalized histogram proportions with epsilon smoothing."""
        counts, _ = np.histogram(values, bins=self._bin_edges)
        proportions = counts / len(values)
        # Smooth zeros to avoid log(0)
        proportions = np.clip(proportions, _EPS, None)
        return proportions / proportions.sum()

    def _compute_psi(self, recent: np.ndarray) -> float:
        """Compute Population Stability Index between reference and recent."""
        recent_proportions = self._histogram_proportions(recent)
        ref = self._ref_proportions

        psi = np.sum((recent_proportions - ref) * np.log(recent_proportions / ref))
        return float(psi)

    def _classify_psi(self, psi_value: float) -> DriftResult:
        """Classify PSI value into severity levels."""
        if psi_value >= self._psi_critical:
            severity: Literal["none", "warning", "critical"] = "critical"
            is_drifted = True
        elif psi_value >= self._psi_warning:
            severity = "warning"
            is_drifted = True
        else:
            severity = "none"
            is_drifted = False

        if is_drifted:
            log.warning(
                "prediction_drift.psi",
                psi=round(psi_value, 4),
                severity=severity,
            )

        return DriftResult(
            is_drifted=is_drifted,
            metric_name="psi",
            metric_value=round(psi_value, 6),
            threshold=self._psi_warning,
            severity=severity,
        )

    def _compute_ks(self, recent: np.ndarray) -> DriftResult:
        """Compute KS test between reference and recent distributions."""
        ks_stat, p_value = stats.ks_2samp(self._reference, recent)

        is_drifted = p_value < self._ks_alpha
        severity: Literal["none", "warning", "critical"]
        if not is_drifted:
            severity = "none"
        elif p_value < self._ks_alpha / 10:
            severity = "critical"
        else:
            severity = "warning"

        if is_drifted:
            log.warning(
                "prediction_drift.ks",
                ks_stat=round(float(ks_stat), 4),
                p_value=round(float(p_value), 6),
                severity=severity,
            )

        return DriftResult(
            is_drifted=is_drifted,
            metric_name="ks",
            metric_value=round(float(ks_stat), 6),
            threshold=self._ks_alpha,
            severity=severity,
        )
