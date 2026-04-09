"""Market regime detection using rolling volatility and volume analysis.

Classifies the current market environment into one of four regimes so
the trading system can adapt position sizing and model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RegimeState:
    """Detected market regime."""

    regime: Literal["low_vol", "normal", "high_vol", "crisis"]
    """Regime classification."""

    volatility: float
    """Current exponentially weighted realized volatility."""

    vol_percentile: float
    """Percentile of current volatility within the lookback window (0-100)."""


class RegimeDetector:
    """Detects market regime changes using rolling volatility and volume.

    Regime classifications:

    * ``low_vol`` -- volatility below 25th percentile
    * ``normal`` -- volatility between 25th and 75th percentile
    * ``high_vol`` -- volatility between 75th and 95th percentile
    * ``crisis`` -- volatility above 95th percentile

    Parameters
    ----------
    ewm_span:
        Span for the exponentially weighted volatility calculation.
    low_vol_pct:
        Percentile threshold below which the regime is ``low_vol``.
    high_vol_pct:
        Percentile threshold above which the regime is ``high_vol``.
    crisis_pct:
        Percentile threshold above which the regime is ``crisis``.
    """

    def __init__(
        self,
        ewm_span: int = 20,
        low_vol_pct: float = 25.0,
        high_vol_pct: float = 75.0,
        crisis_pct: float = 95.0,
    ) -> None:
        self._ewm_span = ewm_span
        self._low_vol_pct = low_vol_pct
        self._high_vol_pct = high_vol_pct
        self._crisis_pct = crisis_pct
        self._last_regime: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        prices: list[float],
        volumes: list[int],
        lookback: int = 100,
    ) -> RegimeState:
        """Classify the current market regime.

        Parameters
        ----------
        prices:
            Chronologically ordered price series.  Needs at least
            ``lookback`` observations for meaningful results.
        volumes:
            Corresponding volume series (same length as *prices*).
        lookback:
            Number of historical observations used to establish
            volatility percentile context.

        Returns
        -------
        RegimeState
            Current regime with volatility metrics.
        """
        if len(prices) < 3:
            return RegimeState(
                regime="normal", volatility=0.0, vol_percentile=50.0
            )

        price_arr = np.array(prices, dtype=float)

        # Compute log returns
        log_returns = np.diff(np.log(np.clip(price_arr, 1e-10, None)))
        if len(log_returns) == 0:
            return RegimeState(
                regime="normal", volatility=0.0, vol_percentile=50.0
            )

        # Exponentially weighted realized volatility
        ewm_vol = self._ewm_std(log_returns, span=self._ewm_span)

        # Use lookback window for percentile context
        vol_series = self._rolling_ewm_vol(
            log_returns, span=self._ewm_span, window=lookback
        )
        current_vol = ewm_vol

        if len(vol_series) > 1:
            vol_percentile = float(
                np.searchsorted(np.sort(vol_series), current_vol)
                / len(vol_series)
                * 100
            )
        else:
            vol_percentile = 50.0

        # Volume-adjusted severity: high volume amplifies regime detection
        vol_arr = np.array(volumes, dtype=float)
        if len(vol_arr) > lookback:
            recent_vol_mean = vol_arr[-lookback:].mean()
            overall_vol_mean = vol_arr.mean()
            if overall_vol_mean > 0:
                volume_ratio = recent_vol_mean / overall_vol_mean
                # Boost percentile when volume is unusually high
                if volume_ratio > 1.5:
                    vol_percentile = min(100.0, vol_percentile * 1.1)

        # Classify
        regime = self._classify(vol_percentile)

        # Log regime transitions
        if self._last_regime is not None and regime != self._last_regime:
            log.info(
                "regime_transition",
                from_regime=self._last_regime,
                to_regime=regime,
                volatility=round(current_vol, 6),
                vol_percentile=round(vol_percentile, 1),
            )
        self._last_regime = regime

        return RegimeState(
            regime=regime,
            volatility=round(current_vol, 6),
            vol_percentile=round(vol_percentile, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self, vol_percentile: float
    ) -> Literal["low_vol", "normal", "high_vol", "crisis"]:
        """Map volatility percentile to regime label."""
        if vol_percentile >= self._crisis_pct:
            return "crisis"
        if vol_percentile >= self._high_vol_pct:
            return "high_vol"
        if vol_percentile <= self._low_vol_pct:
            return "low_vol"
        return "normal"

    @staticmethod
    def _ewm_std(values: np.ndarray, span: int) -> float:
        """Compute exponentially weighted standard deviation."""
        alpha = 2.0 / (span + 1)
        n = len(values)
        if n == 0:
            return 0.0

        # Compute EWM mean
        weights = (1 - alpha) ** np.arange(n - 1, -1, -1)
        ewm_mean = np.average(values, weights=weights)

        # Compute EWM variance
        ewm_var = np.average((values - ewm_mean) ** 2, weights=weights)
        return float(np.sqrt(ewm_var))

    @staticmethod
    def _rolling_ewm_vol(
        returns: np.ndarray, span: int, window: int
    ) -> np.ndarray:
        """Compute a series of EWM volatilities over a rolling window."""
        n = len(returns)
        if n < span:
            return np.array([np.std(returns)])

        alpha = 2.0 / (span + 1)
        vols: list[float] = []
        effective_window = min(window, n)

        for end in range(span, n + 1):
            start = max(0, end - effective_window)
            segment = returns[start:end]
            weights = (1 - alpha) ** np.arange(len(segment) - 1, -1, -1)
            ewm_mean = np.average(segment, weights=weights)
            ewm_var = np.average((segment - ewm_mean) ** 2, weights=weights)
            vols.append(float(np.sqrt(ewm_var)))

        return np.array(vols)
