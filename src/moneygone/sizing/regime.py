"""Market regime detection for dynamic position sizing.

Uses realized volatility percentiles over a configurable lookback window
to classify the current environment into one of four regimes, each with
a sizing adjustment factor:

  - **crisis** (top 5% vol):   factor = 0.0  (halt trading)
  - **high_vol** (top 20%):    factor = 0.5  (halve positions)
  - **normal** (middle 60%):   factor = 1.0  (trade normally)
  - **low_vol** (bottom 20%):  factor = 0.8  (slightly reduce -- potential
    regime shift ahead)

The adjustment factor multiplies the Kelly fraction in the sizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class Regime(str, Enum):
    """Market volatility regime."""

    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"


# Adjustment factors per regime
_REGIME_FACTORS: dict[Regime, float] = {
    Regime.CRISIS: 0.0,
    Regime.HIGH_VOL: 0.5,
    Regime.NORMAL: 1.0,
    Regime.LOW_VOL: 0.8,
}


@dataclass(frozen=True)
class RegimeState:
    """Current regime classification and sizing adjustment."""

    regime: Regime
    """Detected market regime."""

    adjustment_factor: float
    """Multiplier for position sizing, 0.0 to 1.0."""

    realized_vol: float
    """Current realized volatility value."""

    vol_percentile: float
    """Percentile rank of current vol within the lookback window (0-100)."""


class RegimeDetector:
    """Classifies market regime from recent price and volume data.

    Parameters
    ----------
    vol_lookback:
        Number of observations for the lookback volatility calculation.
    crisis_percentile:
        Volatility percentile threshold for crisis regime (default 95).
    high_vol_percentile:
        Volatility percentile threshold for high-vol regime (default 80).
    low_vol_percentile:
        Volatility percentile threshold for low-vol regime (default 20).
    min_observations:
        Minimum data points required; returns NORMAL if insufficient.
    """

    def __init__(
        self,
        vol_lookback: int = 100,
        crisis_percentile: float = 95.0,
        high_vol_percentile: float = 80.0,
        low_vol_percentile: float = 20.0,
        min_observations: int = 20,
    ) -> None:
        self._lookback = vol_lookback
        self._crisis_pct = crisis_percentile
        self._high_vol_pct = high_vol_percentile
        self._low_vol_pct = low_vol_percentile
        self._min_obs = min_observations

    def detect_regime(
        self,
        recent_prices: list[float],
        recent_volumes: list[int],
    ) -> RegimeState:
        """Classify the current market regime.

        Parameters
        ----------
        recent_prices:
            Recent mid-prices or last-trade prices, ordered oldest-first.
        recent_volumes:
            Corresponding trading volumes (used for volume-weighted adjustments).

        Returns
        -------
        RegimeState
            The detected regime with its sizing adjustment factor.
        """
        if len(recent_prices) < self._min_obs:
            logger.debug(
                "regime_insufficient_data",
                n_prices=len(recent_prices),
                min_required=self._min_obs,
            )
            return RegimeState(
                regime=Regime.NORMAL,
                adjustment_factor=1.0,
                realized_vol=0.0,
                vol_percentile=50.0,
            )

        prices = np.array(recent_prices, dtype=np.float64)

        # Compute log returns; avoid log(0) by filtering non-positive prices
        valid_mask = prices > 0
        if valid_mask.sum() < self._min_obs:
            return RegimeState(
                regime=Regime.NORMAL,
                adjustment_factor=1.0,
                realized_vol=0.0,
                vol_percentile=50.0,
            )

        valid_prices = prices[valid_mask]
        log_returns = np.diff(np.log(valid_prices))

        if len(log_returns) < 2:
            return RegimeState(
                regime=Regime.NORMAL,
                adjustment_factor=1.0,
                realized_vol=0.0,
                vol_percentile=50.0,
            )

        # Realized volatility: std of log returns over the lookback window
        returns_window = log_returns[-self._lookback :]
        current_vol = float(np.std(returns_window, ddof=1))

        # Build a distribution of rolling volatilities for percentile ranking
        window_size = min(20, len(log_returns) // 2)
        if window_size < 2:
            window_size = 2

        rolling_vols: list[float] = []
        for i in range(window_size, len(log_returns) + 1):
            segment = log_returns[max(0, i - window_size) : i]
            if len(segment) >= 2:
                rolling_vols.append(float(np.std(segment, ddof=1)))

        if not rolling_vols:
            return RegimeState(
                regime=Regime.NORMAL,
                adjustment_factor=1.0,
                realized_vol=current_vol,
                vol_percentile=50.0,
            )

        # Volume-weighted adjustment: elevated volume during high vol amplifies the signal
        vol_adjustment = 1.0
        if recent_volumes and len(recent_volumes) >= self._min_obs:
            volumes = np.array(
                recent_volumes[-len(returns_window) :], dtype=np.float64
            )
            if len(volumes) > 0:
                mean_vol = float(np.mean(volumes))
                if mean_vol > 0:
                    recent_vol_ratio = float(np.mean(volumes[-5:]) / mean_vol)
                    vol_adjustment = max(1.0, min(1.5, recent_vol_ratio))

        adjusted_vol = current_vol * vol_adjustment

        # Percentile rank of current (adjusted) vol within rolling distribution
        vol_array = np.array(rolling_vols)
        vol_percentile = float(
            np.sum(vol_array <= adjusted_vol) / len(vol_array) * 100
        )

        # Classify regime
        regime = self._classify(vol_percentile)
        factor = _REGIME_FACTORS[regime]

        logger.info(
            "regime_detected",
            regime=regime.value,
            factor=factor,
            realized_vol=round(current_vol, 6),
            vol_percentile=round(vol_percentile, 1),
            vol_adjustment=round(vol_adjustment, 3),
        )

        return RegimeState(
            regime=regime,
            adjustment_factor=factor,
            realized_vol=round(adjusted_vol, 6),
            vol_percentile=round(vol_percentile, 1),
        )

    def _classify(self, percentile: float) -> Regime:
        """Map a volatility percentile to a regime."""
        if percentile >= self._crisis_pct:
            return Regime.CRISIS
        if percentile >= self._high_vol_pct:
            return Regime.HIGH_VOL
        if percentile <= self._low_vol_pct:
            return Regime.LOW_VOL
        return Regime.NORMAL
