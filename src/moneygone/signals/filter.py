"""Pre-trade signal quality filter.

Validates that a computed edge passes a series of quality checks before
the signal is forwarded to the execution layer.  Each filter produces a
clear pass/fail with a human-readable reason on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from moneygone.config import ExecutionConfig, RiskConfig
from moneygone.exchange.types import Market
from moneygone.signals.edge import EdgeResult

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelPrediction:
    """Lightweight container for model output passed through the filter."""

    probability: float
    confidence: float
    timestamp: datetime


@dataclass(frozen=True)
class FilterResult:
    """Outcome of the signal filter pipeline."""

    passed: bool
    """True if the signal passed all filters."""

    reason: str | None
    """Human-readable rejection reason, or None if passed."""

    filter_name: str | None
    """Name of the filter that rejected, or None if passed."""


class SignalFilter:
    """Validates signal quality before execution.

    Parameters
    ----------
    risk_config:
        Risk limits (contract price bounds, tail exposure caps).
    execution_config:
        Execution parameters (edge thresholds, staleness).
    min_liquidity:
        Minimum available contracts at the target price.
    min_spread_ratio:
        Minimum ratio of edge-to-spread; reject if spread too wide
        relative to the edge captured.
    min_time_to_expiry_seconds:
        Reject if market closes sooner than this many seconds.
    min_confidence:
        Reject if model confidence is below this threshold.
    max_data_age_seconds:
        Reject if market data is older than this many seconds.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        execution_config: ExecutionConfig,
        *,
        min_liquidity: int = 5,
        min_spread_ratio: float = 0.5,
        min_time_to_expiry_seconds: int = 3600,
        min_confidence: float = 0.55,
        max_data_age_seconds: int = 60,
    ) -> None:
        self._risk = risk_config
        self._exec = execution_config
        self._min_liquidity = min_liquidity
        self._min_spread_ratio = min_spread_ratio
        self._min_time_to_expiry = min_time_to_expiry_seconds
        self._min_confidence = min_confidence
        self._max_data_age = max_data_age_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        edge_result: EdgeResult,
        market: Market,
        model_prediction: ModelPrediction,
    ) -> FilterResult:
        """Run all filters in sequence and return the first failure, or pass.

        Filters are ordered from cheapest to most expensive to evaluate.
        """
        filters = [
            self._liquidity_filter,
            self._tail_filter,
            self._confidence_filter,
            self._staleness_filter,
            self._time_to_expiry_filter,
            self._spread_ratio_filter,
        ]

        for check in filters:
            result = check(edge_result, market, model_prediction)
            if not result.passed:
                logger.info(
                    "signal_filtered",
                    ticker=market.ticker,
                    filter=result.filter_name,
                    reason=result.reason,
                )
                return result

        return FilterResult(passed=True, reason=None, filter_name=None)

    # ------------------------------------------------------------------
    # Individual filters
    # ------------------------------------------------------------------

    def _liquidity_filter(
        self,
        edge: EdgeResult,
        market: Market,  # noqa: ARG002
        prediction: ModelPrediction,  # noqa: ARG002
    ) -> FilterResult:
        """Reject if available contracts at target price are insufficient."""
        if edge.available_liquidity < self._min_liquidity:
            return FilterResult(
                passed=False,
                reason=(
                    f"Insufficient liquidity: {edge.available_liquidity} contracts "
                    f"available, minimum {self._min_liquidity} required"
                ),
                filter_name="min_liquidity",
            )
        return _PASSED

    def _tail_filter(
        self,
        edge: EdgeResult,
        market: Market,  # noqa: ARG002
        prediction: ModelPrediction,  # noqa: ARG002
    ) -> FilterResult:
        """Reject contracts priced below min or above max (tail risk)."""
        price_f = float(edge.target_price)
        if price_f < self._risk.min_contract_price:
            return FilterResult(
                passed=False,
                reason=(
                    f"Price {price_f:.2f} below minimum {self._risk.min_contract_price:.2f} "
                    f"(tail contract -- excessive risk)"
                ),
                filter_name="tail_filter",
            )
        if price_f > self._risk.max_contract_price:
            return FilterResult(
                passed=False,
                reason=(
                    f"Price {price_f:.2f} above maximum {self._risk.max_contract_price:.2f} "
                    f"(tail contract -- excessive risk)"
                ),
                filter_name="tail_filter",
            )
        return _PASSED

    def _confidence_filter(
        self,
        edge: EdgeResult,  # noqa: ARG002
        market: Market,  # noqa: ARG002
        prediction: ModelPrediction,
    ) -> FilterResult:
        """Reject if model confidence is below threshold."""
        if prediction.confidence < self._min_confidence:
            return FilterResult(
                passed=False,
                reason=(
                    f"Model confidence {prediction.confidence:.3f} below "
                    f"minimum {self._min_confidence:.3f}"
                ),
                filter_name="confidence_filter",
            )
        return _PASSED

    def _staleness_filter(
        self,
        edge: EdgeResult,  # noqa: ARG002
        market: Market,  # noqa: ARG002
        prediction: ModelPrediction,
    ) -> FilterResult:
        """Reject if market data / model prediction is stale."""
        now = datetime.now(timezone.utc)
        age = (now - prediction.timestamp).total_seconds()
        if age > self._max_data_age:
            return FilterResult(
                passed=False,
                reason=(
                    f"Data age {age:.0f}s exceeds maximum {self._max_data_age}s"
                ),
                filter_name="staleness_filter",
            )
        return _PASSED

    def _time_to_expiry_filter(
        self,
        edge: EdgeResult,  # noqa: ARG002
        market: Market,
        prediction: ModelPrediction,  # noqa: ARG002
    ) -> FilterResult:
        """Reject if market closes too soon."""
        now = datetime.now(timezone.utc)
        # market.close_time should be tz-aware; handle naive gracefully
        close_time = market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        remaining = (close_time - now).total_seconds()
        if remaining < self._min_time_to_expiry:
            return FilterResult(
                passed=False,
                reason=(
                    f"Time to expiry {remaining:.0f}s below minimum "
                    f"{self._min_time_to_expiry}s"
                ),
                filter_name="min_time_to_expiry",
            )
        return _PASSED

    def _spread_ratio_filter(
        self,
        edge: EdgeResult,
        market: Market,
        prediction: ModelPrediction,  # noqa: ARG002
    ) -> FilterResult:
        """Reject if spread is too wide relative to edge.

        Spread ratio = fee_adjusted_edge / spread.  If the spread is very
        wide compared to the edge we capture, slippage risk dominates.
        """
        spread = float(market.yes_ask - market.yes_bid)
        if spread <= 0:
            # Zero or negative spread (crossed book): pass through
            return _PASSED
        ratio = edge.fee_adjusted_edge / spread if spread > 0 else float("inf")
        if ratio < self._min_spread_ratio:
            return FilterResult(
                passed=False,
                reason=(
                    f"Spread ratio {ratio:.3f} below minimum {self._min_spread_ratio:.3f} "
                    f"(spread={spread:.4f}, edge={edge.fee_adjusted_edge:.4f})"
                ),
                filter_name="min_spread_ratio",
            )
        return _PASSED


# Sentinel for passing filters (avoid re-allocating on every call)
_PASSED = FilterResult(passed=True, reason=None, filter_name=None)
