"""Tests for EdgeCalculator."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from moneygone.exchange.types import OrderbookLevel, OrderbookSnapshot
from moneygone.signals.edge import EdgeCalculator, EdgeResult
from moneygone.signals.fees import KalshiFeeCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 14, 30, tzinfo=timezone.utc)


def _make_orderbook(
    *,
    yes_prices: list[str] | None = None,
    yes_sizes: list[str] | None = None,
    no_prices: list[str] | None = None,
    no_sizes: list[str] | None = None,
    ticker: str = "TEST-TICKER",
) -> OrderbookSnapshot:
    """Build an OrderbookSnapshot with the given levels."""
    yes_prices = yes_prices or ["0.60"]
    yes_sizes = yes_sizes or ["200"]
    no_prices = no_prices or ["0.40"]
    no_sizes = no_sizes or ["200"]

    yes_levels = tuple(
        OrderbookLevel(price=Decimal(p), contracts=Decimal(s))
        for p, s in zip(yes_prices, yes_sizes)
    )
    no_levels = tuple(
        OrderbookLevel(price=Decimal(p), contracts=Decimal(s))
        for p, s in zip(no_prices, no_sizes)
    )
    return OrderbookSnapshot(
        ticker=ticker,
        yes_bids=yes_levels,
        no_bids=no_levels,
        seq=1,
        timestamp=_NOW,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEdgeCalculator:
    """Test the EdgeCalculator edge computation logic."""

    def test_positive_edge_yes_buy(self) -> None:
        """When model says YES is more likely than market, edge should be positive on YES side."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(fees, min_edge_threshold=0.01, min_liquidity=1)

        ob = _make_orderbook(
            yes_prices=["0.55"],
            yes_sizes=["100"],
            no_prices=["0.45"],
            no_sizes=["100"],
        )

        # Model thinks prob is 0.70, market has yes ask at 0.55
        result = calc.compute_edge(0.70, ob, is_maker=True)

        # YES edge: 0.70 - 0.55 = 0.15 (maker fee = 0)
        assert result.raw_edge > 0
        assert result.fee_adjusted_edge > 0
        assert result.is_actionable
        assert result.side == "yes"

    def test_positive_edge_no_buy(self) -> None:
        """When model says NO is more likely, edge should favour the NO side."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(fees, min_edge_threshold=0.01, max_edge_sanity=0.50, min_liquidity=1)

        ob = _make_orderbook(
            yes_prices=["0.65"],
            yes_sizes=["100"],
            no_prices=["0.45"],
            no_sizes=["100"],
        )

        # Model thinks prob of YES is only 0.40 (so NO = 0.60)
        # NO ask at 0.45, NO edge = 0.60 - 0.45 = 0.15
        # YES edge = 0.40 - 0.65 = -0.25 (negative)
        result = calc.compute_edge(0.40, ob, is_maker=True)

        assert result.fee_adjusted_edge > 0
        assert result.side == "no"
        assert result.is_actionable

    def test_no_edge_when_model_agrees_with_market(self) -> None:
        """When model agrees with market price, there should be no actionable edge."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(fees, min_edge_threshold=0.02, min_liquidity=1)

        ob = _make_orderbook(
            yes_prices=["0.60"],
            yes_sizes=["100"],
            no_prices=["0.40"],
            no_sizes=["100"],
        )

        # Model agrees with the yes ask exactly
        result = calc.compute_edge(0.60, ob, is_maker=True)

        # Edge should be near zero: yes_edge = 0.60-0.60 = 0, no_edge = 0.40-0.40 = 0
        assert not result.is_actionable
        assert abs(result.fee_adjusted_edge) < 0.01

    def test_sanity_check_rejects_large_edge(self) -> None:
        """Edge exceeding max_edge_sanity should not be actionable."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(
            fees, min_edge_threshold=0.01, max_edge_sanity=0.20, min_liquidity=1
        )

        ob = _make_orderbook(
            yes_prices=["0.20"],
            yes_sizes=["100"],
            no_prices=["0.80"],
            no_sizes=["100"],
        )

        # Model prob 0.90 vs market ask 0.20 -> edge = 0.70, far exceeding 0.20 sanity cap
        result = calc.compute_edge(0.90, ob, is_maker=True)

        assert abs(result.raw_edge) > 0.20
        assert not result.is_actionable

    def test_fee_adjusted_edge_lower_than_raw(self) -> None:
        """Fee-adjusted edge should be <= raw edge (taker fees reduce edge)."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(fees, min_edge_threshold=0.01, min_liquidity=1)

        ob = _make_orderbook(
            yes_prices=["0.55"],
            yes_sizes=["100"],
            no_prices=["0.45"],
            no_sizes=["100"],
        )

        # As taker, fees should reduce the edge
        result = calc.compute_edge(0.70, ob, is_maker=False)

        assert result.fee_adjusted_edge < result.raw_edge

    def test_liquidity_check(self) -> None:
        """Insufficient liquidity makes the edge non-actionable."""
        fees = KalshiFeeCalculator()
        calc = EdgeCalculator(
            fees, min_edge_threshold=0.01, min_liquidity=50
        )

        # Only 5 contracts available, but min_liquidity=50
        ob = _make_orderbook(
            yes_prices=["0.55"],
            yes_sizes=["5"],
            no_prices=["0.45"],
            no_sizes=["5"],
        )

        result = calc.compute_edge(0.70, ob, is_maker=True)

        # Edge is positive but liquidity is too thin
        assert result.raw_edge > 0
        assert not result.is_actionable
