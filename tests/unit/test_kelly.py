"""Tests for KellySizer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from moneygone.signals.edge import EdgeResult
from moneygone.sizing.kelly import KellySizer, SizeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge(
    *,
    model_prob: float = 0.70,
    implied_prob: float = 0.60,
    target_price: str = "0.60",
    fee_adjusted_edge: float = 0.08,
    raw_edge: float = 0.10,
    side: str = "yes",
    available_liquidity: int = 500,
    is_actionable: bool = True,
) -> EdgeResult:
    """Build an EdgeResult with sensible defaults."""
    return EdgeResult(
        raw_edge=raw_edge,
        fee_adjusted_edge=fee_adjusted_edge,
        implied_probability=implied_prob,
        model_probability=model_prob,
        available_liquidity=available_liquidity,
        estimated_fill_rate=1.0,
        is_actionable=is_actionable,
        side=side,
        action="buy",
        target_price=Decimal(target_price),
        expected_value=Decimal(str(fee_adjusted_edge)),
    )


class TestKellySizer:
    """Test the KellySizer position sizing logic."""

    def test_positive_edge_sizing(self) -> None:
        """With a positive edge, Kelly should produce positive contracts."""
        sizer = KellySizer(kelly_fraction=0.25, max_position_pct=0.20)
        edge = _make_edge(model_prob=0.70, implied_prob=0.60, target_price="0.60")

        result = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        assert result.contracts > 0
        assert result.kelly_fraction > 0
        assert result.dollar_risk > Decimal("0")
        assert result.dollar_ev > Decimal("0")

    def test_zero_edge_no_trade(self) -> None:
        """When model_prob equals the market price, Kelly fraction should be ~0."""
        sizer = KellySizer(kelly_fraction=0.25)
        # model_prob == implied_prob => edge is 0, not actionable
        edge = _make_edge(
            model_prob=0.60,
            implied_prob=0.60,
            target_price="0.60",
            raw_edge=0.0,
            fee_adjusted_edge=-0.02,
            is_actionable=False,
        )

        result = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        assert result.contracts == 0

    def test_negative_edge_no_trade(self) -> None:
        """When model_prob < market price, Kelly should produce 0 contracts."""
        sizer = KellySizer(kelly_fraction=0.25)
        edge = _make_edge(
            model_prob=0.55,
            implied_prob=0.60,
            target_price="0.60",
            raw_edge=-0.05,
            fee_adjusted_edge=-0.07,
            is_actionable=False,
        )

        result = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        assert result.contracts == 0

    def test_fractional_kelly_reduces_size(self) -> None:
        """Smaller Kelly fraction should produce fewer contracts."""
        edge = _make_edge(model_prob=0.70, target_price="0.60", available_liquidity=100000)

        sizer_full = KellySizer(kelly_fraction=1.0, max_position_pct=1.0)
        sizer_quarter = KellySizer(kelly_fraction=0.25, max_position_pct=1.0)

        result_full = sizer_full.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )
        result_quarter = sizer_quarter.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        assert result_full.contracts > result_quarter.contracts
        assert result_full.adjusted_fraction > result_quarter.adjusted_fraction

    def test_confidence_scaling(self) -> None:
        """Lower model confidence should reduce position size."""
        sizer = KellySizer(kelly_fraction=0.25, max_position_pct=1.0)
        edge = _make_edge(model_prob=0.70, target_price="0.60")

        high_conf = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )
        low_conf = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=0.5,
            existing_exposure=Decimal("0"),
        )

        assert high_conf.contracts >= low_conf.contracts
        assert high_conf.adjusted_fraction > low_conf.adjusted_fraction

    def test_max_position_cap(self) -> None:
        """Position should be capped at max_position_pct of bankroll."""
        sizer = KellySizer(
            kelly_fraction=1.0,
            max_position_pct=0.05,  # Very tight cap
        )
        edge = _make_edge(model_prob=0.90, target_price="0.60")

        result = sizer.size(
            edge_result=edge,
            bankroll=Decimal("10000"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        # Dollar risk should not exceed 5% of bankroll
        max_dollar = Decimal("10000") * Decimal("0.05")
        assert result.dollar_risk <= max_dollar + Decimal("1")  # allow rounding
        assert result.capped_by == "max_position_pct"

    def test_zero_bankroll(self) -> None:
        """With zero bankroll, no position should be taken."""
        sizer = KellySizer(kelly_fraction=0.25)
        edge = _make_edge(model_prob=0.70, target_price="0.60")

        result = sizer.size(
            edge_result=edge,
            bankroll=Decimal("0"),
            model_confidence=1.0,
            existing_exposure=Decimal("0"),
        )

        assert result.contracts == 0
        assert result.capped_by == "zero_bankroll"
