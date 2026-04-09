"""Tests for the fill simulator's bid-only orderbook handling."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.exchange.types import (
    Action,
    OrderRequest,
    OrderbookLevel,
    OrderbookSnapshot,
    Side,
    TimeInForce,
)
from moneygone.execution.simulator import FillSimulator


_NOW = datetime(2026, 4, 9, 14, 30, tzinfo=timezone.utc)


def _make_orderbook(
    *,
    yes_prices: list[str],
    yes_sizes: list[str],
    no_prices: list[str],
    no_sizes: list[str],
) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ticker="TEST",
        yes_bids=tuple(
            OrderbookLevel(price=Decimal(price), contracts=Decimal(size))
            for price, size in zip(yes_prices, yes_sizes)
        ),
        no_bids=tuple(
            OrderbookLevel(price=Decimal(price), contracts=Decimal(size))
            for price, size in zip(no_prices, no_sizes)
        ),
        seq=1,
        timestamp=_NOW,
    )


class TestFillSimulator:
    """Aggressive fills should walk the opposite bid ladder, not the same side."""

    def test_yes_buy_walks_opposite_no_bids(self) -> None:
        simulator = FillSimulator(model="realistic", slippage_bps=0.0)
        orderbook = _make_orderbook(
            yes_prices=["0.50", "0.52", "0.54"],
            yes_sizes=["10", "20", "30"],
            no_prices=["0.38", "0.42", "0.44"],
            no_sizes=["20", "30", "40"],
        )
        order = OrderRequest(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            count=70,
            yes_price=Decimal("0.58"),
            time_in_force=TimeInForce.IOC,
            post_only=False,
        )

        result = simulator.simulate_fill(order, orderbook)

        assert result.filled
        assert result.filled_contracts == 70
        assert result.fill_price == Decimal("0.57")

    def test_no_buy_walks_opposite_yes_bids(self) -> None:
        simulator = FillSimulator(model="realistic", slippage_bps=0.0)
        orderbook = _make_orderbook(
            yes_prices=["0.52", "0.54", "0.56"],
            yes_sizes=["20", "30", "40"],
            no_prices=["0.34", "0.36", "0.38"],
            no_sizes=["10", "20", "30"],
        )
        order = OrderRequest(
            ticker="TEST",
            side=Side.NO,
            action=Action.BUY,
            count=70,
            yes_price=Decimal("0.53"),
            time_in_force=TimeInForce.IOC,
            post_only=False,
        )

        result = simulator.simulate_fill(order, orderbook)

        assert result.filled
        assert result.filled_contracts == 70
        assert result.fill_price == Decimal("0.45")
