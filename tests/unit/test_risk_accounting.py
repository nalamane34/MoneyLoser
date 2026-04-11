"""Tests for risk manager and portfolio accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.config import RiskConfig
from moneygone.exchange.types import Action, Fill, MarketResult, Settlement, Side
from moneygone.risk.drawdown import DrawdownMonitor
from moneygone.risk.exposure import ExposureCalculator
from moneygone.risk.manager import RiskManager
from moneygone.risk.portfolio import PortfolioTracker
from moneygone.sizing.risk_limits import ProposedTrade, RiskLimits


def _fill(
    *,
    trade_id: str,
    ticker: str = "TEST-TICKER",
    side: Side,
    action: Action,
    count: int,
    price: str,
    fee_cost: str = "0",
) -> Fill:
    return Fill(
        fill_id=trade_id,
        ticker=ticker,
        side=side,
        action=action,
        count=count,
        price=Decimal(price),
        no_price=Decimal("1") - Decimal(price),
        fee_cost=Decimal(fee_cost),
        is_taker=True,
        created_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
    )


def _manager(
    risk_config: RiskConfig,
    *,
    initial_cash: str = "10",
) -> RiskManager:
    portfolio = PortfolioTracker(initial_cash=Decimal(initial_cash))
    return RiskManager(
        risk_config=risk_config,
        risk_limits=RiskLimits(risk_config),
        portfolio=portfolio,
        drawdown_monitor=DrawdownMonitor(),
        exposure_calculator=ExposureCalculator(),
        categories={},
    )


def test_yes_and_no_legs_keep_separate_cost_basis() -> None:
    portfolio = PortfolioTracker(initial_cash=Decimal("100"))

    portfolio.on_fill(
        _fill(
            trade_id="buy-yes",
            side=Side.YES,
            action=Action.BUY,
            count=10,
            price="0.40",
        )
    )
    portfolio.on_fill(
        _fill(
            trade_id="buy-no",
            side=Side.NO,
            action=Action.BUY,
            count=10,
            price="0.20",
        )
    )
    portfolio.on_fill(
        _fill(
            trade_id="sell-yes",
            side=Side.YES,
            action=Action.SELL,
            count=5,
            price="0.70",
        )
    )

    pos = portfolio.get_position("TEST-TICKER")
    assert pos is not None
    assert pos.yes_count == 5
    assert pos.no_count == 10
    assert pos.yes_cost_basis == Decimal("2.00")
    assert pos.no_cost_basis == Decimal("2.00")
    assert pos.cost_basis == Decimal("4.00")
    assert pos.realized_pnl == Decimal("1.50")
    assert portfolio.realized_pnl == Decimal("1.50")


def test_fees_reduce_cash_and_realized_pnl() -> None:
    portfolio = PortfolioTracker(initial_cash=Decimal("100"))

    portfolio.on_fill(
        _fill(
            trade_id="buy-yes",
            side=Side.YES,
            action=Action.BUY,
            count=10,
            price="0.40",
            fee_cost="0.10",
        )
    )
    portfolio.on_fill(
        _fill(
            trade_id="sell-yes",
            side=Side.YES,
            action=Action.SELL,
            count=10,
            price="0.70",
            fee_cost="0.05",
        )
    )

    assert portfolio.cash == Decimal("102.85")
    assert portfolio.realized_pnl == Decimal("2.85")


def test_settlement_adds_payout_and_realizes_remaining_pnl() -> None:
    portfolio = PortfolioTracker(initial_cash=Decimal("10"))
    portfolio.on_fill(
        _fill(
            trade_id="buy-yes",
            side=Side.YES,
            action=Action.BUY,
            count=1,
            price="0.40",
        )
    )

    portfolio.on_settlement(
        Settlement(
            ticker="TEST-TICKER",
            market_result=MarketResult.YES,
            revenue=Decimal("1.00"),
            settled_time=datetime(2026, 4, 10, tzinfo=timezone.utc),
        )
    )

    assert portfolio.cash == Decimal("10.60")
    assert portfolio.realized_pnl == Decimal("0.60")


def test_daily_pnl_uses_realized_delta_for_loss_checks(
    risk_config: RiskConfig,
) -> None:
    manager = _manager(risk_config, initial_cash="10")

    manager.post_trade_update(
        _fill(
            trade_id="buy-1",
            side=Side.YES,
            action=Action.BUY,
            count=1,
            price="0.90",
        )
    )
    manager.post_trade_update(
        _fill(
            trade_id="sell-1",
            side=Side.YES,
            action=Action.SELL,
            count=1,
            price="0.10",
        )
    )

    summary = manager.get_risk_summary()
    assert summary.daily_pnl == Decimal("-0.80")

    result = manager.pre_trade_check(
        ProposedTrade(
            ticker="TEST-TICKER",
            category="weather",
            side="yes",
            action="buy",
            contracts=1,
            price=Decimal("0.50"),
        )
    )

    assert not result.approved
    assert result.limit_triggered == "daily_loss_limit"
