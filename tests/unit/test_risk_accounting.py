"""Tests for risk manager and portfolio accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.config import RiskConfig
from moneygone.monitoring.pnl import PnLTracker
from moneygone.models.base import ModelPrediction
from moneygone.exchange.types import Action, Fill, MarketResult, Settlement, Side
from moneygone.risk.drawdown import DrawdownMonitor
from moneygone.risk.exposure import ExposureCalculator
from moneygone.risk.manager import RiskManager
from moneygone.risk.portfolio import PortfolioTracker
from moneygone.signals.edge import EdgeResult
from moneygone.sizing.risk_limits import ProposedTrade, RiskLimits


def _fill(
    *,
    trade_id: str,
    ticker: str = "TEST-TICKER",
    side: Side,
    action: Action,
    count: int,
    price: str,
    no_price: str | None = None,
    fee_cost: str = "0",
) -> Fill:
    price_decimal = Decimal(price)
    return Fill(
        fill_id=trade_id,
        ticker=ticker,
        side=side,
        action=action,
        count=count,
        price=price_decimal,
        no_price=(
            Decimal(no_price)
            if no_price is not None
            else Decimal("1") - price_decimal
        ),
        fee_cost=Decimal(fee_cost),
        is_taker=True,
        created_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
    )


def _prediction() -> ModelPrediction:
    return ModelPrediction(
        probability=0.62,
        raw_probability=0.62,
        confidence=0.55,
        model_name="test-model",
        model_version="v1",
        features_used={},
        prediction_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
    )


def _edge() -> EdgeResult:
    return EdgeResult(
        raw_edge=0.05,
        fee_adjusted_edge=0.04,
        implied_probability=0.58,
        model_probability=0.62,
        available_liquidity=10,
        estimated_fill_rate=1.0,
        is_actionable=True,
        side="yes",
        action="buy",
        target_price=Decimal("0.58"),
        expected_value=Decimal("0.04"),
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
            price="0.80",
            no_price="0.20",
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


def test_manager_counts_gross_inventory_for_position_limit(
    risk_config: RiskConfig,
) -> None:
    manager = _manager(risk_config, initial_cash="100")

    manager.post_trade_update(
        _fill(
            trade_id="buy-yes-1",
            side=Side.YES,
            action=Action.BUY,
            count=30,
            price="0.40",
        )
    )
    manager.post_trade_update(
        _fill(
            trade_id="buy-no-1",
            side=Side.NO,
            action=Action.BUY,
            count=30,
            price="0.20",
        )
    )

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
    assert result.limit_triggered == "max_position_per_market"


def test_manager_reports_existing_tail_exposure_from_state(
    risk_config: RiskConfig,
) -> None:
    manager = _manager(risk_config, initial_cash="100")

    manager.post_trade_update(
        _fill(
            trade_id="buy-safe",
            side=Side.YES,
            action=Action.BUY,
            count=1,
            price="0.40",
        )
    )
    manager.post_trade_update(
        _fill(
            trade_id="buy-tail",
            ticker="TAIL-TICKER",
            side=Side.YES,
            action=Action.BUY,
            count=1,
            price="0.90",
        )
    )

    summary = manager.get_risk_summary()

    assert summary.tail_exposure == Decimal("0.90")
    assert summary.tail_exposure_pct == 0.009


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


def test_no_side_uses_no_price_for_cash_and_realized_pnl() -> None:
    portfolio = PortfolioTracker(initial_cash=Decimal("100"))

    portfolio.on_fill(
        Fill(
            fill_id="buy-no",
            ticker="TEST-TICKER",
            side=Side.NO,
            action=Action.BUY,
            count=10,
            price=Decimal("0.80"),
            no_price=Decimal("0.20"),
            fee_cost=Decimal("0.10"),
            is_taker=True,
            created_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
    )
    portfolio.on_fill(
        Fill(
            fill_id="sell-no",
            ticker="TEST-TICKER",
            side=Side.NO,
            action=Action.SELL,
            count=5,
            price=Decimal("0.75"),
            no_price=Decimal("0.25"),
            fee_cost=Decimal("0.05"),
            is_taker=True,
            created_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
    )

    pos = portfolio.get_position("TEST-TICKER")
    assert pos is not None
    assert pos.no_count == 5
    assert pos.no_cost_basis == Decimal("1.05")
    assert pos.realized_pnl == Decimal("0.15")
    assert portfolio.cash == Decimal("99.10")


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


def test_pnl_tracker_keeps_settlement_revenue_in_dollars() -> None:
    tracker = PnLTracker()
    tracker.record_trade(
        Fill(
            fill_id="no-fill",
            ticker="TEST-TICKER",
            side=Side.NO,
            action=Action.BUY,
            count=1,
            price=Decimal("0.80"),
            no_price=Decimal("0.20"),
            fee_cost=Decimal("0.01"),
            is_taker=True,
            created_time=datetime(2026, 4, 9, tzinfo=timezone.utc),
        ),
        _prediction(),
        _edge(),
        category="sports",
    )
    tracker.record_settlement(
        Settlement(
            ticker="TEST-TICKER",
            market_result=MarketResult.NO,
            revenue=Decimal("1.00"),
            settled_time=datetime(2026, 4, 10, tzinfo=timezone.utc),
        )
    )

    summary = tracker.get_summary()
    assert summary.gross_pnl == 0.8
    assert summary.fees_paid == 0.01
    assert summary.net_pnl == 0.79


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
