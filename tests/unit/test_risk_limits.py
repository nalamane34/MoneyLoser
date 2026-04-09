"""Tests for RiskLimits pre-trade checks."""

from __future__ import annotations

from decimal import Decimal

import pytest

from moneygone.config import RiskConfig
from moneygone.sizing.risk_limits import (
    PortfolioState,
    ProposedTrade,
    RiskCheckResult,
    RiskLimits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(
    *,
    bankroll: str = "10000",
    daily_pnl: str = "0",
    peak_equity: str = "10000",
    current_equity: str = "10000",
    positions: dict[str, int] | None = None,
    category_exposure: dict[str, str] | None = None,
    total_exposure: str = "0",
) -> PortfolioState:
    return PortfolioState(
        positions=positions or {},
        position_costs={},
        category_exposure={
            k: Decimal(v) for k, v in (category_exposure or {}).items()
        },
        total_exposure=Decimal(total_exposure),
        bankroll=Decimal(bankroll),
        daily_pnl=Decimal(daily_pnl),
        peak_equity=Decimal(peak_equity),
        current_equity=Decimal(current_equity),
    )


def _make_trade(
    *,
    ticker: str = "TEST-TICKER",
    category: str = "weather",
    contracts: int = 10,
    price: str = "0.50",
) -> ProposedTrade:
    return ProposedTrade(
        ticker=ticker,
        category=category,
        side="yes",
        action="buy",
        contracts=contracts,
        price=Decimal(price),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRiskLimits:
    """Test the RiskLimits pre-trade engine."""

    def test_approved_within_limits(self, risk_config: RiskConfig) -> None:
        """A small trade within all limits should be approved."""
        limits = RiskLimits(risk_config)
        trade = _make_trade(contracts=5, price="0.50")
        portfolio = _make_portfolio()

        result = limits.check(trade, portfolio)

        assert result.approved
        assert result.rejection_reason is None

    def test_rejected_exceeds_market_limit(self, risk_config: RiskConfig) -> None:
        """Trade exceeding max_position_per_market should be partially or fully rejected."""
        limits = RiskLimits(risk_config)
        # risk_config has max_position_per_market=50
        trade = _make_trade(contracts=60, price="0.50")
        portfolio = _make_portfolio()

        result = limits.check(trade, portfolio)

        # Should be partially approved (adjusted to 50) since we have 0 existing
        assert result.approved
        assert result.adjusted_size is not None
        assert result.adjusted_size <= risk_config.max_position_per_market
        assert result.limit_triggered == "max_position_per_market"

    def test_rejected_daily_loss_exceeded(self, risk_config: RiskConfig) -> None:
        """Trading should be halted when daily loss limit is breached."""
        limits = RiskLimits(risk_config)
        trade = _make_trade(contracts=5, price="0.50")
        # daily_loss_limit_pct = 0.05, so loss of 500 on 10000 bankroll = 5%
        portfolio = _make_portfolio(daily_pnl="-500")

        result = limits.check(trade, portfolio)

        assert not result.approved
        assert result.limit_triggered == "daily_loss_limit"

    def test_rejected_tail_contract(self, risk_config: RiskConfig) -> None:
        """Contracts priced below min_contract_price should be rejected."""
        limits = RiskLimits(risk_config)
        # risk_config has min_contract_price=0.05
        trade = _make_trade(contracts=5, price="0.03")
        portfolio = _make_portfolio()

        result = limits.check(trade, portfolio)

        assert not result.approved
        assert result.limit_triggered == "min_contract_price"

    def test_partial_approval(self, risk_config: RiskConfig) -> None:
        """When position limit would be exceeded, size should be reduced."""
        limits = RiskLimits(risk_config)
        # Already holding 40, requesting 20 more, max is 50
        trade = _make_trade(ticker="EXISTING", contracts=20, price="0.50")
        portfolio = _make_portfolio(positions={"EXISTING": 40})

        result = limits.check(trade, portfolio)

        assert result.approved
        assert result.adjusted_size is not None
        assert result.adjusted_size == 10  # 50 - 40
        assert result.limit_triggered == "max_position_per_market"

    def test_drawdown_circuit_breaker(self, risk_config: RiskConfig) -> None:
        """Trading should halt when drawdown exceeds max_drawdown_pct."""
        limits = RiskLimits(risk_config)
        trade = _make_trade(contracts=5, price="0.50")
        # max_drawdown_pct = 0.15 -> 15% drawdown from peak
        # Peak 10000, current 8400 -> drawdown = 16%
        portfolio = _make_portfolio(
            peak_equity="10000",
            current_equity="8400",
        )

        result = limits.check(trade, portfolio)

        assert not result.approved
        assert result.limit_triggered == "max_drawdown"
