from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from moneygone.strategies.resolution_sniper import (
    ResolutionSniper,
    SnipeConfig,
    SnipeOpportunity,
)


class _FakePortfolio:
    def __init__(self, *, cash: Decimal) -> None:
        self.cash = cash
        self.positions: dict[str, object] = {}
        self.sync_calls = 0

    async def sync_with_exchange(self, _client: object) -> None:
        self.sync_calls += 1


class _FakeRiskManager:
    def __init__(self, *, available_cash: Decimal, paused: bool = False) -> None:
        self.available_cash = available_cash
        self.paused = paused
        self.pause_reasons = {"test": "paused"} if paused else {}

    def check_circuit_breakers(self) -> bool:
        return False

    def is_trading_paused(self) -> bool:
        return self.paused

    def get_capital_view(self):
        return SimpleNamespace(available_cash=self.available_cash)


def _opportunity(*, ticker: str = "KXTEST-YES", price: str = "0.80") -> SnipeOpportunity:
    return SnipeOpportunity(
        ticker=ticker,
        outcome_known=True,
        predicted_resolution="yes",
        confidence=0.99,
        current_market_price=Decimal(price),
        expected_payout=Decimal("1.00"),
        expected_profit=Decimal("0.10"),
        signal_source="espn",
        signal_data={},
        detected_at=datetime.now(timezone.utc),
    )


def test_estimate_safe_size_uses_available_cash_and_event_exposure() -> None:
    sniper = ResolutionSniper(
        rest_client=SimpleNamespace(),
        order_manager=SimpleNamespace(),
        fee_calculator=SimpleNamespace(),
        portfolio=_FakePortfolio(cash=Decimal("10.00")),
        config=SnipeConfig(
            max_contracts_per_snipe=20,
            max_exposure_per_event=20.0,
        ),
    )

    size = sniper._estimate_safe_size(_opportunity())

    assert size == 11

    sniper._portfolio.cash = Decimal("100.00")
    sniper._event_exposure["KXTEST"] = 8.8
    constrained = sniper._estimate_safe_size(_opportunity(ticker="KXTEST-YES"))

    assert constrained == 13


@pytest.mark.asyncio
async def test_maybe_sync_portfolio_refreshes_before_execution() -> None:
    portfolio = _FakePortfolio(cash=Decimal("12.34"))
    sniper = ResolutionSniper(
        rest_client=SimpleNamespace(),
        order_manager=SimpleNamespace(),
        fee_calculator=SimpleNamespace(),
        portfolio=portfolio,
    )

    await sniper._maybe_sync_portfolio(force=True)

    assert portfolio.sync_calls == 1
    assert sniper._last_portfolio_sync is not None


def test_sniper_uses_shared_available_cash_when_present() -> None:
    sniper = ResolutionSniper(
        rest_client=SimpleNamespace(),
        order_manager=SimpleNamespace(),
        fee_calculator=SimpleNamespace(),
        portfolio=_FakePortfolio(cash=Decimal("100.00")),
        risk_manager=_FakeRiskManager(available_cash=Decimal("4.00")),
        config=SnipeConfig(max_contracts_per_snipe=20),
    )

    size = sniper._estimate_safe_size(_opportunity(price="0.80"))

    assert size == 4


def test_sniper_respects_global_pause() -> None:
    sniper = ResolutionSniper(
        rest_client=SimpleNamespace(),
        order_manager=SimpleNamespace(),
        fee_calculator=SimpleNamespace(),
        risk_manager=_FakeRiskManager(available_cash=Decimal("10.00"), paused=True),
    )

    assert sniper._should_execute(_opportunity()) is False
