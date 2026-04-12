from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from moneygone.data.sports.espn import OutcomeSignal
from moneygone.exchange.types import Action, Fill, Order, OrderStatus, Side
from moneygone.strategies.resolution_sniper import (
    ContractMapping,
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


class _FakeClient:
    def __init__(self, *, fills: list[Fill] | None = None) -> None:
        self.fills = fills or []
        self.last_fill_filters: dict[str, object] | None = None

    async def get_fills(self, **filters):
        self.last_fill_filters = dict(filters)
        return list(self.fills)


class _FakeOrderManager:
    def __init__(self, *, order: Order) -> None:
        self.order = order
        self.request = None

    async def submit_order(self, request):
        self.request = request
        return self.order

    def get_open_orders(self):
        return []


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


def _order(
    *,
    order_id: str = "order-1",
    ticker: str = "KXTEST-YES",
    side: Side = Side.YES,
    status: OrderStatus = OrderStatus.RESTING,
) -> Order:
    return Order(
        order_id=order_id,
        ticker=ticker,
        side=side,
        action=Action.BUY,
        status=status,
        count=10,
        remaining_count=10,
        price=Decimal("0.50"),
        no_price=Decimal("0.50"),
        taker_fees=Decimal("0.00"),
        maker_fees=Decimal("0.00"),
        created_time=datetime.now(timezone.utc),
    )


def _fill(
    *,
    fill_id: str = "fill-1",
    ticker: str = "KXTEST-NO",
    side: Side = Side.NO,
    price: str = "0.05",
    no_price: str = "0.95",
    count: int = 3,
    order_id: str = "order-1",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        ticker=ticker,
        side=side,
        action=Action.BUY,
        count=count,
        price=Decimal(price),
        no_price=Decimal(no_price),
        fee_cost=Decimal("0.00"),
        is_taker=True,
        created_time=datetime.now(timezone.utc),
        order_id=order_id,
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


@pytest.mark.asyncio
async def test_execute_snipe_converts_no_orders_and_ignores_unfilled_ioc() -> None:
    client = _FakeClient(fills=[])
    order_manager = _FakeOrderManager(
        order=_order(
            order_id="order-no-fill",
            ticker="KXTEST-NO",
            side=Side.NO,
            status=OrderStatus.CANCELED,
        )
    )
    sniper = ResolutionSniper(
        rest_client=client,
        order_manager=order_manager,
        fee_calculator=SimpleNamespace(fee_per_contract=lambda *_args, **_kwargs: Decimal("0.00")),
        portfolio=_FakePortfolio(cash=Decimal("100.00")),
        config=SnipeConfig(max_contracts_per_snipe=5),
    )

    result = await sniper.execute_snipe(
        SnipeOpportunity(
            ticker="KXTEST-NO",
            outcome_known=True,
            predicted_resolution="no",
            confidence=1.0,
            current_market_price=Decimal("0.95"),
            expected_payout=Decimal("1.00"),
            expected_profit=Decimal("0.05"),
            signal_source="espn",
            signal_data={},
            detected_at=datetime.now(timezone.utc),
        )
    )

    assert result is None
    assert order_manager.request is not None
    assert order_manager.request.side is Side.NO
    assert order_manager.request.yes_price == Decimal("0.05")
    assert client.last_fill_filters == {"order_id": "order-no-fill"}
    assert sniper._event_exposure == {}
    assert sniper.snipe_history == []


@pytest.mark.asyncio
async def test_execute_snipe_records_actual_no_fill_cost_from_fills() -> None:
    fill = _fill(order_id="order-filled")
    client = _FakeClient(fills=[fill])
    order_manager = _FakeOrderManager(
        order=_order(
            order_id="order-filled",
            ticker="KXTEST-NO",
            side=Side.NO,
            status=OrderStatus.CANCELED,
        )
    )
    sniper = ResolutionSniper(
        rest_client=client,
        order_manager=order_manager,
        fee_calculator=SimpleNamespace(fee_per_contract=lambda *_args, **_kwargs: Decimal("0.00")),
        portfolio=_FakePortfolio(cash=Decimal("100.00")),
        config=SnipeConfig(max_contracts_per_snipe=5),
    )

    result = await sniper.execute_snipe(
        SnipeOpportunity(
            ticker="KXTEST-NO",
            outcome_known=True,
            predicted_resolution="no",
            confidence=1.0,
            current_market_price=Decimal("0.95"),
            expected_payout=Decimal("1.00"),
            expected_profit=Decimal("0.05"),
            signal_source="espn",
            signal_data={},
            detected_at=datetime.now(timezone.utc),
        )
    )

    assert result == fill
    assert order_manager.request is not None
    assert order_manager.request.yes_price == Decimal("0.05")
    assert sniper._event_exposure["KXTEST"] == pytest.approx(2.85)
    assert len(sniper.snipe_history) == 1
    record = sniper.snipe_history[0]
    assert record.contracts == 3
    assert record.entry_price == Decimal("0.9500")
    assert record.expected_profit == Decimal("0.1500")


def test_find_matching_mappings_ignores_undated_sports_futures() -> None:
    signal = OutcomeSignal(
        game_id="game-1",
        outcome="home",
        home_team="Pittsburgh",
        away_team="Washington",
        home_score=3,
        away_score=2,
        confidence=1.0,
        source="espn",
        sport="hockey",
        league="nhl",
        detected_at=datetime(2026, 4, 12, 5, 0, tzinfo=timezone.utc),
    )
    sniper = ResolutionSniper(
        rest_client=SimpleNamespace(),
        order_manager=SimpleNamespace(),
        fee_calculator=SimpleNamespace(),
        contract_mappings=[
            ContractMapping(
                ticker="KXNHL-26-MIN",
                category="sports",
                data_source="espn",
                source_params={"sport": "hockey", "league": "nhl"},
                resolution_field="winner",
            ),
            ContractMapping(
                ticker="KXNHLGAME-26APR12PITWSH-PIT",
                category="sports",
                data_source="espn",
                source_params={"sport": "hockey", "league": "nhl"},
                resolution_field="winner",
            ),
        ],
        config=SnipeConfig(sports_lookback_days=1, sports_lookahead_days=1),
    )

    matches = sniper._find_matching_mappings(signal)

    assert [mapping.ticker for mapping in matches] == ["KXNHLGAME-26APR12PITWSH-PIT"]
