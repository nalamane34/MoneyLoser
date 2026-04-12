from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from moneygone.exchange.errors import OrderError
from moneygone.exchange.types import Action, Fill, Market, MarketResult, MarketStatus, Order, OrderStatus, Side, WSEvent
from moneygone.execution.engine import CategoryProvider, ExecutionEngine, TradeDecision
from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.data.market_discovery import MarketCategory
from moneygone.risk.portfolio import LocalPosition
from moneygone.signals.edge import EdgeResult
from moneygone.sizing.kelly import SizeResult
from moneygone.sizing.risk_limits import RiskCheckResult


def _market(ticker: str, *, event_ticker: str, yes_sub_title: str) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker="KXMLBGAME",
        title="Arizona vs Philadelphia",
        subtitle="",
        yes_sub_title=yes_sub_title,
        no_sub_title="",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.53"),
        yes_ask=Decimal("0.55"),
        last_price=Decimal("0.54"),
        volume=100,
        open_interest=10,
        close_time=datetime.now(timezone.utc),
        result=MarketResult.NOT_SETTLED,
        category="sports",
    )


class _FakeOrders:
    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders
        self.reconcile_calls = 0
        self.cancelled_order_ids: list[str] = []

    def get_open_orders(self) -> list[Order]:
        return list(self._orders)

    async def reconcile(self) -> None:
        self.reconcile_calls += 1

    async def cancel_all(self, ticker: str | None = None) -> int:
        return 0

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled_order_ids.append(order_id)
        self._orders = [order for order in self._orders if order.order_id != order_id]
        return True

    @property
    def open_order_count(self) -> int:
        return len(self._orders)


class _FakeSports:
    def __init__(self, snapshots: dict[str, dict[str, int | str]]) -> None:
        self._snapshots = snapshots

    async def get_snapshot(self, market: Market) -> dict[str, int | str] | None:
        return self._snapshots.get(market.ticker)


class _SharedStore:
    def __init__(self, market_row=None, orderbook_row=None) -> None:
        self.market_row = market_row
        self.orderbook_row = orderbook_row
        self.market_calls: list[tuple[str, str]] = []
        self.orderbook_calls: list[tuple[str, str]] = []
        self.market_rows_since_calls: list[tuple[str, int]] = []
        self.orderbook_rows_since_calls: list[tuple[str, int]] = []

    def get_market_state_at(self, ticker, as_of, *, table):
        self.market_calls.append((ticker, table))
        return self.market_row

    def get_orderbook_at(self, ticker, as_of, *, table):
        self.orderbook_calls.append((ticker, table))
        return self.orderbook_row

    def get_market_state_rows_since(self, since, *, table, limit):
        self.market_rows_since_calls.append((table, limit))
        if self.market_row is None:
            return []
        row = dict(self.market_row)
        row.setdefault("ingested_at", since)
        return [row]

    def get_orderbook_rows_since(self, since, *, table, limit):
        self.orderbook_rows_since_calls.append((table, limit))
        if self.orderbook_row is None:
            return []
        row = dict(self.orderbook_row)
        row.setdefault("ingested_at", since)
        return [row]


def _engine_with_context(
    *,
    orders: list[Order] | None = None,
    positions: dict[str, LocalPosition] | None = None,
) -> ExecutionEngine:
    engine = object.__new__(ExecutionEngine)
    engine._market_cache = {
        "KXMLBGAME-ARI": _market(
            "KXMLBGAME-ARI",
            event_ticker="KXMLBGAME-ARIPHI",
            yes_sub_title="Arizona",
        ),
        "KXMLBGAME-PHI": _market(
            "KXMLBGAME-PHI",
            event_ticker="KXMLBGAME-ARIPHI",
            yes_sub_title="Philadelphia",
        ),
    }
    engine._orders = _FakeOrders(orders or [])
    engine._risk = SimpleNamespace(
        _portfolio=SimpleNamespace(positions=positions or {}),
    )
    engine._sports = _FakeSports(
        {
            "KXMLBGAME-ARI": {"event_id": "game-1", "is_home_team": 1},
            "KXMLBGAME-PHI": {"event_id": "game-1", "is_home_team": 0},
        }
    )
    engine._decision_context = {}
    engine._decision_context_by_client_order_id = {}
    return engine


def _non_binary_market(ticker: str, *, event_ticker: str) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker="KXLOWTPHX",
        title="Will Phoenix low temp be below 65.5F?",
        subtitle="",
        yes_sub_title="",
        no_sub_title="",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.53"),
        yes_ask=Decimal("0.55"),
        last_price=Decimal("0.54"),
        volume=100,
        open_interest=10,
        close_time=datetime.now(timezone.utc),
        result=MarketResult.NOT_SETTLED,
        category="weather",
    )


def test_sports_outcome_key_treats_yes_away_and_no_home_as_same_view() -> None:
    engine = _engine_with_context()

    home_no = engine._sports_outcome_key(
        "KXMLBGAME-ARI",
        "no",
        {"event_id": "game-1", "is_home_team": 1},
    )
    away_yes = engine._sports_outcome_key(
        "KXMLBGAME-PHI",
        "yes",
        {"event_id": "game-1", "is_home_team": 0},
    )

    assert home_no == away_yes == ("sports_game_winner", "game-1", "away")


@pytest.mark.asyncio
async def test_event_outcome_key_treats_yes_away_and_no_home_as_same_view() -> None:
    engine = _engine_with_context()

    away_yes = await engine._outcome_key_for_trade(
        "KXMLBGAME-PHI",
        "yes",
        action="buy",
    )
    home_no = await engine._outcome_key_for_trade(
        "KXMLBGAME-ARI",
        "no",
        action="buy",
    )

    assert away_yes == home_no == (
        "event_outcome",
        "KXMLBGAME-ARIPHI",
        "philadelphia",
    )


@pytest.mark.asyncio
async def test_duplicate_equivalent_resting_order_is_blocked() -> None:
    engine = _engine_with_context(
        orders=[
            Order(
                order_id="order-1",
                ticker="KXMLBGAME-PHI",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=3,
                remaining_count=3,
                price=Decimal("0.55"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=datetime.now(timezone.utc),
            )
        ]
    )

    conflict = await engine._find_duplicate_exposure(
        "KXMLBGAME-ARI",
        "no",
        action="buy",
        sports_snapshot={"event_id": "game-1", "is_home_team": 1},
    )

    assert conflict == {
        "kind": "equivalent_open_order",
        "ticker": "KXMLBGAME-PHI",
        "side": "yes",
        "action": "buy",
    }


@pytest.mark.asyncio
async def test_duplicate_equivalent_position_is_blocked() -> None:
    engine = _engine_with_context(
        positions={
            "KXMLBGAME-PHI": LocalPosition(
                ticker="KXMLBGAME-PHI",
                yes_count=3,
            )
        }
    )

    conflict = await engine._find_duplicate_exposure(
        "KXMLBGAME-ARI",
        "no",
        action="buy",
        sports_snapshot={"event_id": "game-1", "is_home_team": 1},
    )

    assert conflict == {
        "kind": "equivalent_position",
        "ticker": "KXMLBGAME-PHI",
        "side": "yes",
        "contracts": 3,
    }


@pytest.mark.asyncio
async def test_duplicate_event_cluster_resting_order_is_blocked_for_non_binary_markets() -> None:
    now = datetime.now(timezone.utc)
    engine = object.__new__(ExecutionEngine)
    engine._market_cache = {
        "KXLOWTPHX-26APR11-B65.5": _non_binary_market(
            "KXLOWTPHX-26APR11-B65.5",
            event_ticker="KXLOWTPHX-26APR11",
        ),
        "KXLOWTPHX-26APR11-B63.5": _non_binary_market(
            "KXLOWTPHX-26APR11-B63.5",
            event_ticker="KXLOWTPHX-26APR11",
        ),
    }
    engine._orders = _FakeOrders(
        [
            Order(
                order_id="weather-order-1",
                ticker="KXLOWTPHX-26APR11-B63.5",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=1,
                remaining_count=1,
                price=Decimal("0.55"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=now,
            )
        ]
    )
    engine._risk = SimpleNamespace(
        _portfolio=SimpleNamespace(positions={}),
    )
    engine._sports = None
    engine._decision_context = {}
    engine._decision_context_by_client_order_id = {}

    conflict = await engine._find_duplicate_exposure(
        "KXLOWTPHX-26APR11-B65.5",
        "yes",
        action="buy",
        sports_snapshot=None,
    )

    assert conflict == {
        "kind": "event_cluster_open_order",
        "ticker": "KXLOWTPHX-26APR11-B63.5",
        "side": "yes",
        "action": "buy",
    }


def test_refresh_market_cache_from_shared_store_updates_market_cache() -> None:
    now = datetime(2026, 4, 11, 4, 0, tzinfo=timezone.utc)
    engine = object.__new__(ExecutionEngine)
    engine._store = _SharedStore(
        market_row={
            "ticker": "KXTEST",
            "event_ticker": "EVT-TEST",
            "series_ticker": "KXSERIES",
            "title": "Shared market",
            "status": "open",
            "yes_bid": 0.44,
            "yes_ask": 0.46,
            "last_price": 0.45,
            "volume": 123,
            "open_interest": 50,
            "close_time": now.isoformat(),
            "category": "sports",
        }
    )
    engine._shared_market_state_table = "market_data.market_states"
    engine._shared_orderbook_table = None
    engine._market_cache = {}

    engine._refresh_market_cache_from_shared_store("KXTEST", now)

    market = engine._market_cache["KXTEST"]
    assert market.title == "Shared market"
    assert market.yes_bid == Decimal("0.44")
    assert market.status == MarketStatus.OPEN


def test_get_orderbook_from_shared_store_returns_sorted_snapshot() -> None:
    now = datetime(2026, 4, 11, 4, 5, tzinfo=timezone.utc)
    engine = object.__new__(ExecutionEngine)
    engine._store = _SharedStore(
        orderbook_row={
            "yes_levels": [
                {"price": 0.48, "contracts": 25},
                {"price": 0.44, "contracts": 10},
            ],
            "no_levels": [
                {"price": 0.40, "contracts": 12},
                {"price": 0.43, "contracts": 5},
            ],
            "seq": 9,
            "snapshot_time": now.isoformat(),
        }
    )
    engine._shared_market_state_table = None
    engine._shared_orderbook_table = "market_data.orderbook_snapshots"

    orderbook = engine._get_orderbook_from_shared_store("KXTEST", now)

    assert orderbook is not None
    assert orderbook.yes_bids[0].price == Decimal("0.44")
    assert orderbook.yes_bids[-1].price == Decimal("0.48")
    assert orderbook.no_bids[-1].price == Decimal("0.43")


def test_poll_shared_market_data_once_updates_hot_caches() -> None:
    now = datetime(2026, 4, 11, 4, 10, tzinfo=timezone.utc)
    store = _SharedStore(
        market_row={
            "ticker": "KXTEST",
            "event_ticker": "EVT-TEST",
            "series_ticker": "KXSERIES",
            "title": "Shared live market",
            "status": "open",
            "yes_bid": 0.51,
            "yes_ask": 0.53,
            "last_price": 0.52,
            "volume": 90,
            "open_interest": 22,
            "close_time": now.isoformat(),
            "snapshot_time": now.isoformat(),
            "ingested_at": now,
        },
        orderbook_row={
            "ticker": "KXTEST",
            "yes_levels": [{"price": 0.51, "contracts": 15}],
            "no_levels": [{"price": 0.47, "contracts": 11}],
            "seq": 12,
            "snapshot_time": now.isoformat(),
            "ingested_at": now,
        },
    )
    engine = object.__new__(ExecutionEngine)
    engine._store = store
    engine._market_cache = {}
    engine._shared_orderbook_cache = {}
    engine._shared_market_state_table = "market_data.market_states"
    engine._shared_orderbook_table = "market_data.orderbook_snapshots"
    engine._shared_market_cursor_time = now - timedelta(seconds=1)
    engine._shared_market_cursor_keys = set()
    engine._shared_orderbook_cursor_time = now - timedelta(seconds=1)
    engine._shared_orderbook_cursor_keys = set()

    engine._poll_shared_market_data_once()

    assert engine._market_cache["KXTEST"].last_price == Decimal("0.52")
    assert engine._shared_orderbook_cache["KXTEST"].seq == 12


@pytest.mark.asyncio
async def test_reconcile_runs_on_startup() -> None:
    orders = _FakeOrders([])
    engine = object.__new__(ExecutionEngine)
    engine._watched = []
    engine._config = SimpleNamespace(evaluation_interval_seconds=30)
    engine._running = False
    engine._eval_task = None
    engine._market_cache = {}
    engine._market_categories = {}
    engine._last_eval = {}
    engine._decision_context = {}
    engine._decision_context_by_client_order_id = {}
    engine._last_universe_refresh = None
    engine._last_order_reconcile = None
    engine._sportsbook_parquet_path = None
    engine._store = None
    engine._shared_market_state_table = None
    engine._shared_orderbook_table = None
    engine._sports = None
    engine._category_providers = {}
    engine._rest = SimpleNamespace()
    engine._fills = SimpleNamespace()
    engine._strategy = SimpleNamespace()
    engine._recorder = None
    engine._pipeline = SimpleNamespace()
    engine._model = SimpleNamespace()
    engine._edge_calc = SimpleNamespace()
    engine._sizer = SimpleNamespace()
    engine._orders = orders
    engine._prediction_rows = []
    engine._feature_rows = []
    engine._risk = SimpleNamespace(
        _portfolio=SimpleNamespace(
            cash=Decimal("10"),
            positions={},
            sync_with_exchange=lambda _rest: None,
        )
    )

    class _FakeWS:
        def set_on_event(self, _cb) -> None:
            return None

        async def connect(self) -> None:
            return None

        async def wait_connected(self) -> None:
            return None

        async def subscribe_orderbook(self, _tickers) -> None:
            return None

        async def subscribe_ticker(self, _tickers) -> None:
            return None

        async def subscribe_trades(self, _tickers) -> None:
            return None

        async def subscribe_fills(self) -> None:
            return None

        async def subscribe_positions(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    engine._ws = _FakeWS()

    async def _sync_with_exchange(_rest) -> None:
        return None

    engine._risk._portfolio.sync_with_exchange = _sync_with_exchange

    async def _refresh_market_universe(*, force: bool = False) -> None:
        return None

    engine._refresh_market_universe = _refresh_market_universe

    await engine.start()
    await engine.stop()

    assert orders.reconcile_calls >= 1


@pytest.mark.asyncio
async def test_cancel_stale_open_orders_uses_config_threshold() -> None:
    now = datetime.now(timezone.utc)
    orders = _FakeOrders(
        [
            Order(
                order_id="stale-order",
                ticker="KXMLBGAME-PHI",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=3,
                remaining_count=3,
                price=Decimal("0.55"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=now - timedelta(seconds=180),
            ),
            Order(
                order_id="fresh-order",
                ticker="KXMLBGAME-ARI",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=2,
                remaining_count=2,
                price=Decimal("0.54"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=now - timedelta(seconds=10),
            ),
        ]
    )
    engine = object.__new__(ExecutionEngine)
    engine._config = SimpleNamespace(max_order_staleness_seconds=90)
    engine._orders = orders

    cancelled = await engine._cancel_stale_open_orders()

    assert cancelled == 1
    assert orders.cancelled_order_ids == ["stale-order"]
    assert [order.order_id for order in orders.get_open_orders()] == ["fresh-order"]


@pytest.mark.asyncio
async def test_cancel_stale_open_orders_rechecks_ticker_immediately() -> None:
    now = datetime.now(timezone.utc)
    orders = _FakeOrders(
        [
            Order(
                order_id="stale-order",
                ticker="KXMLBGAME-PHI",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=1,
                remaining_count=1,
                price=Decimal("0.55"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=now - timedelta(seconds=45),
            ),
        ]
    )
    engine = object.__new__(ExecutionEngine)
    engine._config = SimpleNamespace(max_order_staleness_seconds=30)
    engine._orders = orders
    engine._running = True
    engine._watched = ["KXMLBGAME-PHI"]
    engine._market_cache = {
        "KXMLBGAME-PHI": _market(
            "KXMLBGAME-PHI",
            event_ticker="KXMLBGAME-ARIPHI",
            yes_sub_title="Philadelphia",
        )
    }
    engine._last_eval = {}
    engine._stale_rechecks_inflight = set()

    calls: list[tuple[str, str, str | None] | tuple[str, str]] = []

    async def _evaluate_market(ticker: str, *, cycle_id: str | None = None):
        calls.append(("evaluate", ticker, cycle_id))
        return "decision"

    async def _execute_decision(decision) -> None:
        calls.append(("execute", decision))

    engine.evaluate_market = _evaluate_market
    engine.execute_decision = _execute_decision

    cancelled = await engine._cancel_stale_open_orders()

    assert cancelled == 1
    assert orders.cancelled_order_ids == ["stale-order"]
    assert calls[0][0] == "evaluate"
    assert calls[0][1] == "KXMLBGAME-PHI"
    assert str(calls[0][2]).startswith("stale-recheck:")
    assert calls[1] == ("execute", "decision")


@pytest.mark.asyncio
async def test_cancel_stale_open_orders_does_not_recheck_until_cancel_is_confirmed() -> None:
    class _PendingCancelOrders(_FakeOrders):
        async def cancel_order(self, order_id: str) -> bool:
            self.cancelled_order_ids.append(order_id)
            return False

    now = datetime.now(timezone.utc)
    orders = _PendingCancelOrders(
        [
            Order(
                order_id="stale-order",
                ticker="KXMLBGAME-PHI",
                side=Side.YES,
                action=Action.BUY,
                status=OrderStatus.RESTING,
                count=2,
                remaining_count=2,
                price=Decimal("0.54"),
                taker_fees=Decimal("0"),
                maker_fees=Decimal("0"),
                created_time=now - timedelta(seconds=45),
            ),
        ]
    )
    engine = object.__new__(ExecutionEngine)
    engine._config = SimpleNamespace(max_order_staleness_seconds=30)
    engine._orders = orders

    calls: list[tuple[str, str, str | None] | tuple[str, str]] = []

    async def _evaluate_market(ticker: str, *, cycle_id: str | None = None):
        calls.append(("evaluate", ticker, cycle_id))
        return "decision"

    async def _execute_decision(decision) -> None:
        calls.append(("execute", decision))

    engine.evaluate_market = _evaluate_market
    engine.execute_decision = _execute_decision

    cancelled = await engine._cancel_stale_open_orders()

    assert cancelled == 1
    assert orders.cancelled_order_ids == ["stale-order"]
    assert calls == []


@pytest.mark.asyncio
async def test_handle_fill_event_ignores_duplicate_fill_ids() -> None:
    engine = object.__new__(ExecutionEngine)
    engine._seen_fill_ids = set()
    from collections import deque
    engine._recent_fill_ids = deque()
    engine._orders = SimpleNamespace(on_fill=lambda _fill: (_ for _ in ()).throw(AssertionError("should not reprocess")))
    engine._risk = SimpleNamespace(post_trade_update=lambda _fill: (_ for _ in ()).throw(AssertionError("should not reprocess")))
    engine._fills = SimpleNamespace(on_fill=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not reprocess")))
    engine._resolve_decision_context = lambda _fill: None

    event = WSEvent(
        channel="fill",
        type="fill",
        data={
            "fill_id": "fill-1",
            "market_ticker": "KXTEST-1",
            "side": "yes",
            "action": "buy",
            "count_fp": "1.00",
            "yes_price_dollars": "0.55",
            "fee_cost": "0.00",
        },
        seq=1,
        timestamp=datetime.now(timezone.utc),
    )

    engine._seen_fill_ids.add("fill-1")

    await engine._handle_fill_event(event)


@pytest.mark.asyncio
async def test_on_event_market_positions_resyncs_portfolio() -> None:
    engine = object.__new__(ExecutionEngine)
    synced: list[str] = []

    async def _sync_with_exchange(_rest) -> None:
        synced.append("sync")

    engine._risk = SimpleNamespace(
        _portfolio=SimpleNamespace(
            sync_with_exchange=_sync_with_exchange,
            positions={},
            cash=Decimal("10"),
        )
    )
    engine._rest = SimpleNamespace()
    engine._last_portfolio_sync = datetime.now(timezone.utc) - timedelta(seconds=10)

    await engine.on_event(
        WSEvent(
            channel="market_positions",
            type="position",
            data={"market_ticker": "KXTEST-1"},
            seq=1,
            timestamp=datetime.now(timezone.utc),
        )
    )

    assert synced == ["sync"]


def test_prune_settled_tickers_cleans_decision_context_by_ticker() -> None:
    engine = object.__new__(ExecutionEngine)
    now = datetime.now(timezone.utc)
    engine._watched = ["KXSETTLED-1"]
    engine._market_cache = {
        "KXSETTLED-1": Market(
            ticker="KXSETTLED-1",
            event_ticker="KXEVENT-1",
            series_ticker="KXSERIES-1",
            title="Settled market",
            subtitle="",
            yes_sub_title="Yes",
            no_sub_title="No",
            status=MarketStatus.SETTLED,
            yes_bid=Decimal("0.50"),
            yes_ask=Decimal("0.52"),
            last_price=Decimal("0.51"),
            volume=10,
            open_interest=5,
            close_time=now - timedelta(minutes=1),
            result=MarketResult.YES,
            category="politics",
        )
    }
    engine._market_categories = {"KXSETTLED-1": MarketCategory.POLITICS}
    engine._last_eval = {"KXSETTLED-1": now}
    decision = _make_decision(ticker="KXSETTLED-1")
    engine._decision_context = {"order-1": decision}
    engine._decision_context_by_client_order_id = {"coid-1": decision}

    pruned = engine._prune_settled_tickers()

    assert pruned == 1
    assert engine._decision_context == {}
    assert engine._decision_context_by_client_order_id == {}


# ---------------------------------------------------------------------------
# Demo-only model safety tests
# ---------------------------------------------------------------------------


def _make_prediction(*, model_name: str = "sharp_sportsbook") -> ModelPrediction:
    return ModelPrediction(
        probability=0.60,
        raw_probability=0.60,
        confidence=0.50,
        model_name=model_name,
        model_version="v1",
        features_used={},
        prediction_time=datetime.now(timezone.utc),
    )


def _make_edge() -> EdgeResult:
    return EdgeResult(
        raw_edge=0.05,
        fee_adjusted_edge=0.04,
        implied_probability=0.55,
        model_probability=0.60,
        available_liquidity=10,
        estimated_fill_rate=0.8,
        is_actionable=True,
        side="yes",
        action="buy",
        target_price=Decimal("0.55"),
        expected_value=Decimal("0.04"),
    )


def _make_size() -> SizeResult:
    return SizeResult(
        kelly_fraction=0.10,
        adjusted_fraction=0.025,
        contracts=3,
        dollar_risk=Decimal("1.65"),
        dollar_ev=Decimal("0.12"),
        capped_by=None,
    )


def _make_risk_check() -> RiskCheckResult:
    return RiskCheckResult(approved=True, adjusted_size=None, limit_triggered=None, rejection_reason=None)


def _make_decision(*, model_name: str = "sharp_sportsbook", ticker: str = "KXTEST-YES") -> TradeDecision:
    return TradeDecision(
        ticker=ticker,
        edge_result=_make_edge(),
        size_result=_make_size(),
        risk_check=_make_risk_check(),
        prediction=_make_prediction(model_name=model_name),
        timestamp=datetime.now(timezone.utc),
        cycle_id="test-cycle",
        category="economics",
    )


class _DemoOnlyModel:
    """Fake model that declares itself as demo-only."""
    name = "future_experimental"
    version = "v0"
    demo_only = True


class _RealModel:
    """Fake model with real edge."""
    name = "sharp_sportsbook"
    version = "v1"
    demo_only = False


class _FakeOrderbook:
    """Minimal orderbook for execute_decision tests."""
    def __init__(self):
        self.yes_bids = [{"price": Decimal("0.55"), "quantity": 10}]
        self.no_bids = [{"price": Decimal("0.45"), "quantity": 10}]


class _FakeStrategy:
    """Tracks whether execute was called."""
    def __init__(self):
        self.executed = False

    async def execute(self, edge, size, orders, orderbook):
        self.executed = True
        return None  # No actual order created


class _FailingStrategy:
    """Raises a deterministic order error for execute_decision tests."""

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, edge, size, orders, orderbook):
        self.executed = True
        raise OrderError("boom", status_code=403)


class _FakeWSForGuard:
    """Returns a fake orderbook."""
    def get_orderbook(self, ticker):
        return _FakeOrderbook()


def _engine_for_guard_test(*, demo_mode: bool, category_providers=None) -> ExecutionEngine:
    """Create a minimal engine for testing execute_decision guards."""
    engine = object.__new__(ExecutionEngine)
    engine._demo_mode = demo_mode
    engine._category_providers = category_providers or {}
    engine._market_categories = {}
    engine._market_cache = {}
    engine._decision_context = {}
    engine._decision_context_by_client_order_id = {}
    engine._ws = _FakeWSForGuard()
    engine._rest = SimpleNamespace()
    engine._fills = SimpleNamespace(record_submission=lambda: None)
    engine._strategy = _FakeStrategy()
    engine._orders = _FakeOrders([])
    return engine


@pytest.mark.asyncio
async def test_execute_decision_blocks_market_baseline_in_live_mode() -> None:
    """market_baseline model must be blocked when demo_mode=False."""
    engine = _engine_for_guard_test(demo_mode=False)
    decision = _make_decision(model_name="market_baseline")

    await engine.execute_decision(decision)

    # Strategy should NOT have been called
    assert engine._strategy.executed is False


@pytest.mark.asyncio
async def test_execute_decision_allows_market_baseline_in_demo_mode() -> None:
    """market_baseline model should be allowed when demo_mode=True."""
    engine = _engine_for_guard_test(demo_mode=True)
    decision = _make_decision(model_name="market_baseline")

    await engine.execute_decision(decision)

    # Strategy SHOULD have been called
    assert engine._strategy.executed is True


@pytest.mark.asyncio
async def test_execute_decision_allows_real_model_in_live_mode() -> None:
    """Real models (sharp_sportsbook) must NOT be blocked in live mode."""
    engine = _engine_for_guard_test(demo_mode=False)
    decision = _make_decision(model_name="sharp_sportsbook")

    await engine.execute_decision(decision)

    # Strategy SHOULD have been called
    assert engine._strategy.executed is True


@pytest.mark.asyncio
async def test_execute_decision_blocks_demo_only_model_via_flag() -> None:
    """Any model with demo_only=True must be blocked in live mode, even
    if its name isn't 'market_baseline'."""
    providers = {
        MarketCategory.ECONOMICS: CategoryProvider(
            category=MarketCategory.ECONOMICS,
            model=_DemoOnlyModel(),  # type: ignore[arg-type]
            pipeline=SimpleNamespace(),  # type: ignore[arg-type]
            get_context_data=None,
        ),
    }
    engine = _engine_for_guard_test(demo_mode=False, category_providers=providers)
    # Simulate the engine knowing this ticker's category
    engine._market_categories["KXTEST-YES"] = MarketCategory.ECONOMICS

    decision = _make_decision(model_name="future_experimental", ticker="KXTEST-YES")

    await engine.execute_decision(decision)

    # Strategy should NOT have been called
    assert engine._strategy.executed is False


@pytest.mark.asyncio
async def test_execute_decision_allows_non_demo_only_model_via_flag() -> None:
    """Models with demo_only=False must be allowed in live mode."""
    providers = {
        MarketCategory.SPORTS: CategoryProvider(
            category=MarketCategory.SPORTS,
            model=_RealModel(),  # type: ignore[arg-type]
            pipeline=SimpleNamespace(),  # type: ignore[arg-type]
            get_context_data=None,
        ),
    }
    engine = _engine_for_guard_test(demo_mode=False, category_providers=providers)
    engine._market_categories["KXTEST-YES"] = MarketCategory.SPORTS

    decision = _make_decision(model_name="sharp_sportsbook", ticker="KXTEST-YES")

    await engine.execute_decision(decision)

    # Strategy SHOULD have been called
    assert engine._strategy.executed is True


@pytest.mark.asyncio
async def test_execute_decision_handles_order_error_without_raising() -> None:
    """Order placement failures should be logged and contained."""
    engine = _engine_for_guard_test(demo_mode=False)
    engine._strategy = _FailingStrategy()
    decision = _make_decision(model_name="sharp_sportsbook")

    await engine.execute_decision(decision)

    assert engine._strategy.executed is True
    assert engine._decision_context == {}


def test_market_baseline_model_has_demo_only_flag() -> None:
    """MarketBaselineModel must always declare itself as demo_only."""
    from moneygone.models.market_baseline import MarketBaselineModel

    model = MarketBaselineModel()
    assert model.demo_only is True, "MarketBaselineModel must have demo_only=True"


def test_real_models_have_demo_only_false() -> None:
    """Production models should NOT be demo_only."""
    from moneygone.models.sharp_sportsbook import SharpSportsbookModel

    model = SharpSportsbookModel()
    assert model.demo_only is False, "SharpSportsbookModel should have demo_only=False"
