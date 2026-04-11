from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from moneygone.exchange.errors import OrderError
from moneygone.exchange.types import Action, Market, MarketResult, MarketStatus, Order, OrderStatus, Side
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

    @property
    def open_order_count(self) -> int:
        return len(self._orders)


class _FakeSports:
    def __init__(self, snapshots: dict[str, dict[str, int | str]]) -> None:
        self._snapshots = snapshots

    async def get_snapshot(self, market: Market) -> dict[str, int | str] | None:
        return self._snapshots.get(market.ticker)


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
