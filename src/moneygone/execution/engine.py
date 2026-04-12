"""Main event-driven execution engine for live trading.

Orchestrates the full decision pipeline:
  1. Receive market events via WebSocket
  2. Build feature context from current market state
  3. Run feature pipeline to extract features
  4. Run model to predict probability
  5. Compute edge against the live orderbook
  6. Size the position via Kelly criterion
  7. Validate against risk limits
  8. Execute the trade via the chosen strategy

Supports multiple market categories (sports, crypto, weather, economics)
with category-specific models, feature pipelines, and data providers.

Also runs periodic re-evaluation of watched markets on a timer.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from moneygone.config import ExecutionConfig
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.market_discovery import MarketCategory, classify_market, MarketDiscoveryService
from moneygone.exchange.errors import OrderError
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import (
    Action,
    Fill,
    Market,
    MarketResult,
    MarketStatus,
    OrderbookSnapshot,
    Side,
    WSEvent,
)
from moneygone.exchange.ws_client import KalshiWebSocket
from moneygone.execution.fill_tracker import FillTracker
from moneygone.execution.order_manager import OrderManager
from moneygone.execution.strategies import ExecutionStrategy
from moneygone.features.base import FeatureContext
from moneygone.features.pipeline import FeaturePipeline
from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.risk.manager import RiskManager
from moneygone.signals.edge import EdgeCalculator, EdgeResult
from moneygone.sizing.kelly import KellySizer, SizeResult
from moneygone.sizing.risk_limits import ProposedTrade, RiskCheckResult

logger = structlog.get_logger(__name__)


@dataclass
class CategoryProvider:
    """Model + pipeline + data provider for a market category."""
    category: MarketCategory
    model: ProbabilityModel
    pipeline: FeaturePipeline
    get_context_data: Any = None  # async callable(market) -> dict for FeatureContext

_ZERO = Decimal("0")


# ---------------------------------------------------------------------------
# Trade decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """Full context for a trade decision, whether acted upon or not."""

    ticker: str
    edge_result: EdgeResult
    size_result: SizeResult
    risk_check: RiskCheckResult
    prediction: ModelPrediction
    timestamp: datetime
    cycle_id: str | None = None
    category: str = ""
    rank_score: float = 0.0


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Event-driven trading engine connecting all pipeline components.

    Parameters
    ----------
    rest_client:
        Authenticated REST client for exchange operations.
    ws_client:
        WebSocket client for real-time market data.
    feature_pipeline:
        Feature computation pipeline.
    model:
        Probability prediction model.
    edge_calculator:
        Edge computation engine.
    sizer:
        Kelly criterion position sizer.
    risk_manager:
        Pre-trade risk validator.
    order_manager:
        Order lifecycle manager.
    fill_tracker:
        Fill recording with prediction context.
    strategy:
        Execution strategy (passive, aggressive, or adaptive).
    config:
        Execution configuration (thresholds, intervals).
    watched_tickers:
        List of tickers to actively evaluate.
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        ws_client: KalshiWebSocket,
        feature_pipeline: FeaturePipeline,
        model: ProbabilityModel,
        edge_calculator: EdgeCalculator,
        sizer: KellySizer,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        fill_tracker: FillTracker,
        strategy: ExecutionStrategy,
        config: ExecutionConfig,
        watched_tickers: list[str] | None = None,
        store: Any | None = None,
        sports_snapshot_provider: Any | None = None,
        recorder: MarketDataRecorder | None = None,
        category_providers: dict[MarketCategory, CategoryProvider] | None = None,
        discovery_cache_path: Path | None = None,
        sportsbook_parquet_path: Path | None = None,
        shared_market_state_table: str | None = None,
        shared_orderbook_table: str | None = None,
        demo_mode: bool = False,
    ) -> None:
        self._rest = rest_client
        self._ws = ws_client
        self._demo_mode = demo_mode
        self._pipeline = feature_pipeline
        self._model = model
        self._edge_calc = edge_calculator
        self._sizer = sizer
        self._risk = risk_manager
        self._orders = order_manager
        self._fills = fill_tracker
        self._strategy = strategy
        self._config = config
        self._watched: list[str] = watched_tickers or []
        self._store = store
        self._sports = sports_snapshot_provider
        self._recorder = recorder
        self._category_providers = category_providers or {}
        self._discovery_cache_path = discovery_cache_path
        self._sportsbook_parquet_path = sportsbook_parquet_path
        self._shared_market_state_table = shared_market_state_table
        self._shared_orderbook_table = shared_orderbook_table
        self._last_parquet_mtime: float = 0.0

        self._running = False
        self._eval_task: asyncio.Task[None] | None = None
        self._stale_order_task: asyncio.Task[None] | None = None
        self._shared_market_data_task: asyncio.Task[None] | None = None
        self._market_cache: dict[str, Market] = {}
        self._shared_orderbook_cache: dict[str, OrderbookSnapshot] = {}
        self._market_categories: dict[str, MarketCategory] = {}
        self._last_eval: dict[str, datetime] = {}
        self._stale_rechecks_inflight: set[str] = set()
        self._decision_context: dict[str, TradeDecision] = {}
        self._decision_context_by_client_order_id: dict[str, TradeDecision] = {}
        self._last_universe_refresh: datetime | None = None
        self._last_order_reconcile: datetime | None = None
        self._active_cycle_id: str | None = None
        self._last_prune: datetime = datetime.now(timezone.utc)
        self._last_portfolio_sync: datetime = datetime.now(timezone.utc)
        self._seen_fill_ids: set[str] = set()
        self._recent_fill_ids: deque[str] = deque()
        self._prediction_rows: list[dict[str, Any]] = []
        self._feature_rows: list[dict[str, Any]] = []
        self._shared_market_cursor_time: datetime | None = None
        self._shared_market_cursor_keys: set[str] = set()
        self._shared_orderbook_cursor_time: datetime | None = None
        self._shared_orderbook_cursor_keys: set[str] = set()

    def set_shared_market_data_tables(
        self,
        *,
        market_state_table: str | None,
        orderbook_table: str | None,
    ) -> None:
        """Enable shared read-only market-data tables after startup."""
        self._shared_market_state_table = market_state_table
        self._shared_orderbook_table = orderbook_table

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the execution engine.

        Connects WebSocket, subscribes to channels, and launches the
        periodic evaluation loop.
        """
        logger.info(
            "engine.starting",
            watched_tickers=self._watched,
            eval_interval=self._config.evaluation_interval_seconds,
        )

        self._running = True

        # Sync portfolio with exchange to get current balance + positions
        try:
            await self._risk._portfolio.sync_with_exchange(self._rest)
            cash = self._risk._portfolio.cash
            logger.info(
                "engine.portfolio_synced",
                cash=str(cash),
                positions=len(self._risk._portfolio.positions),
            )
            if cash <= 0:
                logger.critical(
                    "engine.ZERO_BANKROLL",
                    cash=str(cash),
                    msg="Portfolio synced but bankroll is $0 — Kelly sizing will produce zero contracts. "
                        "Check account balance or deposit funds.",
                )
        except Exception:
            logger.critical(
                "engine.PORTFOLIO_SYNC_FAILED",
                msg="Could not sync portfolio with exchange. Trading will use bankroll=$0, "
                    "effectively disabling all position sizing. This is a critical failure.",
                exc_info=True,
            )

        await self._maybe_reconcile_open_orders(force=True)
        await self._cancel_stale_open_orders()

        self._ws.set_on_event(self.on_event)
        await self._refresh_market_universe(force=True)
        self._initialize_shared_market_data_sync()

        # Connect WebSocket and subscribe to essential channels only.
        # Subscribing to orderbook/ticker/trades for 10K+ tickers hangs
        # the connection — we fetch orderbooks on-demand via REST instead.
        await self._ws.connect()
        await self._ws.wait_connected()
        await self._ws.subscribe_fills()
        await self._ws.subscribe_positions()

        # Launch periodic maintenance/evaluation loops
        if self._shared_market_state_table is not None or self._shared_orderbook_table is not None:
            self._shared_market_data_task = asyncio.create_task(
                self._shared_market_data_loop()
            )
        self._stale_order_task = asyncio.create_task(self._stale_order_loop())
        self._eval_task = asyncio.create_task(self._periodic_evaluation_loop())

        logger.info("engine.started")

    async def stop(self) -> None:
        """Gracefully stop the engine.

        Cancels all open orders, stops WebSocket, and shuts down tasks.
        """
        logger.info("engine.stopping")
        self._running = False

        # Cancel periodic evaluation
        if self._eval_task is not None:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
            self._eval_task = None

        stale_order_task = getattr(self, "_stale_order_task", None)
        if stale_order_task is not None:
            stale_order_task.cancel()
            try:
                await stale_order_task
            except asyncio.CancelledError:
                pass
            self._stale_order_task = None

        shared_market_task = getattr(self, "_shared_market_data_task", None)
        if shared_market_task is not None:
            shared_market_task.cancel()
            try:
                await shared_market_task
            except asyncio.CancelledError:
                pass
            self._shared_market_data_task = None

        # Cancel all open orders
        try:
            cancelled = await self._orders.cancel_all()
            logger.info("engine.orders_cancelled", count=cancelled)
        except Exception:
            logger.warning("engine.cancel_all_failed", exc_info=True)

        self._flush_observability_buffers()

        # Disconnect WebSocket
        await self._ws.disconnect()

        logger.info("engine.stopped")

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, event: WSEvent) -> None:
        """Dispatch a WebSocket event to the appropriate handler.

        This method is set as the ``on_event`` callback on the
        WebSocket client.
        """
        try:
            if event.channel == "fill":
                await self._handle_fill_event(event)
            elif event.channel == "market_positions":
                await self._handle_position_event(event)
            elif event.channel == "ticker":
                await self._handle_ticker_event(event)
            elif event.channel == "orderbook_delta":
                await self._handle_orderbook_event(event)
            elif event.channel == "trade":
                await self._handle_trade_event(event)
        except Exception:
            logger.exception(
                "engine.event_handler_error",
                channel=event.channel,
                type=event.type,
            )

    async def _handle_fill_event(self, event: WSEvent) -> None:
        """Process a fill notification from the WebSocket."""
        data = event.data
        fill_id = data.get("fill_id", data.get("trade_id", ""))
        if fill_id and self._fill_already_processed(fill_id):
            logger.warning("engine.duplicate_fill_ignored", fill_id=fill_id)
            return

        count_raw = data.get("count_fp", data.get("count", 0))
        fill = Fill(
            fill_id=fill_id,
            ticker=data.get("market_ticker", data.get("ticker", "")),
            side=Side(data.get("side", "yes")),
            action=Action(data.get("action", "buy")),
            count=int(float(str(count_raw))),
            price=Decimal(str(data.get("yes_price_dollars", data.get("yes_price", 0)))),
            no_price=Decimal(str(data.get("no_price_dollars", 0))),
            fee_cost=Decimal(str(data.get("fee_cost", 0))),
            is_taker=bool(data.get("is_taker", False)),
            created_time=event.timestamp or datetime.now(timezone.utc),
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
            trade_id=data.get("trade_id", fill_id),
        )

        # Update order manager
        self._orders.on_fill(fill)

        # Update risk state (portfolio, drawdown, etc.)
        self._risk.post_trade_update(fill)
        decision = self._resolve_decision_context(fill)
        if decision is not None:
            self._fills.on_fill(
                fill,
                decision.prediction,
                decision.edge_result,
                cycle_id=decision.cycle_id,
                category=decision.category,
            )

        logger.info(
            "engine.fill_received",
            fill_id=fill.fill_id,
            ticker=fill.ticker,
            count=fill.count,
            price=str(fill.contract_price),
        )

    async def _handle_position_event(self, event: WSEvent) -> None:
        """Resync portfolio state when the exchange pushes a position update."""
        now = datetime.now(timezone.utc)
        if (now - self._last_portfolio_sync).total_seconds() < 5:
            return
        try:
            await self._risk._portfolio.sync_with_exchange(self._rest)
            self._last_portfolio_sync = now
            logger.info(
                "engine.position_resynced",
                ticker=event.data.get("market_ticker", event.data.get("ticker", "")),
                positions=len(self._risk._portfolio.positions),
                cash=str(self._risk._portfolio.cash),
            )
        except Exception:
            logger.warning("engine.position_resync_failed", exc_info=True)

    async def _handle_ticker_event(self, event: WSEvent) -> None:
        """Process a ticker update from the WebSocket."""
        data = event.data
        ticker = data.get("market_ticker", data.get("ticker", ""))
        if not ticker:
            return

        market = self._merge_market_update(ticker, data, event.timestamp)
        if market is not None:
            self._market_cache[ticker] = market
            if self._recorder is not None:
                await self._recorder.on_ticker_update({"data": self._market_to_row(market)})
        logger.debug("engine.ticker_update", ticker=ticker)

    async def _handle_orderbook_event(self, event: WSEvent) -> None:
        """Process an orderbook snapshot or delta from the WebSocket."""
        ticker = event.data.get("market_ticker", event.data.get("ticker", ""))
        if not ticker or self._recorder is None:
            return

        orderbook = self._ws.get_orderbook(ticker)
        if orderbook is None:
            return

        await self._recorder.on_orderbook_update(
            {"data": self._orderbook_to_row(orderbook)}
        )

    async def _handle_trade_event(self, event: WSEvent) -> None:
        """Process a public trade event."""
        if self._recorder is None:
            return
        data = event.data
        ticker = data.get("market_ticker", data.get("ticker", ""))
        if not ticker:
            return
        await self._recorder.on_trade(
            {
                "data": {
                    "trade_id": data.get("trade_id", ""),
                    "ticker": ticker,
                    "count": int(data.get("count", 0)),
                    "yes_price": float(data.get("yes_price_dollars", data.get("yes_price", 0)) or 0),
                    "taker_side": data.get("taker_side", "yes"),
                    "trade_time": (event.timestamp or datetime.now(timezone.utc)).isoformat(),
                }
            }
        )

    # ------------------------------------------------------------------
    # Market evaluation
    # ------------------------------------------------------------------

    def _log_candidate(
        self,
        *,
        ticker: str,
        category: str,
        status: str,
        cycle_id: str | None = None,
        reject_reason: str = "",
        model_prob: float | None = None,
        market_prob: float | None = None,
        raw_edge: float | None = None,
        fee_adjusted_edge: float | None = None,
        confidence: float | None = None,
        side: str | None = None,
        action: str | None = None,
        contracts: int | None = None,
        target_price: str | None = None,
        kelly_fraction: float | None = None,
        spread: float | None = None,
        liquidity: int | None = None,
        fill_rate: float | None = None,
        features: dict[str, float] | None = None,
        risk_limit: str | None = None,
        rank_score: float | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        cluster_id: str | None = None,
        time_to_expiry_h: float | None = None,
        open_order_count: int | None = None,
    ) -> None:
        """Emit a structured log entry for every candidate trade evaluation.

        This is the foundation for the stress testing framework: every
        market that reaches evaluate_market() gets logged, whether it
        was selected for trading or rejected at any gate.
        """
        log_fields: dict[str, Any] = {
            "ticker": ticker,
            "category": category,
            "status": status,  # "selected" | "rejected"
        }
        if cycle_id is not None:
            log_fields["cycle_id"] = cycle_id
        if reject_reason:
            log_fields["reject_reason"] = reject_reason
        if model_prob is not None:
            log_fields["model_prob"] = round(model_prob, 4)
        if market_prob is not None:
            log_fields["market_prob"] = round(market_prob, 4)
        if raw_edge is not None:
            log_fields["raw_edge"] = round(raw_edge, 4)
        if fee_adjusted_edge is not None:
            log_fields["fee_adjusted_edge"] = round(fee_adjusted_edge, 4)
        if confidence is not None:
            log_fields["confidence"] = round(confidence, 4)
        if side is not None:
            log_fields["side"] = side
        if action is not None:
            log_fields["action"] = action
        if contracts is not None:
            log_fields["contracts"] = contracts
        if target_price is not None:
            log_fields["target_price"] = target_price
        if kelly_fraction is not None:
            log_fields["kelly_fraction"] = round(kelly_fraction, 4)
        if spread is not None:
            log_fields["spread"] = round(spread, 4)
        if liquidity is not None:
            log_fields["liquidity"] = liquidity
        if fill_rate is not None:
            log_fields["fill_rate"] = round(fill_rate, 4)
        if risk_limit is not None:
            log_fields["risk_limit"] = risk_limit
        if rank_score is not None:
            log_fields["rank_score"] = round(rank_score, 6)
        if event_ticker:
            log_fields["event_ticker"] = event_ticker
        if series_ticker:
            log_fields["series_ticker"] = series_ticker
        if cluster_id:
            log_fields["cluster_id"] = cluster_id
        if time_to_expiry_h is not None:
            log_fields["time_to_expiry_h"] = round(time_to_expiry_h, 3)
        if open_order_count is not None:
            log_fields["open_order_count"] = open_order_count
        if features is not None:
            # Include key feature values for post-hoc analysis
            for k in (
                "mid_price", "weighted_mid_price", "orderbook_imbalance",
                "depth_ratio", "bid_ask_spread", "time_to_expiry",
                "sportsbook_win_prob", "pinnacle_win_prob", "moneyline_movement",
                "power_rating_edge", "home_field_advantage", "team_injury_impact",
                "ensemble_exceedance_prob", "ensemble_mean", "ensemble_spread",
                "model_disagreement", "forecast_horizon",
            ):
                if k in features:
                    log_fields[f"f_{k}"] = round(features[k], 4)

        logger.info("engine.candidate", **log_fields)

    @staticmethod
    def _candidate_rank_score(
        *,
        fee_adjusted_edge: float,
        confidence: float,
        fill_rate: float,
        spread: float | None,
    ) -> float:
        spread_penalty = max(0.1, 1.0 - max(spread or 0.0, 0.0))
        return fee_adjusted_edge * confidence * fill_rate * spread_penalty

    def _candidate_context_fields(
        self,
        ticker: str,
        *,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        active_cycle_id = cycle_id if cycle_id is not None else self._active_cycle_id
        market = self._market_cache.get(ticker)
        if market is None:
            return {
                "cycle_id": active_cycle_id,
                "cluster_id": f"ticker:{ticker}",
                "open_order_count": len(self._orders.get_open_orders()),
            }

        time_to_expiry_h = max(
            0.0,
            (market.close_time - datetime.now(timezone.utc)).total_seconds() / 3600.0,
        )
        cluster_id = (
            f"event:{market.event_ticker}"
            if market.event_ticker
            else (
                f"series:{market.series_ticker}"
                if market.series_ticker
                else f"ticker:{market.ticker}"
            )
        )
        return {
            "cycle_id": active_cycle_id,
            "event_ticker": market.event_ticker or None,
            "series_ticker": market.series_ticker or None,
            "cluster_id": cluster_id,
            "time_to_expiry_h": time_to_expiry_h,
            "open_order_count": len(self._orders.get_open_orders()),
        }

    async def evaluate_market(
        self,
        ticker: str,
        *,
        cycle_id: str | None = None,
    ) -> TradeDecision | None:
        """Run the full decision pipeline for a single market.

        Steps:
          1. Build FeatureContext from current market state + data
          2. Run feature pipeline -> feature_vector
          3. Run model.predict_proba(features) -> ModelPrediction
          4. Run edge_calculator.compute_edge(prob, orderbook) -> EdgeResult
          5. If actionable: run sizer.size() -> SizeResult
          6. Run risk_manager.pre_trade_check() -> RiskCheckResult
          7. If approved: return TradeDecision

        Returns None if the market is not tradeable or edge is insufficient.
        Every evaluation — selected or rejected — is logged via _log_candidate()
        for stress test analysis.
        """
        now = datetime.now(timezone.utc)
        category = self._market_categories.get(ticker, MarketCategory.UNKNOWN)
        cat_str = category.value
        candidate_ctx = self._candidate_context_fields(ticker, cycle_id=cycle_id)

        self._refresh_market_cache_from_shared_store(ticker, now)

        if any(order.ticker == ticker for order in self._orders.get_open_orders()):
            self._log_candidate(
                ticker=ticker,
                category=cat_str,
                status="rejected",
                reject_reason="open_order_exists",
                **candidate_ctx,
            )
            return None

        # 1. Get current orderbook (try WS first, fall back to REST)
        orderbook = self._ws.get_orderbook(ticker)
        if orderbook is None:
            orderbook = self._get_orderbook_from_shared_store(ticker, now)
        if orderbook is None:
            try:
                orderbook = await self._rest.get_orderbook(ticker)
            except Exception:
                pass
        if orderbook is None:
            # Synthesize minimal orderbook from market state bid/ask
            market = self._market_cache.get(ticker)
            if market is not None and market.yes_bid > 0 and market.yes_ask > 0:
                from moneygone.exchange.types import OrderbookLevel
                implied_no_bid = market.no_bid
                if implied_no_bid <= 0:
                    implied_no_bid = Decimal("1") - market.yes_ask
                orderbook = OrderbookSnapshot(
                    ticker=ticker,
                    yes_bids=(OrderbookLevel(price=market.yes_bid, contracts=Decimal("100")),),
                    no_bids=(
                        OrderbookLevel(price=implied_no_bid, contracts=Decimal("100")),
                    ),
                    seq=0,
                    timestamp=now,
                )
            else:
                self._log_candidate(
                    ticker=ticker, category=cat_str,
                    status="rejected", reject_reason="no_orderbook",
                    **candidate_ctx,
                )
                return None

        # Compute spread from orderbook
        market_obj = self._market_cache.get(ticker)
        ob_spread: float | None = None
        if market_obj is not None and market_obj.yes_bid > 0 and market_obj.yes_ask > 0:
            ob_spread = float(market_obj.yes_ask - market_obj.yes_bid)

        # 2. Determine market category and get the right model/pipeline
        provider = self._category_providers.get(category)
        sports_snapshot: dict[str, Any] | None = None
        features: dict[str, float] = {}

        if provider is not None:
            # Use category-specific model + pipeline + data provider
            context_data: dict[str, Any] = {}
            if provider.get_context_data is not None:
                context_data = await provider.get_context_data(
                    self._market_cache.get(ticker),
                ) or {}
                if not context_data:
                    self._log_candidate(
                        ticker=ticker, category=cat_str,
                        status="rejected", reject_reason="no_category_data",
                        **candidate_ctx,
                    )
                    return None
            # If get_context_data is None (baseline categories), proceed
            # with empty context_data — model uses orderbook features only.

            context = FeatureContext(
                ticker=ticker,
                observation_time=now,
                orderbook=orderbook,
                market_state=self._market_cache.get(ticker),
                sports_snapshot=context_data if category == MarketCategory.SPORTS else None,
                crypto_snapshot=context_data if category == MarketCategory.CRYPTO else None,
                weather_ensemble=context_data.get("ensemble") if category == MarketCategory.WEATHER else None,
                weather_threshold=context_data.get("threshold") if category == MarketCategory.WEATHER else None,
                weather_direction=context_data.get("direction") if category == MarketCategory.WEATHER else None,
                weather_location=context_data.get("location_name") if category == MarketCategory.WEATHER else None,
                weather_variable=context_data.get("weather_variable") if category == MarketCategory.WEATHER else None,
                store=self._store,
            )
            features = provider.pipeline.compute(context)
            if not features:
                self._log_candidate(
                    ticker=ticker, category=cat_str,
                    status="rejected", reject_reason="no_features",
                    **candidate_ctx,
                )
                return None
            prediction = provider.model.predict_proba(features)
        else:
            # Legacy path: sports-only via sports_snapshot_provider
            sports_snapshot = await self._get_sports_snapshot(ticker)
            if sports_snapshot is None:
                self._log_candidate(
                    ticker=ticker, category=cat_str,
                    status="rejected", reject_reason="no_data_provider",
                    **candidate_ctx,
                )
                return None

            context = FeatureContext(
                ticker=ticker,
                observation_time=now,
                orderbook=orderbook,
                market_state=self._market_cache.get(ticker),
                sports_snapshot=sports_snapshot,
                store=self._store,
            )
            features = self._pipeline.compute(context)
            if not features:
                self._log_candidate(
                    ticker=ticker, category=cat_str,
                    status="rejected", reject_reason="no_features",
                    **candidate_ctx,
                )
                return None
            prediction = self._model.predict_proba(features)

        self._queue_prediction_row(ticker, prediction)

        # Common fields for all subsequent log entries
        market_prob_est: float | None = None
        if orderbook.mid_price is not None:
            market_prob_est = float(orderbook.mid_price)
        elif market_obj is not None and market_obj.last_price > 0:
            market_prob_est = float(market_obj.last_price)
        elif orderbook.best_yes_bid is not None:
            market_prob_est = float(orderbook.best_yes_bid)

        # 3. Reject low-confidence predictions
        if prediction.confidence < 0.30:
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected", reject_reason="low_confidence",
                model_prob=prediction.probability,
                market_prob=market_prob_est,
                confidence=prediction.confidence,
                spread=ob_spread,
                features=features,
                **candidate_ctx,
            )
            return None

        # 5. Compute edge
        edge = self._edge_calc.compute_edge(
            prediction.probability,
            orderbook,
            is_maker=self._config.prefer_maker,
        )
        trade_model_prob = (
            prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
        )
        rank_score = self._candidate_rank_score(
            fee_adjusted_edge=edge.fee_adjusted_edge,
            confidence=prediction.confidence,
            fill_rate=edge.estimated_fill_rate,
            spread=ob_spread,
        )

        if not edge.is_actionable:
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected",
                reject_reason=(
                    f"edge_not_actionable:{edge.actionable_reason}"
                    if edge.actionable_reason
                    else "edge_not_actionable"
                ),
                model_prob=trade_model_prob,
                market_prob=edge.implied_probability,
                raw_edge=edge.raw_edge,
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                side=edge.side,
                spread=ob_spread,
                liquidity=edge.available_liquidity,
                features=features,
                rank_score=rank_score,
                **candidate_ctx,
            )
            return None

        if rank_score < getattr(self._config, "min_conviction_score", 0.0):
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected",
                reject_reason="conviction_below_threshold",
                model_prob=trade_model_prob,
                market_prob=edge.implied_probability,
                raw_edge=edge.raw_edge,
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                side=edge.side,
                spread=ob_spread,
                liquidity=edge.available_liquidity,
                fill_rate=edge.estimated_fill_rate,
                features=features,
                rank_score=rank_score,
                **candidate_ctx,
            )
            return None

        duplicate = await self._find_duplicate_exposure(
            ticker,
            edge.side,
            action=edge.action,
            sports_snapshot=sports_snapshot,
        )
        if duplicate is not None:
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected", reject_reason=f"duplicate_exposure:{duplicate['kind']}",
                model_prob=trade_model_prob,
                market_prob=edge.implied_probability,
                raw_edge=edge.raw_edge,
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                side=edge.side,
                spread=ob_spread,
                liquidity=edge.available_liquidity,
                features=features,
                rank_score=rank_score,
                **candidate_ctx,
            )
            return None

        # 6. Size the position
        bankroll = self._risk._portfolio.get_equity()
        existing_exposure = self._risk._portfolio.get_total_exposure()  # type: ignore[attr-defined]

        size = self._sizer.size(
            edge_result=edge,
            bankroll=bankroll,
            model_confidence=prediction.confidence,
            existing_exposure=existing_exposure,
        )

        if size.contracts <= 0:
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected", reject_reason=f"zero_size:{size.capped_by or 'kelly'}",
                model_prob=trade_model_prob,
                market_prob=edge.implied_probability,
                raw_edge=edge.raw_edge,
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                side=edge.side,
                action=edge.action,
                kelly_fraction=size.kelly_fraction,
                spread=ob_spread,
                liquidity=edge.available_liquidity,
                fill_rate=edge.estimated_fill_rate,
                features=features,
                rank_score=rank_score,
                **candidate_ctx,
            )
            return None

        # 7. Risk check
        proposed = ProposedTrade(
            ticker=ticker,
            category=self._market_cache[ticker].category if ticker in self._market_cache else "",
            side=edge.side,
            action=edge.action,
            contracts=size.contracts,
            price=edge.target_price,
        )

        risk_check = self._risk.pre_trade_check(proposed)

        # Apply adjusted size if risk limits partially approve
        actual_contracts = size.contracts
        if risk_check.adjusted_size is not None and risk_check.adjusted_size > 0:
            actual_contracts = risk_check.adjusted_size

        if not risk_check.approved:
            self._log_candidate(
                ticker=ticker, category=cat_str,
                status="rejected",
                reject_reason=(
                    f"risk_rejected:{risk_check.limit_triggered}"
                    if risk_check.limit_triggered
                    else "risk_rejected"
                ),
                model_prob=trade_model_prob,
                market_prob=edge.implied_probability,
                raw_edge=edge.raw_edge,
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                side=edge.side,
                action=edge.action,
                contracts=size.contracts,
                target_price=str(edge.target_price),
                kelly_fraction=size.kelly_fraction,
                spread=ob_spread,
                liquidity=edge.available_liquidity,
                fill_rate=edge.estimated_fill_rate,
                risk_limit=risk_check.limit_triggered,
                features=features,
                rank_score=rank_score,
                **candidate_ctx,
            )
            return None

        decision = TradeDecision(
            ticker=ticker,
            edge_result=edge,
            size_result=SizeResult(
                kelly_fraction=size.kelly_fraction,
                adjusted_fraction=size.adjusted_fraction,
                contracts=actual_contracts,
                dollar_risk=Decimal(actual_contracts) * edge.target_price,
                dollar_ev=size.dollar_ev,
                capped_by=size.capped_by,
            ),
            risk_check=risk_check,
            prediction=prediction,
            timestamp=now,
            cycle_id=cycle_id if cycle_id is not None else self._active_cycle_id,
            category=cat_str,
            rank_score=rank_score,
        )

        # Log as selected candidate with full metadata
        self._log_candidate(
            ticker=ticker, category=cat_str,
            status="selected",
            model_prob=trade_model_prob,
            market_prob=edge.implied_probability,
            raw_edge=edge.raw_edge,
            fee_adjusted_edge=edge.fee_adjusted_edge,
            confidence=prediction.confidence,
            side=edge.side,
            action=edge.action,
            contracts=actual_contracts,
            target_price=str(edge.target_price),
            kelly_fraction=size.kelly_fraction,
            spread=ob_spread,
            liquidity=edge.available_liquidity,
            fill_rate=edge.estimated_fill_rate,
            features=features,
            rank_score=rank_score,
            **candidate_ctx,
        )

        self._queue_feature_rows(ticker, now, features)

        return decision

    async def execute_decision(self, decision: TradeDecision) -> None:
        """Submit an order based on a trade decision.

        Uses the configured execution strategy to place the order.
        Records the fill with prediction context in the fill tracker.
        """
        # SAFETY: Block demo_only models from placing real orders.
        # Check both the model's demo_only flag (via category provider) AND
        # the model name as a belt-and-suspenders guard.
        if not self._demo_mode:
            model_name = decision.prediction.model_name
            provider = self._category_providers.get(
                self._market_categories.get(decision.ticker),  # type: ignore[arg-type]
            )
            is_demo_only = (
                model_name == "market_baseline"
                or (provider is not None and getattr(provider.model, "demo_only", False))
            )
            if is_demo_only:
                logger.error(
                    "engine.BLOCKED_demo_only_model_in_live",
                    ticker=decision.ticker,
                    model=model_name,
                    msg="Demo-only model has no edge — refusing to trade with real money",
                )
                return

        orderbook = self._ws.get_orderbook(decision.ticker)
        if orderbook is None:
            orderbook = self._get_orderbook_from_shared_store(
                decision.ticker,
                datetime.now(timezone.utc),
            )
        if orderbook is None:
            try:
                orderbook = await self._rest.get_orderbook(decision.ticker)
            except Exception:
                pass
        if orderbook is None:
            logger.warning(
                "engine.execute_no_orderbook",
                ticker=decision.ticker,
            )
            return

        # Record submission for fill-rate tracking
        self._fills.record_submission()

        try:
            order = await self._strategy.execute(
                decision.edge_result,
                decision.size_result,
                self._orders,
                orderbook,
            )
        except OrderError as exc:
            logger.error(
                "engine.order_execution_failed",
                ticker=decision.ticker,
                cycle_id=decision.cycle_id,
                category=decision.category,
                status_code=exc.status_code,
                subaccount=getattr(self._rest, "_subaccount", 0),
                error=str(exc),
            )
            return

        if order is not None:
            self._decision_context[order.order_id] = decision
            client_order_id = getattr(self._orders, "_order_client_order_ids", {}).get(order.order_id)
            if client_order_id:
                self._decision_context_by_client_order_id[client_order_id] = decision
            logger.info(
                "engine.order_executed",
                order_id=order.order_id,
                ticker=order.ticker,
                status=order.status.value,
                cycle_id=decision.cycle_id,
                category=decision.category,
            )

    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------

    async def _periodic_evaluation_loop(self) -> None:
        """Periodically re-evaluate all watched markets."""
        interval = self._config.evaluation_interval_seconds

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._refresh_market_universe()
                await self._maybe_reconcile_open_orders()
                await self._cancel_stale_open_orders()

                # Sync portfolio with exchange every 2 minutes for fresh cash balance
                if (datetime.now(timezone.utc) - self._last_portfolio_sync).total_seconds() >= 120:
                    try:
                        await self._risk._portfolio.sync_with_exchange(self._rest)
                        self._last_portfolio_sync = datetime.now(timezone.utc)
                        logger.info(
                            "engine.portfolio_resynced",
                            cash=str(self._risk._portfolio.cash),
                            positions=len(self._risk._portfolio.positions),
                        )
                    except Exception:
                        logger.warning("engine.portfolio_resync_failed", exc_info=True)

                # Prune settled/closed tickers every 10 minutes
                if (datetime.now(timezone.utc) - self._last_prune).total_seconds() >= 600:
                    self._prune_settled_tickers()

                cycle_selected = 0
                cycle_start = datetime.now(timezone.utc)
                self._active_cycle_id = cycle_start.isoformat()

                # Build list of tickers eligible for evaluation this cycle
                eligible_tickers: list[str] = []
                now = datetime.now(timezone.utc)
                for ticker in self._watched:
                    if not self._running:
                        break
                    if ticker in self._stale_rechecks_inflight:
                        continue
                    last = self._last_eval.get(ticker)
                    if last is not None and (now - last).total_seconds() < interval:
                        continue
                    self._last_eval[ticker] = now
                    eligible_tickers.append(ticker)

                cycle_evaluated = len(eligible_tickers)

                # Evaluate markets concurrently with bounded parallelism
                sem = asyncio.Semaphore(8)

                async def _eval_one(t: str) -> object | None:
                    async with sem:
                        if not self._running:
                            return None
                        try:
                            return await self.evaluate_market(t)
                        except Exception:
                            logger.exception("engine.parallel_eval_error", ticker=t)
                            return None

                results = await asyncio.gather(
                    *[_eval_one(t) for t in eligible_tickers],
                    return_exceptions=True,
                )

                # Execute decisions sequentially to preserve risk accounting
                for result in results:
                    if isinstance(result, BaseException):
                        logger.error("engine.gather_exception", error=str(result))
                        continue
                    if result is not None:
                        cycle_selected += 1
                        await self.execute_decision(result)

                # Per-cycle summary for stress test analysis
                if cycle_evaluated > 0:
                    self._flush_observability_buffers()
                    elapsed_s = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                    logger.info(
                        "engine.cycle_summary",
                        cycle_id=self._active_cycle_id,
                        evaluated=cycle_evaluated,
                        selected=cycle_selected,
                        rejected=cycle_evaluated - cycle_selected,
                        watched_total=len(self._watched),
                        elapsed_s=round(elapsed_s, 1),
                    )
                self._active_cycle_id = None

            except asyncio.CancelledError:
                return
            except Exception:
                self._active_cycle_id = None
                logger.exception("engine.periodic_eval_error")
                await asyncio.sleep(1.0)

    async def _stale_order_loop(self) -> None:
        """Cancel stale maker orders on wall-clock cadence, independent of eval speed."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                await self._cancel_stale_open_orders()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("engine.stale_order_loop_error")
                await asyncio.sleep(1.0)

    async def _shared_market_data_loop(self) -> None:
        """Continuously tail shared market-data tables into local hot caches."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                self._poll_shared_market_data_once()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("engine.shared_market_data_loop_error")
                await asyncio.sleep(1.0)

    async def _recheck_ticker_after_stale_cancel(self, ticker: str) -> None:
        """Re-evaluate a ticker immediately after its stale order is cancelled."""
        if not getattr(self, "_running", False):
            return
        inflight = getattr(self, "_stale_rechecks_inflight", None)
        if inflight is None:
            return
        if ticker in inflight:
            return
        watched = getattr(self, "_watched", [])
        market_cache = getattr(self, "_market_cache", {})
        if ticker not in watched or ticker not in market_cache:
            return

        recheck_cycle_id = f"stale-recheck:{datetime.now(timezone.utc).isoformat()}"
        inflight.add(ticker)
        self._last_eval[ticker] = datetime.now(timezone.utc)
        try:
            decision = await self.evaluate_market(ticker, cycle_id=recheck_cycle_id)
            if decision is not None:
                await self.execute_decision(decision)
        finally:
            inflight.discard(ticker)

    # ------------------------------------------------------------------
    # Ticker management
    # ------------------------------------------------------------------

    def add_ticker(self, ticker: str) -> None:
        """Add a ticker to the watch list."""
        if ticker not in self._watched:
            self._watched.append(ticker)
            logger.info("engine.ticker_added", ticker=ticker)

    def remove_ticker(self, ticker: str) -> None:
        """Remove a ticker from the watch list."""
        if ticker in self._watched:
            self._watched.remove(ticker)
            logger.info("engine.ticker_removed", ticker=ticker)

    def _prune_settled_tickers(self) -> int:
        """Remove settled/closed tickers from the watch list.

        Prevents unbounded growth of ``_watched`` over multi-day runs by
        dropping tickers whose market has closed (close_time in the past)
        or whose status is settled/closed.

        Returns the number of tickers pruned.
        """
        now = datetime.now(timezone.utc)
        to_remove: list[str] = []

        for ticker in self._watched:
            market = self._market_cache.get(ticker)
            if market is None:
                continue
            if market.status in (MarketStatus.SETTLED, MarketStatus.CLOSED):
                to_remove.append(ticker)
            elif market.close_time < now:
                to_remove.append(ticker)

        for ticker in to_remove:
            self._watched.remove(ticker)
            self._market_cache.pop(ticker, None)
            self._market_categories.pop(ticker, None)
            self._last_eval.pop(ticker, None)
            self._drop_decision_context_for_ticker(ticker)

        if to_remove:
            logger.info(
                "engine.pruned_settled_tickers",
                pruned=len(to_remove),
                watched_remaining=len(self._watched),
            )

        self._last_prune = now
        return len(to_remove)

    async def _refresh_market_universe(self, *, force: bool = False) -> None:
        """Refresh watched markets across all enabled categories.

        Reads from the shared discovery cache (written by market_data worker)
        instead of calling get_all_markets directly.  Falls back to REST if
        the cache is missing or stale (> 5 min).
        """
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._last_universe_refresh is not None
            and (now - self._last_universe_refresh).total_seconds() < 900
        ):
            return

        # Try reading from shared discovery cache first
        markets: list[Market] = []
        classified: list[tuple[Market, MarketCategory]] = []
        cache_used = False

        if self._discovery_cache_path is not None:
            classified, refreshed_at = MarketDiscoveryService.load_cache(
                self._discovery_cache_path,
            )
            if classified and refreshed_at is not None:
                age = (now - refreshed_at).total_seconds()
                if age < 300:  # Cache valid for 5 minutes
                    markets = [m for m, _ in classified]
                    cache_used = True
                    logger.debug("engine.using_discovery_cache", age_s=int(age), markets=len(markets))

        # Fallback: fetch directly via REST for all open markets
        if not markets:
            all_fetched = await self._rest.get_all_markets(
                limit=1_000,
                status="open",
                mve_filter="exclude",
            )
            markets = [m for m in all_fetched if m.status == MarketStatus.OPEN]
            classified = [(m, classify_market(m)) for m in markets]
            logger.debug("engine.direct_rest_fetch", fetched=len(all_fetched), open=len(markets))

        new_tickers: list[str] = []
        category_counts: dict[str, int] = {}

        # Reload sportsbook data from parquet if collector has updated it
        if self._sportsbook_parquet_path is not None and self._store is not None:
            try:
                p = self._sportsbook_parquet_path
                if p.exists():
                    mtime = p.stat().st_mtime
                    if mtime > self._last_parquet_mtime:
                        count = self._store.load_parquet_into_table(
                            "sportsbook_game_lines", p,
                        )
                        self._last_parquet_mtime = mtime
                        logger.info("engine.sportsbook_parquet_reloaded", rows=count)
            except Exception:
                logger.debug("engine.parquet_reload_failed", exc_info=True)

        # Phase 1: Sports markets via sports snapshot provider
        if self._sports is not None:
            matched = await self._sports.refresh(markets)
            for market in matched:
                self._market_cache[market.ticker] = market
                self._market_categories[market.ticker] = MarketCategory.SPORTS
                if market.ticker not in self._watched:
                    self._watched.append(market.ticker)
                    new_tickers.append(market.ticker)
            category_counts["sports"] = len(matched)

        # Phase 2: Other categories — use pre-classified data
        # Only watch markets with minimum liquidity (volume > 0, has bid/ask)
        skipped = 0
        for market, category in classified:
            if market.ticker in self._market_cache:
                continue  # Already matched as sports

            if category not in self._category_providers:
                continue

            # Skip illiquid/empty markets — no point evaluating them
            if market.volume < 10 or market.yes_bid <= 0 or market.yes_ask <= 0:
                skipped += 1
                continue

            self._market_cache[market.ticker] = market
            self._market_categories[market.ticker] = category
            if market.ticker not in self._watched:
                self._watched.append(market.ticker)
                new_tickers.append(market.ticker)
            category_counts[category.value] = category_counts.get(category.value, 0) + 1

        # NOTE: We do NOT subscribe to WS orderbook/ticker/trades for all
        # watched tickers — subscribing to 10K+ tickers is infeasible.
        # The eval loop fetches orderbooks via REST on-demand instead.

        self._last_universe_refresh = now
        logger.info(
            "engine.market_universe_refreshed",
            watched=len(self._watched),
            new=len(new_tickers),
            skipped_illiquid=skipped,
            categories=category_counts,
            cache_used=cache_used,
        )

    async def _cancel_stale_open_orders(self) -> int:
        """Cancel lingering resting orders so they don't block fresh decisions."""
        max_age_seconds = int(getattr(self._config, "max_order_staleness_seconds", 0) or 0)
        if max_age_seconds <= 0:
            return 0

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=max_age_seconds)
        stale_orders = []
        for order in self._orders.get_open_orders():
            created_time = order.created_time
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=timezone.utc)
            else:
                created_time = created_time.astimezone(timezone.utc)
            if created_time <= cutoff:
                stale_orders.append((order, int((now - created_time).total_seconds())))

        if not stale_orders:
            return 0

        logger.info(
            "engine.cancelling_stale_orders",
            count=len(stale_orders),
            max_age_seconds=max_age_seconds,
            oldest_age_seconds=max(age for _, age in stale_orders),
        )

        cancelled = 0
        cancelled_tickers: list[str] = []
        for order, age_seconds in stale_orders:
            try:
                confirmed_closed = await self._orders.cancel_order(order.order_id)
                cancelled += 1
                if confirmed_closed:
                    cancelled_tickers.append(order.ticker)
            except Exception:
                logger.warning(
                    "engine.cancel_stale_order_failed",
                    order_id=order.order_id,
                    ticker=order.ticker,
                    age_seconds=age_seconds,
                    exc_info=True,
                )

        if cancelled:
            logger.info(
                "engine.stale_orders_cancelled",
                count=cancelled,
                remaining_open=len(self._orders.get_open_orders()),
            )
            for ticker in dict.fromkeys(cancelled_tickers):
                await self._recheck_ticker_after_stale_cancel(ticker)
        return cancelled

    async def _get_sports_snapshot(self, ticker: str) -> dict[str, Any] | None:
        if self._sports is None:
            return None
        market = self._market_cache.get(ticker)
        if market is None:
            return None
        return await self._sports.get_snapshot(market)

    async def _find_duplicate_exposure(
        self,
        ticker: str,
        side: str,
        *,
        action: str = "buy",
        sports_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Block same-ticker resting orders and equivalent event exposure."""
        market = await self._get_market(ticker)
        binary_event_key = (
            await self._binary_event_key_for_market(market)
            if market is not None
            else None
        )
        event_cluster_key = (
            self._event_cluster_key_for_market(market)
            if market is not None
            else None
        )

        for order in self._orders.get_open_orders():
            if order.ticker == ticker:
                return {
                    "kind": "open_order_same_ticker",
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "action": order.action.value,
                }

        outcome_key = await self._outcome_key_for_trade(
            ticker,
            side,
            action=action,
            sports_snapshot=sports_snapshot,
        )
        if outcome_key is None and event_cluster_key is None:
            return None

        for order in self._orders.get_open_orders():
            if order.ticker == ticker:
                continue
            other_key = None
            if outcome_key is not None:
                other_key = await self._outcome_key_for_trade(
                    order.ticker,
                    order.side.value,
                    action=order.action.value,
                )
            if outcome_key is not None and other_key == outcome_key:
                return {
                    "kind": "equivalent_open_order",
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "action": order.action.value,
                }
            if binary_event_key is not None:
                other_binary_event_key = await self._binary_event_key_for_ticker(order.ticker)
                if other_binary_event_key == binary_event_key:
                    return {
                        "kind": "binary_event_open_order",
                        "ticker": order.ticker,
                        "side": order.side.value,
                        "action": order.action.value,
                    }
            if event_cluster_key is not None:
                other_event_cluster_key = await self._event_cluster_key_for_ticker(order.ticker)
                if other_event_cluster_key == event_cluster_key:
                    return {
                        "kind": "event_cluster_open_order",
                        "ticker": order.ticker,
                        "side": order.side.value,
                        "action": order.action.value,
                    }

        for decision in self._decision_context.values():
            if decision.ticker == ticker:
                continue
            if binary_event_key is not None:
                other_binary_event_key = await self._binary_event_key_for_ticker(decision.ticker)
                if other_binary_event_key == binary_event_key:
                    return {
                        "kind": "binary_event_pending_order",
                        "ticker": decision.ticker,
                        "category": getattr(decision, "category", ""),
                    }
            if event_cluster_key is not None:
                other_event_cluster_key = await self._event_cluster_key_for_ticker(decision.ticker)
                if other_event_cluster_key == event_cluster_key:
                    return {
                        "kind": "event_cluster_pending_order",
                        "ticker": decision.ticker,
                        "category": getattr(decision, "category", ""),
                    }

        for other_ticker, position in self._risk._portfolio.positions.items():
            if other_ticker == ticker:
                continue
            if outcome_key is not None and position.yes_count > 0:
                other_key = await self._outcome_key_for_trade(
                    other_ticker,
                    "yes",
                    action="buy",
                )
                if other_key == outcome_key:
                    return {
                        "kind": "equivalent_position",
                        "ticker": other_ticker,
                        "side": "yes",
                        "contracts": position.yes_count,
                    }
            if outcome_key is not None and position.no_count > 0:
                other_key = await self._outcome_key_for_trade(
                    other_ticker,
                    "no",
                    action="buy",
                )
                if other_key == outcome_key:
                    return {
                        "kind": "equivalent_position",
                        "ticker": other_ticker,
                        "side": "no",
                        "contracts": position.no_count,
                    }
            if binary_event_key is not None and (position.yes_count > 0 or position.no_count > 0):
                other_binary_event_key = await self._binary_event_key_for_ticker(other_ticker)
                if other_binary_event_key == binary_event_key:
                    return {
                        "kind": "binary_event_position",
                        "ticker": other_ticker,
                        "contracts": position.yes_count + position.no_count,
                    }
            if event_cluster_key is not None and (position.yes_count > 0 or position.no_count > 0):
                other_event_cluster_key = await self._event_cluster_key_for_ticker(other_ticker)
                if other_event_cluster_key == event_cluster_key:
                    return {
                        "kind": "event_cluster_position",
                        "ticker": other_ticker,
                        "contracts": position.yes_count + position.no_count,
                    }

        return None

    async def _outcome_key_for_trade(
        self,
        ticker: str,
        side: str,
        *,
        action: str = "buy",
        sports_snapshot: dict[str, Any] | None = None,
    ) -> tuple[str, str, str] | None:
        market = await self._get_market(ticker)
        if market is None:
            return None

        event_key = await self._event_outcome_key(
            market,
            side,
            action=action,
        )
        if event_key is not None:
            return event_key

        if sports_snapshot is None:
            sports_snapshot = await self._get_sports_snapshot(ticker)
        return self._sports_outcome_key(
            ticker,
            side,
            sports_snapshot,
            action=action,
        )

    async def _get_market(self, ticker: str) -> Market | None:
        market = self._market_cache.get(ticker)
        if market is not None:
            return market

        self._refresh_market_cache_from_shared_store(ticker, datetime.now(timezone.utc))
        market = self._market_cache.get(ticker)
        if market is not None:
            return market

        try:
            market = await self._rest.get_market(ticker)
        except Exception:
            logger.debug("engine.market_lookup_failed", ticker=ticker, exc_info=True)
            return None

        self._market_cache[ticker] = market
        return market

    async def _event_outcome_key(
        self,
        market: Market,
        side: str,
        *,
        action: str,
    ) -> tuple[str, str, str] | None:
        event_ticker = (market.event_ticker or "").strip()
        own_label = self._normalize_outcome_label(market.yes_sub_title)
        if not event_ticker or not own_label:
            return None

        event_markets = await self._get_event_markets(event_ticker)
        labels = {
            self._normalize_outcome_label(candidate.yes_sub_title)
            for candidate in event_markets
            if self._normalize_outcome_label(candidate.yes_sub_title)
        }
        if len(labels) != 2:
            return None

        wants_yes_outcome = (side == "yes" and action == "buy") or (
            side == "no" and action == "sell"
        )
        if wants_yes_outcome:
            return ("event_outcome", event_ticker, own_label)

        other_labels = sorted(label for label in labels if label != own_label)
        if len(other_labels) != 1:
            return None
        return ("event_outcome", event_ticker, other_labels[0])

    async def _binary_event_key_for_ticker(
        self,
        ticker: str,
    ) -> tuple[str, str] | None:
        market = await self._get_market(ticker)
        if market is None:
            return None
        return await self._binary_event_key_for_market(market)

    async def _event_cluster_key_for_ticker(
        self,
        ticker: str,
    ) -> tuple[str, str] | None:
        market = await self._get_market(ticker)
        if market is None:
            return None
        return self._event_cluster_key_for_market(market)

    async def _binary_event_key_for_market(
        self,
        market: Market,
    ) -> tuple[str, str] | None:
        event_ticker = (market.event_ticker or "").strip()
        own_label = self._normalize_outcome_label(market.yes_sub_title)
        if not event_ticker or not own_label:
            return None

        event_markets = await self._get_event_markets(event_ticker)
        labels = {
            self._normalize_outcome_label(candidate.yes_sub_title)
            for candidate in event_markets
            if self._normalize_outcome_label(candidate.yes_sub_title)
        }
        if len(labels) != 2 or own_label not in labels:
            return None
        return ("binary_event", event_ticker)

    @staticmethod
    def _event_cluster_key_for_market(
        market: Market,
    ) -> tuple[str, str] | None:
        event_ticker = (market.event_ticker or "").strip()
        if not event_ticker:
            return None
        return ("event_cluster", event_ticker)

    async def _get_event_markets(self, event_ticker: str) -> list[Market]:
        cached = {
            market.ticker: market
            for market in self._market_cache.values()
            if market.event_ticker == event_ticker
        }
        if len(cached) >= 2:
            return list(cached.values())

        try:
            fetched = await self._rest.get_markets(event_ticker=event_ticker)
        except Exception:
            logger.debug(
                "engine.event_markets_lookup_failed",
                event_ticker=event_ticker,
                exc_info=True,
            )
            return list(cached.values())

        for market in fetched:
            self._market_cache[market.ticker] = market
            cached[market.ticker] = market

        return list(cached.values())

    @staticmethod
    def _normalize_outcome_label(label: str | None) -> str:
        if not label:
            return ""
        return " ".join(label.strip().lower().split())

    def _sports_outcome_key(
        self,
        ticker: str,
        side: str,
        sports_snapshot: dict[str, Any] | None,
        *,
        action: str = "buy",
    ) -> tuple[str, str, str] | None:
        if sports_snapshot is None:
            return None

        market = self._market_cache.get(ticker)
        if market is None:
            return None

        event_id = str(sports_snapshot.get("event_id") or market.event_ticker or "")
        if not event_id:
            return None

        try:
            is_home_team = bool(int(sports_snapshot.get("is_home_team", 0)))
        except (TypeError, ValueError):
            return None

        longs_yes_side = (side == "yes" and action == "buy") or (
            side == "no" and action == "sell"
        )
        chosen_home = longs_yes_side == is_home_team
        winner = "home" if chosen_home else "away"
        return ("sports_game_winner", event_id, winner)

    async def _maybe_reconcile_open_orders(
        self,
        *,
        force: bool = False,
        min_interval_seconds: int = 60,
    ) -> None:
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._last_order_reconcile is not None
            and (now - self._last_order_reconcile).total_seconds() < min_interval_seconds
        ):
            return

        try:
            await self._orders.reconcile()
            self._last_order_reconcile = now
            logger.info(
                "engine.orders_reconciled",
                open_orders=self._orders.open_order_count,
            )
        except Exception:
            logger.warning("engine.orders_reconcile_failed", exc_info=True)

    def _resolve_decision_context(self, fill: Fill) -> TradeDecision | None:
        decision = None
        if fill.order_id is not None:
            decision = self._decision_context.get(fill.order_id)
        if decision is None and fill.client_order_id is not None:
            decision = self._decision_context_by_client_order_id.get(fill.client_order_id)

        # Clean up context for fully filled orders (no longer open)
        if fill.order_id is not None and not any(
            order.order_id == fill.order_id for order in self._orders.get_open_orders()
        ):
            self._decision_context.pop(fill.order_id, None)
            # Also clean up the client_order_id mapping
            if fill.client_order_id is not None:
                self._decision_context_by_client_order_id.pop(fill.client_order_id, None)

        # Bound the context dicts to prevent unbounded memory growth
        max_context_size = 500
        if len(self._decision_context) > max_context_size:
            open_ids = {o.order_id for o in self._orders.get_open_orders()}
            stale = [k for k in self._decision_context if k not in open_ids]
            for k in stale:
                self._decision_context.pop(k, None)
        if len(self._decision_context_by_client_order_id) > max_context_size:
            to_remove = list(self._decision_context_by_client_order_id.keys())[
                : len(self._decision_context_by_client_order_id) - max_context_size
            ]
            for k in to_remove:
                self._decision_context_by_client_order_id.pop(k, None)

        return decision

    def _fill_already_processed(self, fill_id: str) -> bool:
        if fill_id in self._seen_fill_ids:
            return True
        self._seen_fill_ids.add(fill_id)
        self._recent_fill_ids.append(fill_id)
        max_seen = 10_000
        while len(self._recent_fill_ids) > max_seen:
            old_fill_id = self._recent_fill_ids.popleft()
            self._seen_fill_ids.discard(old_fill_id)
        return False

    def _drop_decision_context_for_ticker(self, ticker: str) -> None:
        stale_order_ids = [
            order_id
            for order_id, decision in self._decision_context.items()
            if decision.ticker == ticker
        ]
        for order_id in stale_order_ids:
            self._decision_context.pop(order_id, None)

        stale_client_order_ids = [
            client_order_id
            for client_order_id, decision in self._decision_context_by_client_order_id.items()
            if decision.ticker == ticker
        ]
        for client_order_id in stale_client_order_ids:
            self._decision_context_by_client_order_id.pop(client_order_id, None)

    def _queue_prediction_row(
        self,
        ticker: str,
        prediction: ModelPrediction,
    ) -> None:
        if self._store is None:
            return
        self._prediction_rows.append(
            {
                "ticker": ticker,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
                "probability": prediction.probability,
                "raw_probability": prediction.raw_probability,
                "confidence": prediction.confidence,
                "prediction_time": prediction.prediction_time.isoformat(),
            }
        )

    def _queue_feature_rows(
        self,
        ticker: str,
        observation_time: datetime,
        features: dict[str, float],
    ) -> None:
        if self._store is None or not features:
            return
        self._feature_rows.extend(
            {
                "ticker": ticker,
                "observation_time": observation_time.isoformat(),
                "feature_name": name,
                "feature_value": value,
            }
            for name, value in features.items()
        )

    def _flush_observability_buffers(self) -> None:
        if self._store is None:
            self._prediction_rows.clear()
            self._feature_rows.clear()
            return

        if self._prediction_rows:
            try:
                self._store.insert_predictions(self._prediction_rows)
            except Exception:
                logger.warning("engine.prediction_persist_failed", exc_info=True)
            finally:
                self._prediction_rows.clear()

        if self._feature_rows:
            try:
                self._store.insert_features(self._feature_rows)
            except Exception:
                logger.warning("engine.feature_persist_failed", exc_info=True)
            finally:
                self._feature_rows.clear()

    def _initialize_shared_market_data_sync(self) -> None:
        """Seed cursors so shared-market polling tails only new rows."""
        if self._store is None:
            return
        if (
            getattr(self, "_shared_market_state_table", None) is not None
            and getattr(self, "_shared_market_cursor_time", None) is None
        ):
            self._shared_market_cursor_time = datetime.now(timezone.utc) - timedelta(seconds=2)
            self._shared_market_cursor_keys.clear()
        if (
            getattr(self, "_shared_orderbook_table", None) is not None
            and getattr(self, "_shared_orderbook_cursor_time", None) is None
        ):
            self._shared_orderbook_cursor_time = datetime.now(timezone.utc) - timedelta(seconds=2)
            self._shared_orderbook_cursor_keys.clear()

    def _poll_shared_market_data_once(self) -> None:
        """Poll attached market-data tables once and update local caches."""
        if self._store is None:
            return

        if (
            self._shared_market_state_table is not None
            and self._shared_market_cursor_time is not None
        ):
            try:
                rows = self._store.get_market_state_rows_since(
                    self._shared_market_cursor_time,
                    table=self._shared_market_state_table,
                    limit=5000,
                )
                self._ingest_shared_market_rows(rows)
            except Exception:
                logger.debug("engine.shared_market_state_poll_failed", exc_info=True)

        if (
            self._shared_orderbook_table is not None
            and self._shared_orderbook_cursor_time is not None
        ):
            try:
                rows = self._store.get_orderbook_rows_since(
                    self._shared_orderbook_cursor_time,
                    table=self._shared_orderbook_table,
                    limit=5000,
                )
                self._ingest_shared_orderbook_rows(rows)
            except Exception:
                logger.debug("engine.shared_orderbook_poll_failed", exc_info=True)

    def _ingest_shared_market_rows(self, rows: list[dict[str, Any]]) -> None:
        """Apply new shared market-state rows to the local cache."""
        latest_time = self._shared_market_cursor_time
        latest_keys = set(self._shared_market_cursor_keys)
        cursor_time = self._shared_market_cursor_time
        cursor_keys = self._shared_market_cursor_keys

        for row in rows:
            ingested_at = self._parse_timestamp(row.get("ingested_at"))
            if ingested_at is None:
                continue
            row_key = (
                f"{row.get('ticker', '')}|"
                f"{row.get('snapshot_time', '')}|"
                f"{row.get('last_price', '')}"
            )
            if cursor_time is not None:
                if ingested_at < cursor_time:
                    continue
                if ingested_at == cursor_time and row_key in cursor_keys:
                    continue

            market = self._merge_market_update(
                str(row.get("ticker", "")),
                row,
                self._parse_timestamp(row.get("snapshot_time"), fallback=ingested_at),
            )
            if market is not None:
                self._market_cache[market.ticker] = market

            if latest_time is None or ingested_at > latest_time:
                latest_time = ingested_at
                latest_keys = {row_key}
            elif ingested_at == latest_time:
                latest_keys.add(row_key)

        if latest_time is not None:
            self._shared_market_cursor_time = latest_time
            self._shared_market_cursor_keys = latest_keys

    def _ingest_shared_orderbook_rows(self, rows: list[dict[str, Any]]) -> None:
        """Apply new shared orderbook rows to the local hot cache."""
        latest_time = self._shared_orderbook_cursor_time
        latest_keys = set(self._shared_orderbook_cursor_keys)
        cursor_time = self._shared_orderbook_cursor_time
        cursor_keys = self._shared_orderbook_cursor_keys

        for row in rows:
            ingested_at = self._parse_timestamp(row.get("ingested_at"))
            if ingested_at is None:
                continue
            row_key = (
                f"{row.get('ticker', '')}|{row.get('seq', 0)}|"
                f"{row.get('snapshot_time', '')}"
            )
            if cursor_time is not None:
                if ingested_at < cursor_time:
                    continue
                if ingested_at == cursor_time and row_key in cursor_keys:
                    continue

            snapshot = self._build_orderbook_from_shared_row(
                str(row.get("ticker", "")),
                row,
                fallback_time=ingested_at,
            )
            if snapshot is not None:
                self._shared_orderbook_cache[snapshot.ticker] = snapshot

            if latest_time is None or ingested_at > latest_time:
                latest_time = ingested_at
                latest_keys = {row_key}
            elif ingested_at == latest_time:
                latest_keys.add(row_key)

        if latest_time is not None:
            self._shared_orderbook_cursor_time = latest_time
            self._shared_orderbook_cursor_keys = latest_keys

    def _refresh_market_cache_from_shared_store(
        self,
        ticker: str,
        as_of: datetime,
    ) -> None:
        if self._store is None or self._shared_market_state_table is None:
            return
        try:
            row = self._store.get_market_state_at(
                ticker,
                as_of,
                table=self._shared_market_state_table,
            )
        except Exception:
            logger.debug("engine.shared_market_state_lookup_failed", ticker=ticker, exc_info=True)
            return
        if row is None:
            return
        market = self._merge_market_update(ticker, row, as_of)
        if market is not None:
            self._market_cache[ticker] = market

    def _get_orderbook_from_shared_store(
        self,
        ticker: str,
        as_of: datetime,
    ) -> OrderbookSnapshot | None:
        cached = getattr(self, "_shared_orderbook_cache", {}).get(ticker)
        if cached is not None:
            if (as_of - cached.timestamp).total_seconds() <= 5.0:
                return cached
        if self._store is None or getattr(self, "_shared_orderbook_table", None) is None:
            return None
        try:
            row = self._store.get_orderbook_at(
                ticker,
                as_of,
                table=self._shared_orderbook_table,
            )
        except Exception:
            logger.debug("engine.shared_orderbook_lookup_failed", ticker=ticker, exc_info=True)
            return None
        if row is None:
            return None
        snapshot = self._build_orderbook_from_shared_row(
            ticker,
            row,
            fallback_time=as_of,
        )
        if snapshot is None:
            return None
        if (as_of - snapshot.timestamp).total_seconds() > 5.0:
            return None
        return snapshot

    def _build_orderbook_from_shared_row(
        self,
        ticker: str,
        row: dict[str, Any],
        *,
        fallback_time: datetime,
    ) -> OrderbookSnapshot | None:
        snapshot_time = self._parse_timestamp(row.get("snapshot_time"), fallback=fallback_time)
        if snapshot_time is None:
            snapshot_time = fallback_time

        def _parse_levels(raw_levels: Any) -> tuple[Any, ...]:
            from moneygone.exchange.types import OrderbookLevel

            levels = []
            for level in raw_levels or []:
                if isinstance(level, dict):
                    price = level.get("price")
                    contracts = level.get("contracts")
                else:
                    price, contracts = level
                levels.append(
                    OrderbookLevel(
                        price=Decimal(str(price)),
                        contracts=Decimal(str(contracts)),
                    )
                )
            return tuple(sorted(levels, key=lambda lvl: lvl.price))

        return OrderbookSnapshot(
            ticker=ticker,
            yes_bids=_parse_levels(row.get("yes_levels")),
            no_bids=_parse_levels(row.get("no_levels")),
            seq=int(row.get("seq", 0) or 0),
            timestamp=snapshot_time,
        )

    def _merge_market_update(
        self,
        ticker: str,
        data: dict[str, Any],
        timestamp: datetime | None,
    ) -> Market | None:
        existing = self._market_cache.get(ticker)
        close_time = self._parse_timestamp(
            data.get("close_time"),
            fallback=existing.close_time if existing is not None else None,
        )
        if close_time is None:
            close_time = timestamp or datetime.now(timezone.utc)

        status_value = data.get("status", existing.status.value if existing is not None else "open")
        try:
            status = MarketStatus(status_value)
        except ValueError:
            status = existing.status if existing is not None else MarketStatus.OPEN

        try:
            result_value = data.get(
                "result",
                existing.result.value if existing is not None else "",
            )
            try:
                result = MarketResult(result_value) if result_value else MarketResult.NOT_SETTLED
            except ValueError:
                result = existing.result if existing is not None else MarketResult.NOT_SETTLED

            return Market(
                ticker=ticker,
                event_ticker=data.get("event_ticker", existing.event_ticker if existing is not None else ""),
                series_ticker=data.get("series_ticker", existing.series_ticker if existing is not None else ""),
                title=data.get("title", existing.title if existing is not None else ""),
                status=status,
                yes_bid=self._decimal_field(data, "yes_bid_dollars", "yes_bid", fallback=existing.yes_bid if existing is not None else _ZERO),
                yes_ask=self._decimal_field(data, "yes_ask_dollars", "yes_ask", fallback=existing.yes_ask if existing is not None else _ZERO),
                last_price=self._decimal_field(
                    data,
                    "last_price_dollars",
                    "last_price",
                    fallback=existing.last_price if existing is not None else _ZERO,
                ),
                volume=int(data.get("volume", existing.volume if existing is not None else 0) or 0),
                open_interest=int(
                    data.get("open_interest", existing.open_interest if existing is not None else 0) or 0
                ),
                close_time=close_time,
                result=result,
                category=data.get("category", existing.category if existing is not None else ""),
                subtitle=data.get("subtitle", existing.subtitle if existing is not None else ""),
                yes_sub_title=data.get("yes_sub_title", existing.yes_sub_title if existing is not None else ""),
                no_sub_title=data.get("no_sub_title", existing.no_sub_title if existing is not None else ""),
                created_time=self._parse_timestamp(
                    data.get("created_time"),
                    fallback=existing.created_time if existing is not None else None,
                ),
                open_time=self._parse_timestamp(
                    data.get("open_time"),
                    fallback=existing.open_time if existing is not None else None,
                ),
                previous_price=self._decimal_field(
                    data,
                    "previous_price_dollars",
                    "previous_price",
                    fallback=existing.previous_price if existing is not None else _ZERO,
                ),
                liquidity_dollars=self._decimal_field(
                    data,
                    "liquidity_dollars",
                    "liquidity",
                    fallback=existing.liquidity_dollars if existing is not None else _ZERO,
                ),
                strike_type=str(data.get("strike_type", existing.strike_type if existing is not None else "") or ""),
                floor_strike=(
                    self._decimal_field(
                        data,
                        "floor_strike",
                        "floor_strike",
                        fallback=existing.floor_strike if existing is not None and existing.floor_strike is not None else _ZERO,
                    )
                    if data.get("floor_strike") is not None or (existing is not None and existing.floor_strike is not None)
                    else None
                ),
                cap_strike=(
                    self._decimal_field(
                        data,
                        "cap_strike",
                        "cap_strike",
                        fallback=existing.cap_strike if existing is not None and existing.cap_strike is not None else _ZERO,
                    )
                    if data.get("cap_strike") is not None or (existing is not None and existing.cap_strike is not None)
                    else None
                ),
                mve_selected_legs=tuple(
                    data.get(
                        "mve_selected_legs",
                        existing.mve_selected_legs if existing is not None else (),
                    )
                    or ()
                ),
            )
        except Exception:
            logger.warning("engine.market_cache_update_failed", ticker=ticker, exc_info=True)
            return existing

    @staticmethod
    def _decimal_field(
        data: dict[str, Any],
        primary: str,
        secondary: str,
        *,
        fallback: Decimal,
    ) -> Decimal:
        value = data.get(primary, data.get(secondary, fallback))
        return Decimal(str(value))

    @staticmethod
    def _parse_timestamp(
        value: Any,
        *,
        fallback: datetime | None = None,
    ) -> datetime | None:
        if value is None:
            return fallback
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _market_to_row(market: Market) -> dict[str, Any]:
        return {
            "ticker": market.ticker,
            "event_ticker": market.event_ticker,
            "title": market.title,
            "status": market.status.value,
            "yes_bid": float(market.yes_bid),
            "yes_ask": float(market.yes_ask),
            "last_price": float(market.last_price),
            "volume": market.volume,
            "open_interest": market.open_interest,
            "close_time": market.close_time.isoformat(),
            "result": market.result.value,
            "category": market.category,
        }

    @staticmethod
    def _orderbook_to_row(orderbook: OrderbookSnapshot) -> dict[str, Any]:
        return {
            "ticker": orderbook.ticker,
            "yes_levels": [
                [float(level.price), float(level.contracts)]
                for level in orderbook.yes_bids
            ],
            "no_levels": [
                [float(level.price), float(level.contracts)]
                for level in orderbook.no_bids
            ],
            "seq": orderbook.seq,
            "snapshot_time": orderbook.timestamp.isoformat(),
        }
