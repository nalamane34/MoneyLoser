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
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from moneygone.config import ExecutionConfig
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.market_discovery import MarketCategory, classify_market, MarketDiscoveryService
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
        self._last_parquet_mtime: float = 0.0

        self._running = False
        self._eval_task: asyncio.Task[None] | None = None
        self._market_cache: dict[str, Market] = {}
        self._market_categories: dict[str, MarketCategory] = {}
        self._last_eval: dict[str, datetime] = {}
        self._decision_context: dict[str, TradeDecision] = {}
        self._decision_context_by_client_order_id: dict[str, TradeDecision] = {}
        self._last_universe_refresh: datetime | None = None
        self._last_order_reconcile: datetime | None = None
        self._active_cycle_id: str | None = None

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
            logger.info(
                "engine.portfolio_synced",
                cash=str(self._risk._portfolio.cash),
                positions=len(self._risk._portfolio.positions),
            )
        except Exception:
            logger.warning("engine.portfolio_sync_failed", exc_info=True)

        await self._maybe_reconcile_open_orders(force=True)

        self._ws.set_on_event(self.on_event)
        await self._refresh_market_universe(force=True)

        # Connect WebSocket and subscribe to essential channels only.
        # Subscribing to orderbook/ticker/trades for 10K+ tickers hangs
        # the connection — we fetch orderbooks on-demand via REST instead.
        await self._ws.connect()
        await self._ws.wait_connected()
        await self._ws.subscribe_fills()
        await self._ws.subscribe_positions()

        # Launch periodic evaluation
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

        # Cancel all open orders
        try:
            cancelled = await self._orders.cancel_all()
            logger.info("engine.orders_cancelled", count=cancelled)
        except Exception:
            logger.warning("engine.cancel_all_failed", exc_info=True)

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
            price=str(fill.price),
        )

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

    def _candidate_context_fields(self, ticker: str) -> dict[str, Any]:
        market = self._market_cache.get(ticker)
        if market is None:
            return {
                "cycle_id": self._active_cycle_id,
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
            "cycle_id": self._active_cycle_id,
            "event_ticker": market.event_ticker or None,
            "series_ticker": market.series_ticker or None,
            "cluster_id": cluster_id,
            "time_to_expiry_h": time_to_expiry_h,
            "open_order_count": len(self._orders.get_open_orders()),
        }

    async def evaluate_market(self, ticker: str) -> TradeDecision | None:
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
        candidate_ctx = self._candidate_context_fields(ticker)

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
            try:
                orderbook = await self._rest.get_orderbook(ticker)
            except Exception:
                pass
        if orderbook is None:
            # Synthesize minimal orderbook from market state bid/ask
            market = self._market_cache.get(ticker)
            if market is not None and market.yes_bid > 0 and market.yes_ask > 0:
                from moneygone.exchange.types import OrderbookLevel
                orderbook = OrderbookSnapshot(
                    ticker=ticker,
                    yes_bids=(OrderbookLevel(price=market.yes_bid, contracts=Decimal("100")),),
                    no_bids=(),
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

        if not edge.is_actionable:
            trade_model_prob = (
                prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
            )
            rank_score = self._candidate_rank_score(
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                fill_rate=edge.estimated_fill_rate,
                spread=ob_spread,
            )
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

        duplicate = await self._find_duplicate_exposure(
            ticker,
            edge.side,
            action=edge.action,
            sports_snapshot=sports_snapshot,
        )
        if duplicate is not None:
            trade_model_prob = (
                prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
            )
            rank_score = self._candidate_rank_score(
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                fill_rate=edge.estimated_fill_rate,
                spread=ob_spread,
            )
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
            trade_model_prob = (
                prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
            )
            rank_score = self._candidate_rank_score(
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                fill_rate=edge.estimated_fill_rate,
                spread=ob_spread,
            )
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
            trade_model_prob = (
                prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
            )
            rank_score = self._candidate_rank_score(
                fee_adjusted_edge=edge.fee_adjusted_edge,
                confidence=prediction.confidence,
                fill_rate=edge.estimated_fill_rate,
                spread=ob_spread,
            )
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
            cycle_id=self._active_cycle_id,
            category=cat_str,
        )

        # Log as selected candidate with full metadata
        trade_model_prob = (
            prediction.probability if edge.side == "yes" else 1.0 - prediction.probability
        )
        rank_score = self._candidate_rank_score(
            fee_adjusted_edge=edge.fee_adjusted_edge,
            confidence=prediction.confidence,
            fill_rate=edge.estimated_fill_rate,
            spread=ob_spread,
        )
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

        return decision

    async def execute_decision(self, decision: TradeDecision) -> None:
        """Submit an order based on a trade decision.

        Uses the configured execution strategy to place the order.
        Records the fill with prediction context in the fill tracker.
        """
        # SAFETY: Block demo_only models from placing real orders
        if not self._demo_mode and decision.prediction.model_name == "market_baseline":
            logger.error(
                "engine.BLOCKED_demo_only_model_in_live",
                ticker=decision.ticker,
                model=decision.prediction.model_name,
                msg="MarketBaselineModel has no edge — refusing to trade with real money",
            )
            return

        orderbook = self._ws.get_orderbook(decision.ticker)
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

        order = await self._strategy.execute(
            decision.edge_result,
            decision.size_result,
            self._orders,
            orderbook,
        )

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

                cycle_evaluated = 0
                cycle_selected = 0
                cycle_start = datetime.now(timezone.utc)
                self._active_cycle_id = cycle_start.isoformat()

                for ticker in self._watched:
                    if not self._running:
                        break

                    now = datetime.now(timezone.utc)
                    last = self._last_eval.get(ticker)

                    # Skip if recently evaluated
                    if last is not None:
                        elapsed = (now - last).total_seconds()
                        if elapsed < interval:
                            continue

                    self._last_eval[ticker] = now
                    cycle_evaluated += 1

                    decision = await self.evaluate_market(ticker)
                    if decision is not None:
                        cycle_selected += 1
                        await self.execute_decision(decision)

                # Per-cycle summary for stress test analysis
                if cycle_evaluated > 0:
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
        if outcome_key is None:
            return None

        for order in self._orders.get_open_orders():
            if order.ticker == ticker:
                continue
            other_key = await self._outcome_key_for_trade(
                order.ticker,
                order.side.value,
                action=order.action.value,
            )
            if other_key == outcome_key:
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

        for decision in self._decision_context.values():
            if decision.ticker == ticker:
                continue
            if binary_event_key is None:
                continue
            other_binary_event_key = await self._binary_event_key_for_ticker(decision.ticker)
            if other_binary_event_key == binary_event_key:
                return {
                    "kind": "binary_event_pending_order",
                    "ticker": decision.ticker,
                    "category": getattr(decision, "category", ""),
                }

        for other_ticker, position in self._risk._portfolio.positions.items():
            if other_ticker == ticker:
                continue
            if position.yes_count > 0:
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
            if position.no_count > 0:
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
