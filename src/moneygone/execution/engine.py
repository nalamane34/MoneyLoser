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
    ) -> None:
        self._rest = rest_client
        self._ws = ws_client
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

        self._running = False
        self._eval_task: asyncio.Task[None] | None = None
        self._market_cache: dict[str, Market] = {}
        self._market_categories: dict[str, MarketCategory] = {}
        self._last_eval: dict[str, datetime] = {}
        self._decision_context: dict[str, TradeDecision] = {}
        self._decision_context_by_client_order_id: dict[str, TradeDecision] = {}
        self._last_universe_refresh: datetime | None = None

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
        self._ws.set_on_event(self.on_event)
        await self._refresh_market_universe(force=True)

        # Connect WebSocket and subscribe to channels
        await self._ws.connect()
        await self._ws.wait_connected()

        if self._watched:
            await self._ws.subscribe_orderbook(self._watched)
            await self._ws.subscribe_ticker(self._watched)
            await self._ws.subscribe_trades(self._watched)
        await self._ws.subscribe_fills()

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
        fill = Fill(
            trade_id=data.get("trade_id", ""),
            ticker=data.get("market_ticker", data.get("ticker", "")),
            side=Side(data.get("side", "yes")),
            action=Action(data.get("action", "buy")),
            count=int(data.get("count", 0)),
            price=Decimal(str(data.get("yes_price_dollars", data.get("yes_price", 0)))),
            is_taker=bool(data.get("is_taker", False)),
            created_time=event.timestamp or datetime.now(timezone.utc),
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
        )

        # Update order manager
        self._orders.on_fill(fill)

        # Update risk state (portfolio, drawdown, etc.)
        self._risk.post_trade_update(fill)
        decision = self._resolve_decision_context(fill)
        if decision is not None:
            self._fills.on_fill(fill, decision.prediction, decision.edge_result)

        logger.info(
            "engine.fill_received",
            trade_id=fill.trade_id,
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
        """
        now = datetime.now(timezone.utc)

        # 1. Get current orderbook
        orderbook = self._ws.get_orderbook(ticker)
        if orderbook is None:
            logger.debug("engine.no_orderbook", ticker=ticker)
            return None

        # 2. Determine market category and get the right model/pipeline
        category = self._market_categories.get(ticker, MarketCategory.UNKNOWN)
        provider = self._category_providers.get(category)

        if provider is not None:
            # Use category-specific model + pipeline + data provider
            context_data: dict[str, Any] = {}
            if provider.get_context_data is not None:
                context_data = await provider.get_context_data(
                    self._market_cache.get(ticker),
                ) or {}
            if not context_data:
                logger.debug(
                    "engine.no_category_data",
                    ticker=ticker,
                    category=category.value,
                )
                return None

            context = FeatureContext(
                ticker=ticker,
                observation_time=now,
                orderbook=orderbook,
                market_state=self._market_cache.get(ticker),
                sports_snapshot=context_data if category == MarketCategory.SPORTS else None,
                crypto_snapshot=context_data if category == MarketCategory.CRYPTO else None,
                weather_ensemble=context_data.get("ensemble") if category == MarketCategory.WEATHER else None,
                store=self._store,
            )
            features = provider.pipeline.compute(context)
            if not features:
                logger.debug("engine.no_features", ticker=ticker, category=category.value)
                return None
            prediction = provider.model.predict_proba(features)
        else:
            # Legacy path: sports-only via sports_snapshot_provider
            sports_snapshot = await self._get_sports_snapshot(ticker)
            if sports_snapshot is None:
                logger.debug(
                    "engine.no_data_for_category",
                    ticker=ticker,
                    category=category.value,
                    msg="No provider and no sports snapshot — skipping",
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
                logger.debug("engine.no_features", ticker=ticker)
                return None
            prediction = self._model.predict_proba(features)

        # 3. Reject low-confidence predictions
        if prediction.confidence < 0.30:
            logger.debug(
                "engine.low_confidence",
                ticker=ticker,
                confidence=prediction.confidence,
                msg="Model confidence too low to act",
            )
            return None

        # 5. Compute edge
        edge = self._edge_calc.compute_edge(
            prediction.probability,
            orderbook,
            is_maker=self._config.prefer_maker,
        )

        if not edge.is_actionable:
            logger.debug(
                "engine.edge_not_actionable",
                ticker=ticker,
                net_edge=edge.fee_adjusted_edge,
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
            logger.debug(
                "engine.zero_size",
                ticker=ticker,
                capped_by=size.capped_by,
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
            logger.info(
                "engine.risk_rejected",
                ticker=ticker,
                limit=risk_check.limit_triggered,
                reason=risk_check.rejection_reason,
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
        )

        logger.info(
            "engine.trade_decision",
            ticker=ticker,
            side=edge.side,
            contracts=actual_contracts,
            net_edge=round(edge.fee_adjusted_edge, 4),
            model_prob=round(prediction.probability, 4),
            kelly=round(size.kelly_fraction, 4),
        )

        return decision

    async def execute_decision(self, decision: TradeDecision) -> None:
        """Submit an order based on a trade decision.

        Uses the configured execution strategy to place the order.
        Records the fill with prediction context in the fill tracker.
        """
        orderbook = self._ws.get_orderbook(decision.ticker)
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

                    decision = await self.evaluate_market(ticker)
                    if decision is not None:
                        await self.execute_decision(decision)

            except asyncio.CancelledError:
                return
            except Exception:
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

        # Fallback: fetch directly via REST
        if not markets:
            markets = await self._rest.get_all_markets(status="open", limit=1000)
            classified = [(m, classify_market(m)) for m in markets]
            logger.debug("engine.direct_rest_fetch", markets=len(markets))

        new_tickers: list[str] = []
        category_counts: dict[str, int] = {}

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

            if category == MarketCategory.UNKNOWN:
                continue
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

        if self._ws.is_connected and new_tickers:
            await self._ws.subscribe_orderbook(new_tickers)
            await self._ws.subscribe_ticker(new_tickers)
            await self._ws.subscribe_trades(new_tickers)

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
