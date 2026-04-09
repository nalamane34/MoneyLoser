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

Also runs periodic re-evaluation of watched markets on a timer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from moneygone.config import ExecutionConfig
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import (
    Action,
    Fill,
    Market,
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

        self._running = False
        self._eval_task: asyncio.Task[None] | None = None
        self._market_cache: dict[str, Market] = {}
        self._last_eval: dict[str, datetime] = {}

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

        # Connect WebSocket and subscribe to channels
        self._ws = self._ws  # ws_client should have on_event set
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
            ticker=data.get("ticker", ""),
            side=Side(data.get("side", "yes")),
            action=Action(data.get("action", "buy")),
            count=int(data.get("count", 0)),
            price=Decimal(str(data.get("yes_price_dollars", data.get("yes_price", 0)))),
            is_taker=bool(data.get("is_taker", False)),
            created_time=datetime.now(timezone.utc),
        )

        # Update order manager
        self._orders.on_fill(fill)

        # Update risk state (portfolio, drawdown, etc.)
        self._risk.post_trade_update(fill)

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
        if ticker:
            logger.debug("engine.ticker_update", ticker=ticker)

    async def _handle_orderbook_event(self, event: WSEvent) -> None:
        """Process an orderbook snapshot or delta from the WebSocket."""
        # Orderbook is maintained internally by ws_client
        pass

    async def _handle_trade_event(self, event: WSEvent) -> None:
        """Process a public trade event."""
        pass

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

        # 2. Build feature context
        context = FeatureContext(
            ticker=ticker,
            observation_time=now,
            orderbook=orderbook,
            market_state=self._market_cache.get(ticker),
        )

        # 3. Run feature pipeline
        features = self._pipeline.compute(context)
        if not features:
            logger.debug("engine.no_features", ticker=ticker)
            return None

        # 4. Run model
        prediction = self._model.predict_proba(features)

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
