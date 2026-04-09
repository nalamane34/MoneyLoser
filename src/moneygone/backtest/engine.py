"""Backtest engine: replays historical events through the live pipeline.

Drives the same decision pipeline as the live :class:`ExecutionEngine`
but uses simulated fills against historical orderbook snapshots instead
of real exchange execution.  Enforces temporal fencing via
:class:`LeakageGuard` to prevent lookahead bias.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import structlog

from moneygone.backtest.data_loader import (
    EventType,
    HistoricalDataLoader,
    HistoricalEvent,
)
from moneygone.backtest.guards import LeakageGuard, LookaheadError, TimeFencedStore
from moneygone.backtest.results import BacktestResult
from moneygone.backtest.sim_exchange import SimulatedExchange
from moneygone.config import BacktestConfig, RiskConfig
from moneygone.exchange.types import (
    Action,
    Market,
    MarketResult,
    MarketStatus,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderRequest,
    Side,
    TimeInForce,
)
from moneygone.execution.simulator import FillSimulator
from moneygone.features.base import FeatureContext
from moneygone.features.pipeline import FeaturePipeline
from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.signals.edge import EdgeCalculator, EdgeResult
from moneygone.signals.fees import KalshiFeeCalculator
from moneygone.sizing.kelly import KellySizer, SizeResult
from moneygone.sizing.risk_limits import PortfolioState, ProposedTrade, RiskCheckResult, RiskLimits

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


class BacktestEngine:
    """Replays historical events through the live decision pipeline.

    At each event, updates simulated market state and (if applicable)
    evaluates the market using the same pipeline as live trading:
    features -> model -> edge -> sizing -> risk check -> simulated fill.

    Parameters
    ----------
    data_loader:
        Historical data loader.
    feature_pipeline:
        Feature computation pipeline.
    model:
        Probability prediction model.
    edge_calculator:
        Edge computation engine.
    sizer:
        Kelly criterion position sizer.
    risk_limits:
        Pre-trade risk limit checker.
    fill_simulator:
        Fill simulation model for backtesting.
    leakage_guard:
        Temporal constraint validator.
    config:
        Backtest configuration.
    risk_config:
        Risk configuration for limit thresholds.
    fee_calculator:
        Fee calculator.
    evaluation_on_orderbook:
        If True, evaluate markets on every orderbook event.
        If False, only evaluate on tick events.
    progress_interval:
        Log progress every N events.
    """

    def __init__(
        self,
        data_loader: HistoricalDataLoader,
        feature_pipeline: FeaturePipeline,
        model: ProbabilityModel,
        edge_calculator: EdgeCalculator,
        sizer: KellySizer,
        risk_limits: RiskLimits,
        fill_simulator: FillSimulator,
        leakage_guard: LeakageGuard,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
        fee_calculator: KalshiFeeCalculator | None = None,
        evaluation_on_orderbook: bool = True,
        progress_interval: int = 1000,
    ) -> None:
        self._loader = data_loader
        self._pipeline = feature_pipeline
        self._model = model
        self._edge_calc = edge_calculator
        self._sizer = sizer
        self._risk_limits = risk_limits
        self._fill_sim = fill_simulator
        self._guard = leakage_guard
        self._config = config or BacktestConfig()
        self._risk_config = risk_config or RiskConfig()
        self._fees = fee_calculator or KalshiFeeCalculator()
        self._eval_on_orderbook = evaluation_on_orderbook
        self._progress_interval = progress_interval

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_bankroll: Decimal | None = None,
        tickers: list[str] | None = None,
    ) -> BacktestResult:
        """Run the backtest over the specified date range.

        Parameters
        ----------
        start_date:
            Start of the backtest period.
        end_date:
            End of the backtest period.
        initial_bankroll:
            Starting cash.  Defaults to config value.
        tickers:
            If provided, only evaluate these tickers.

        Returns
        -------
        BacktestResult
            Complete results with trades, equity curve, and metrics.
        """
        bankroll = initial_bankroll or Decimal(str(self._config.initial_bankroll))

        logger.info(
            "backtest.starting",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            bankroll=str(bankroll),
            n_tickers=len(tickers) if tickers else "all",
            fill_model=self._config.fill_model,
        )

        # Load historical events
        events = self._loader.load(start_date, end_date, tickers=tickers)
        if not events:
            logger.warning("backtest.no_events")
            return self._empty_result(bankroll)

        # Initialize simulated exchange
        exchange = SimulatedExchange(
            initial_cash=bankroll,
            fee_calculator=self._fees,
        )

        # Simulated market state
        market_states: dict[str, Market] = {}
        orderbooks: dict[str, OrderbookSnapshot] = {}

        # Results tracking
        trades: list[dict[str, Any]] = []
        equity_timestamps: list[datetime] = []
        equity_values: list[float] = []
        predicted_probs: list[float] = []
        outcomes: list[int] = []

        # Track settlement outcomes for Brier score
        settlement_outcomes: dict[str, int] = {}

        # Process events
        for i, event in enumerate(events):
            # Progress logging
            if i > 0 and i % self._progress_interval == 0:
                logger.info(
                    "backtest.progress",
                    event=i,
                    total=len(events),
                    pct=round(100 * i / len(events), 1),
                    equity=str(exchange.get_equity()),
                    trades=len(trades),
                )

            try:
                if event.event_type == EventType.TICK:
                    market = self._process_tick(event)
                    if market is not None:
                        market_states[event.ticker] = market

                elif event.event_type == EventType.ORDERBOOK:
                    ob = self._process_orderbook(event)
                    if ob is not None:
                        orderbooks[event.ticker] = ob

                elif event.event_type == EventType.SETTLEMENT:
                    pnl = self._process_settlement(event, exchange)
                    # Record outcome for Brier score
                    result_str = event.data.get("market_result", "")
                    if result_str in ("yes", "all_yes"):
                        settlement_outcomes[event.ticker] = 1
                    elif result_str in ("no", "all_no"):
                        settlement_outcomes[event.ticker] = 0

                elif event.event_type == EventType.TRADE:
                    pass  # Trades are informational, no action needed

            except LookaheadError as exc:
                logger.warning(
                    "backtest.lookahead_blocked",
                    ticker=event.ticker,
                    error=str(exc),
                )
                continue

            # Evaluate market for trading opportunity
            ob = orderbooks.get(event.ticker)
            market = market_states.get(event.ticker)

            should_evaluate = (
                (event.event_type == EventType.TICK)
                or (event.event_type == EventType.ORDERBOOK and self._eval_on_orderbook)
            )

            if should_evaluate and ob is not None:
                trade_record = self._evaluate_and_trade(
                    ticker=event.ticker,
                    observation_time=event.timestamp,
                    market=market,
                    orderbook=ob,
                    exchange=exchange,
                )
                if trade_record is not None:
                    trades.append(trade_record)

                    # Track prediction for Brier score
                    predicted_probs.append(trade_record.get("model_prob", 0.5))

            # Record equity curve point
            equity_timestamps.append(event.timestamp)
            equity_values.append(float(exchange.get_equity()))

        # Match predictions to outcomes for Brier score
        matched_probs: list[float] = []
        matched_outcomes: list[int] = []
        for trade in trades:
            ticker = trade.get("ticker", "")
            if ticker in settlement_outcomes:
                matched_probs.append(trade.get("model_prob", 0.5))
                matched_outcomes.append(settlement_outcomes[ticker])

        # Build equity curve
        equity_curve = pd.Series(
            equity_values,
            index=pd.DatetimeIndex(equity_timestamps),
            name="equity",
        )

        logger.info(
            "backtest.complete",
            n_events=len(events),
            n_trades=len(trades),
            final_equity=str(exchange.get_equity()),
            total_fees=str(exchange.portfolio.total_fees),
        )

        return BacktestResult.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_curve,
            total_fees=exchange.portfolio.total_fees,
            predicted_probs=matched_probs if matched_probs else None,
            outcomes=matched_outcomes if matched_outcomes else None,
        )

    # ------------------------------------------------------------------
    # Event processors
    # ------------------------------------------------------------------

    def _process_tick(self, event: HistoricalEvent) -> Market | None:
        """Process a tick event and return a Market object."""
        data = event.data
        try:
            return Market(
                ticker=data["ticker"],
                event_ticker=data.get("event_ticker", ""),
                series_ticker=data.get("series_ticker", ""),
                title=data.get("title", ""),
                status=MarketStatus(data.get("status", "open")),
                yes_bid=Decimal(str(data.get("yes_bid", 0) or 0)),
                yes_ask=Decimal(str(data.get("yes_ask", 0) or 0)),
                last_price=Decimal(str(data.get("last_price", 0) or 0)),
                volume=int(data.get("volume", 0) or 0),
                open_interest=int(data.get("open_interest", 0) or 0),
                close_time=data.get("close_time", event.timestamp),
                result=MarketResult(data.get("result", "")) if data.get("result") else MarketResult.NOT_SETTLED,
                category=data.get("category", ""),
            )
        except Exception:
            logger.warning(
                "backtest.tick_parse_error",
                ticker=event.ticker,
                exc_info=True,
            )
            return None

    def _process_orderbook(self, event: HistoricalEvent) -> OrderbookSnapshot | None:
        """Process an orderbook event and return an OrderbookSnapshot."""
        data = event.data
        try:
            yes_levels_raw = data.get("yes_levels", [])
            no_levels_raw = data.get("no_levels", [])

            # Handle both list-of-lists and list-of-dicts formats
            yes_levels = self._parse_levels(yes_levels_raw)
            no_levels = self._parse_levels(no_levels_raw)

            return OrderbookSnapshot(
                ticker=data["ticker"],
                yes_bids=tuple(yes_levels),
                no_bids=tuple(no_levels),
                seq=int(data.get("seq", 0) or 0),
                timestamp=event.timestamp,
            )
        except Exception:
            logger.warning(
                "backtest.orderbook_parse_error",
                ticker=event.ticker,
                exc_info=True,
            )
            return None

    @staticmethod
    def _parse_levels(levels_raw: Any) -> list[OrderbookLevel]:
        """Parse orderbook levels from various formats."""
        levels: list[OrderbookLevel] = []
        if not levels_raw:
            return levels

        for lvl in levels_raw:
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                levels.append(OrderbookLevel(
                    price=Decimal(str(lvl[0])),
                    contracts=Decimal(str(lvl[1])),
                ))
            elif isinstance(lvl, dict):
                levels.append(OrderbookLevel(
                    price=Decimal(str(lvl.get("price", 0))),
                    contracts=Decimal(str(lvl.get("contracts", lvl.get("quantity", 0)))),
                ))

        return levels

    def _process_settlement(
        self, event: HistoricalEvent, exchange: SimulatedExchange
    ) -> Decimal:
        """Process a settlement event."""
        data = event.data
        ticker = data["ticker"]
        result_str = data.get("market_result", "")

        try:
            result = MarketResult(result_str)
        except ValueError:
            result = MarketResult.NOT_SETTLED

        # Record settlement time for leakage guard
        self._guard.add_settlement_time(ticker, event.timestamp)

        return exchange.process_settlement(ticker, result)

    # ------------------------------------------------------------------
    # Trading evaluation
    # ------------------------------------------------------------------

    def _evaluate_and_trade(
        self,
        ticker: str,
        observation_time: datetime,
        market: Market | None,
        orderbook: OrderbookSnapshot,
        exchange: SimulatedExchange,
    ) -> dict[str, Any] | None:
        """Evaluate a market and execute a simulated trade if warranted.

        Runs the full live pipeline:
          1. Validate temporal constraints
          2. Build FeatureContext
          3. Compute features
          4. Run model
          5. Compute edge
          6. Size position
          7. Risk check
          8. Simulate fill

        Returns a trade record dict, or None if no trade was taken.
        """
        # 1. Validate temporal constraints
        try:
            self._guard.validate_no_label_access(ticker, observation_time)
        except LookaheadError:
            return None

        # 2. Build FeatureContext
        context = FeatureContext(
            ticker=ticker,
            observation_time=observation_time,
            market_state=market,
            orderbook=orderbook,
        )

        # Validate feature context
        try:
            self._guard.validate_feature_context(context)
        except LookaheadError as exc:
            logger.debug("backtest.feature_lookahead", ticker=ticker, error=str(exc))
            return None

        # 3. Compute features
        features = self._pipeline.compute(context)
        if not features:
            return None

        # 4. Run model
        prediction = self._model.predict_proba(features)

        # 5. Compute edge
        edge = self._edge_calc.compute_edge(
            prediction.probability,
            orderbook,
            is_maker=True,  # Assume maker for backtest (conservative)
        )

        if not edge.is_actionable:
            return None

        # 6. Size position
        bankroll = exchange.get_equity()
        existing_exposure = exchange.portfolio.get_total_exposure()

        size = self._sizer.size(
            edge_result=edge,
            bankroll=bankroll,
            model_confidence=prediction.confidence,
            existing_exposure=existing_exposure,
        )

        if size.contracts <= 0:
            return None

        # 7. Risk check
        portfolio_state = PortfolioState(
            positions={
                t: p.contracts
                for t, p in exchange.portfolio.positions.items()
            },
            position_costs={
                t: p.cost_basis
                for t, p in exchange.portfolio.positions.items()
            },
            total_exposure=existing_exposure,
            bankroll=bankroll,
            current_equity=bankroll,
            peak_equity=exchange.portfolio.peak_equity,
        )

        proposed = ProposedTrade(
            ticker=ticker,
            category=market.category if market else "",
            side=edge.side,
            action=edge.action,
            contracts=size.contracts,
            price=edge.target_price,
        )

        risk_check = self._risk_limits.check(proposed, portfolio_state)

        if not risk_check.approved:
            return None

        # Apply adjusted size if applicable
        actual_contracts = size.contracts
        if risk_check.adjusted_size is not None and risk_check.adjusted_size > 0:
            actual_contracts = risk_check.adjusted_size

        # 8. Simulate fill
        order = OrderRequest(
            ticker=ticker,
            side=Side(edge.side),
            action=Action(edge.action),
            count=actual_contracts,
            yes_price=edge.target_price,
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )

        sim_fill = exchange.process_order(order, orderbook, self._fill_sim)

        if not sim_fill.filled or sim_fill.filled_contracts <= 0:
            return None

        # Build trade record
        trade_record = {
            "timestamp": observation_time.isoformat(),
            "ticker": ticker,
            "side": edge.side,
            "action": edge.action,
            "contracts": sim_fill.filled_contracts,
            "fill_price": float(sim_fill.fill_price),
            "fees": float(sim_fill.fees),
            "slippage": float(sim_fill.slippage),
            "model_prob": prediction.probability,
            "raw_prob": prediction.raw_probability,
            "confidence": prediction.confidence,
            "model_name": prediction.model_name,
            "edge": edge.fee_adjusted_edge,
            "raw_edge": edge.raw_edge,
            "implied_prob": edge.implied_probability,
            "kelly_fraction": size.kelly_fraction,
            "adjusted_fraction": size.adjusted_fraction,
            "dollar_risk": float(sim_fill.fill_price * Decimal(sim_fill.filled_contracts)),
            "equity_at_trade": float(exchange.get_equity()),
            "fill_model": sim_fill.fill_model,
            "pnl": 0.0,  # Will be determined at settlement
        }

        logger.debug(
            "backtest.trade_executed",
            ticker=ticker,
            side=edge.side,
            contracts=sim_fill.filled_contracts,
            fill_price=str(sim_fill.fill_price),
            edge=round(edge.fee_adjusted_edge, 4),
        )

        return trade_record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, bankroll: Decimal) -> BacktestResult:
        """Return an empty BacktestResult when no events are available."""
        now = datetime.now(timezone.utc)
        equity_curve = pd.Series(
            [float(bankroll)],
            index=pd.DatetimeIndex([now]),
            name="equity",
        )

        return BacktestResult(
            trades=[],
            equity_curve=equity_curve,
            daily_returns=pd.Series(dtype=float),
            total_pnl=_ZERO,
            total_fees=_ZERO,
            net_pnl=_ZERO,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_edge_predicted=0.0,
            avg_edge_realized=0.0,
            brier_score=1.0,
            num_trades=0,
        )
