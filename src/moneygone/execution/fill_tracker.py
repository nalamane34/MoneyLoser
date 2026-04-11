"""Fill tracking with prediction context for post-trade analysis.

Records every fill alongside the model prediction and edge calculation
that motivated the trade, enabling slippage analysis and model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from moneygone.data.store import DataStore
from moneygone.exchange.types import Fill
from moneygone.models.base import ModelPrediction
from moneygone.signals.edge import EdgeResult

if TYPE_CHECKING:
    from moneygone.exchange.rest_client import KalshiRestClient
    from moneygone.risk.portfolio import PortfolioTracker

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class FillStats:
    """Aggregate statistics over recent fills."""

    count: int
    """Total number of fills tracked."""

    avg_fill_price: float
    """Average fill price across all fills."""

    avg_slippage: float
    """Average slippage vs. target price (fill_price - target_price)."""

    fill_rate: float
    """Fraction of submitted orders that resulted in fills (0-1)."""

    total_fees: Decimal
    """Total fees paid across all fills."""

    total_volume: int
    """Total contracts filled."""


@dataclass
class FillRecord:
    """A fill with its associated prediction and edge context."""

    fill: Fill
    prediction: ModelPrediction
    edge: EdgeResult
    slippage: float
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class CloserFillRecord:
    """A fill from a closer strategy (ResolutionSniper or LiveEventEdge).

    Closer strategies bypass the standard model/edge pipeline, so we
    capture their decision context directly instead of requiring
    ModelPrediction and EdgeResult objects.
    """

    fill: Fill
    strategy: str
    """Which closer strategy produced this fill ('sniper' or 'live_edge')."""

    signal_source: str
    """Data feed that triggered the trade (e.g. 'espn', 'noaa', 'coinalyze')."""

    confidence: float
    """Strategy confidence at the time of execution (0-1)."""

    expected_profit: float
    """Expected profit per contract after fees."""

    entry_price: float
    """Price at which the order was placed."""

    contracts: int
    """Number of contracts in the fill."""

    category: str
    """Market category (e.g. 'sports', 'weather', 'crypto')."""

    signal_data: dict[str, Any] = field(default_factory=dict)
    """Arbitrary signal context for post-trade analysis."""

    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FillTracker:
    """Records fills with prediction context for analysis.

    Persists fill data to the DataStore and maintains an in-memory
    buffer for quick statistics.

    Parameters
    ----------
    store:
        DataStore for persistence.
    max_memory_fills:
        Maximum number of fills to keep in memory for stats.
    """

    def __init__(
        self,
        store: DataStore | None = None,
        max_memory_fills: int = 10_000,
    ) -> None:
        self._store = store
        self._max_fills = max_memory_fills
        self._fills: list[FillRecord] = []
        self._closer_fills: list[CloserFillRecord] = []
        self._total_submitted: int = 0

    def record_submission(self) -> None:
        """Record that an order was submitted (for fill-rate tracking)."""
        self._total_submitted += 1

    def on_fill(
        self,
        fill: Fill,
        prediction: ModelPrediction,
        edge: EdgeResult,
        *,
        cycle_id: str | None = None,
        category: str | None = None,
    ) -> None:
        """Record a fill with its associated prediction context.

        Parameters
        ----------
        fill:
            The fill event from the exchange.
        prediction:
            The model prediction that motivated this trade.
        edge:
            The edge calculation at the time of the trade decision.
        """
        slippage = float(fill.price - edge.target_price)

        record = FillRecord(
            fill=fill,
            prediction=prediction,
            edge=edge,
            slippage=slippage,
        )

        self._fills.append(record)

        # Trim memory buffer
        if len(self._fills) > self._max_fills:
            self._fills = self._fills[-self._max_fills:]

        # Persist to store
        if self._store is not None:
            try:
                self._store.insert_fills([
                    {
                        "trade_id": fill.fill_id,
                        "ticker": fill.ticker,
                        "side": fill.side.value,
                        "action": fill.action.value,
                        "count": fill.count,
                        "price": float(fill.price),
                        "is_taker": fill.is_taker,
                        "fill_time": fill.created_time.isoformat(),
                    }
                ])
            except Exception:
                logger.warning(
                    "fill_tracker.persist_failed",
                    trade_id=fill.fill_id,
                    exc_info=True,
                )

        logger.info(
            "fill_tracker.recorded",
            trade_id=fill.fill_id,
            ticker=fill.ticker,
            cycle_id=cycle_id,
            category=category,
            price=str(fill.price),
            target_price=str(edge.target_price),
            slippage=round(slippage, 4),
            model_prob=round(prediction.probability, 4),
            confidence=round(prediction.confidence, 4),
            edge=round(edge.fee_adjusted_edge, 4),
            fill_edge=round(edge.fee_adjusted_edge - slippage, 4),
        )

    def on_closer_fill(
        self,
        fill: Fill,
        *,
        strategy: str,
        signal_source: str,
        confidence: float,
        expected_profit: float,
        entry_price: float,
        contracts: int,
        category: str,
        signal_data: dict[str, Any] | None = None,
    ) -> None:
        """Record a fill from a closer strategy (ResolutionSniper / LiveEventEdge).

        Closer strategies operate outside the standard model/edge pipeline,
        so this method captures their decision context directly.

        Parameters
        ----------
        fill:
            The fill event from the exchange.
        strategy:
            Which closer strategy ('sniper' or 'live_edge').
        signal_source:
            Data feed that triggered the trade (e.g. 'espn', 'noaa').
        confidence:
            Strategy confidence at execution time (0-1).
        expected_profit:
            Expected profit per contract after fees.
        entry_price:
            Price at which the order was placed.
        contracts:
            Number of contracts in the fill.
        category:
            Market category (e.g. 'sports', 'weather').
        signal_data:
            Arbitrary signal context for post-trade analysis.
        """
        record = CloserFillRecord(
            fill=fill,
            strategy=strategy,
            signal_source=signal_source,
            confidence=confidence,
            expected_profit=expected_profit,
            entry_price=entry_price,
            contracts=contracts,
            category=category,
            signal_data=signal_data or {},
        )

        self._closer_fills.append(record)
        self._total_submitted += 1

        # Trim memory buffer
        if len(self._closer_fills) > self._max_fills:
            self._closer_fills = self._closer_fills[-self._max_fills:]

        # Persist to store
        if self._store is not None:
            try:
                self._store.insert_fills([
                    {
                        "trade_id": fill.fill_id,
                        "ticker": fill.ticker,
                        "side": fill.side.value,
                        "action": fill.action.value,
                        "count": fill.count,
                        "price": float(fill.price),
                        "is_taker": fill.is_taker,
                        "fill_time": fill.created_time.isoformat(),
                    }
                ])
            except Exception:
                logger.warning(
                    "fill_tracker.persist_failed",
                    trade_id=fill.fill_id,
                    exc_info=True,
                )

        logger.info(
            "fill_tracker.closer_recorded",
            trade_id=fill.fill_id,
            ticker=fill.ticker,
            strategy=strategy,
            signal_source=signal_source,
            category=category,
            price=str(fill.price),
            entry_price=round(entry_price, 4),
            contracts=contracts,
            confidence=round(confidence, 4),
            expected_profit=round(expected_profit, 4),
            signal_data=signal_data,
        )

    def get_recent_closer_fills(self, n: int = 100) -> list[CloserFillRecord]:
        """Return the *n* most recent closer strategy fill records."""
        return self._closer_fills[-n:]

    def get_recent_fills(self, n: int = 100) -> list[FillRecord]:
        """Return the *n* most recent fill records.

        Parameters
        ----------
        n:
            Maximum number of fills to return.

        Returns
        -------
        list[FillRecord]
            Most recent fills, newest last.
        """
        return self._fills[-n:]

    def get_fill_stats(self) -> FillStats:
        """Compute aggregate statistics over all tracked fills.

        Returns
        -------
        FillStats
            Summary statistics including average fill price, slippage,
            and fill rate.
        """
        if not self._fills:
            return FillStats(
                count=0,
                avg_fill_price=0.0,
                avg_slippage=0.0,
                fill_rate=0.0,
                total_fees=_ZERO,
                total_volume=0,
            )

        prices = [float(r.fill.price) for r in self._fills]
        slippages = [r.slippage for r in self._fills]
        volumes = [r.fill.count for r in self._fills]

        # Calculate fees
        from moneygone.signals.fees import KalshiFeeCalculator
        fee_calc = KalshiFeeCalculator()
        total_fees = _ZERO
        for r in self._fills:
            if r.fill.is_taker:
                total_fees += fee_calc.taker_fee(r.fill.count, r.fill.price)

        fill_rate = len(self._fills) / self._total_submitted if self._total_submitted > 0 else 0.0

        return FillStats(
            count=len(self._fills),
            avg_fill_price=sum(prices) / len(prices),
            avg_slippage=sum(slippages) / len(slippages),
            fill_rate=min(1.0, fill_rate),
            total_fees=total_fees,
            total_volume=sum(volumes),
        )

    async def reconcile_with_exchange(
        self,
        rest_client: KalshiRestClient,
        portfolio: PortfolioTracker,
    ) -> dict[str, Any]:
        """Compare locally tracked P&L against the exchange-reported balance.

        Parameters
        ----------
        rest_client:
            Kalshi REST client used to fetch the authoritative balance.
        portfolio:
            Local portfolio tracker that maintains realized P&L and cash.

        Returns
        -------
        dict
            Keys: ``local_pnl``, ``exchange_balance``, ``discrepancy``.
        """
        try:
            balance = await rest_client.get_balance()
        except Exception:
            logger.warning("reconciliation.exchange_fetch_failed", exc_info=True)
            return {
                "local_pnl": float(portfolio.realized_pnl),
                "exchange_balance": None,
                "discrepancy": None,
            }

        local_pnl = float(portfolio.realized_pnl)
        exchange_balance = float(balance.total)
        discrepancy = round(exchange_balance - local_pnl, 4)

        if abs(discrepancy) > 1.0:
            logger.warning(
                "reconciliation.pnl_discrepancy",
                local_pnl=local_pnl,
                exchange_balance=exchange_balance,
                discrepancy=discrepancy,
            )
        else:
            logger.info(
                "reconciliation.ok",
                local_pnl=local_pnl,
                exchange_balance=exchange_balance,
                discrepancy=discrepancy,
            )

        return {
            "local_pnl": local_pnl,
            "exchange_balance": exchange_balance,
            "discrepancy": discrepancy,
        }
