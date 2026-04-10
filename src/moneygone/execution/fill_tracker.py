"""Fill tracking with prediction context for post-trade analysis.

Records every fill alongside the model prediction and edge calculation
that motivated the trade, enabling slippage analysis and model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from moneygone.data.store import DataStore
from moneygone.exchange.types import Fill
from moneygone.models.base import ModelPrediction
from moneygone.signals.edge import EdgeResult

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
