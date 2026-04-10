"""Profit and loss tracking with trade-level attribution.

Records every trade alongside the model prediction and edge estimate that
motivated it, enabling detailed performance attribution by category,
time period, and model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from moneygone.exchange.types import Fill, Settlement
from moneygone.signals.edge import EdgeResult

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight prediction container (avoids coupling to filter module)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PredictionSnapshot:
    """Minimal copy of a model prediction stored alongside a trade."""

    probability: float
    confidence: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal trade record
# ---------------------------------------------------------------------------


@dataclass
class _TradeRecord:
    """Internal record of a single trade with context."""

    trade_id: str
    ticker: str
    side: str
    action: str
    count: int
    price: float
    is_taker: bool
    fill_time: datetime
    predicted_prob: float
    predicted_confidence: float
    raw_edge: float
    fee_adjusted_edge: float
    category: str = ""
    settlement_result: str | None = None
    settlement_payout: float | None = None


# ---------------------------------------------------------------------------
# PnLSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PnLSummary:
    """Aggregate PnL statistics for a period."""

    gross_pnl: float
    fees_paid: float
    net_pnl: float
    num_trades: int
    win_rate: float
    avg_edge_predicted: float
    avg_edge_realized: float
    sharpe: float


# ---------------------------------------------------------------------------
# PnLTracker
# ---------------------------------------------------------------------------


class PnLTracker:
    """Records all trades and settlements, providing PnL summaries and
    attribution analysis.

    Parameters
    ----------
    fee_rate_taker:
        Approximate taker fee rate for estimation (actual fees come from
        the fill data where available).
    """

    def __init__(self, fee_rate_taker: float = 0.02) -> None:
        self._fee_rate_taker = fee_rate_taker
        self._trades: list[_TradeRecord] = []
        self._settlements: dict[str, tuple[str, float]] = {}  # ticker -> (result, payout)
        log.info("pnl_tracker.initialized")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(
        self,
        fill: Fill,
        prediction: Any,
        edge: EdgeResult,
        category: str = "",
    ) -> None:
        """Record a trade fill with the prediction and edge that produced it.

        Parameters
        ----------
        fill:
            Exchange fill object.
        prediction:
            Model prediction object (must have ``.probability``,
            ``.confidence``, ``.timestamp`` attributes).
        edge:
            Edge calculation result.
        category:
            Market category for attribution.
        """
        record = _TradeRecord(
            trade_id=fill.fill_id,
            ticker=fill.ticker,
            side=fill.side.value,
            action=fill.action.value,
            count=fill.count,
            price=float(fill.price),
            is_taker=fill.is_taker,
            fill_time=fill.created_time,
            predicted_prob=prediction.probability,
            predicted_confidence=prediction.confidence,
            raw_edge=edge.raw_edge,
            fee_adjusted_edge=edge.fee_adjusted_edge,
            category=category,
        )
        self._trades.append(record)
        log.debug(
            "pnl.trade_recorded",
            ticker=fill.ticker,
            side=fill.side.value,
            price=float(fill.price),
            edge=round(edge.fee_adjusted_edge, 4),
        )

    def record_settlement(self, settlement: Settlement) -> None:
        """Record a market settlement so PnL can be finalized.

        Parameters
        ----------
        settlement:
            Exchange settlement object.
        """
        revenue_dollars = float(settlement.revenue) / 100  # cents to dollars
        self._settlements[settlement.ticker] = (
            settlement.market_result.value,
            revenue_dollars,
        )

        # Retroactively update trade records for this ticker
        for trade in self._trades:
            if trade.ticker == settlement.ticker:
                trade.settlement_result = settlement.market_result.value
                trade.settlement_payout = revenue_dollars

        log.debug(
            "pnl.settlement_recorded",
            ticker=settlement.ticker,
            result=settlement.market_result.value,
            revenue=revenue_dollars,
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_summary(self, period: str = "all") -> PnLSummary:
        """Compute PnL summary statistics.

        Parameters
        ----------
        period:
            ``"daily"``, ``"weekly"``, or ``"all"``.

        Returns
        -------
        PnLSummary
        """
        trades = self._filter_by_period(period)
        if not trades:
            return PnLSummary(
                gross_pnl=0.0,
                fees_paid=0.0,
                net_pnl=0.0,
                num_trades=0,
                win_rate=0.0,
                avg_edge_predicted=0.0,
                avg_edge_realized=0.0,
                sharpe=0.0,
            )

        gross_pnl = 0.0
        fees_paid = 0.0
        wins = 0
        edges_predicted: list[float] = []
        edges_realized: list[float] = []
        trade_pnls: list[float] = []

        for t in trades:
            # Estimate per-trade PnL
            trade_pnl = self._estimate_trade_pnl(t)
            fee = self._estimate_fee(t)

            gross_pnl += trade_pnl + fee  # gross = before fees
            fees_paid += fee
            trade_pnls.append(trade_pnl)

            if trade_pnl > 0:
                wins += 1

            edges_predicted.append(t.fee_adjusted_edge)
            if t.settlement_result is not None:
                realized_edge = trade_pnl / (t.count * 1.0) if t.count > 0 else 0.0
                edges_realized.append(realized_edge)

        net_pnl = gross_pnl - fees_paid
        win_rate = wins / len(trades) if trades else 0.0
        avg_edge_predicted = (
            sum(edges_predicted) / len(edges_predicted) if edges_predicted else 0.0
        )
        avg_edge_realized = (
            sum(edges_realized) / len(edges_realized) if edges_realized else 0.0
        )

        # Sharpe ratio (annualized, assuming ~252 trading days)
        pnl_arr = np.array(trade_pnls)
        if len(pnl_arr) > 1 and pnl_arr.std() > 0:
            sharpe = float(pnl_arr.mean() / pnl_arr.std() * math.sqrt(252))
        else:
            sharpe = 0.0

        return PnLSummary(
            gross_pnl=round(gross_pnl, 4),
            fees_paid=round(fees_paid, 4),
            net_pnl=round(net_pnl, 4),
            num_trades=len(trades),
            win_rate=round(win_rate, 4),
            avg_edge_predicted=round(avg_edge_predicted, 6),
            avg_edge_realized=round(avg_edge_realized, 6),
            sharpe=round(sharpe, 4),
        )

    def get_attribution(self) -> dict[str, PnLSummary]:
        """Compute PnL summary grouped by market category.

        Returns
        -------
        dict
            Mapping of category name to :class:`PnLSummary`.
        """
        by_category: dict[str, list[_TradeRecord]] = defaultdict(list)
        for t in self._trades:
            key = t.category or "uncategorized"
            by_category[key].append(t)

        result: dict[str, PnLSummary] = {}
        for category, trades in by_category.items():
            # Temporarily swap trades list, compute, restore
            original = self._trades
            self._trades = trades
            result[category] = self.get_summary("all")
            self._trades = original

        return result

    def get_cumulative_pnl(self) -> list[tuple[datetime, float]]:
        """Return a time series of cumulative net PnL.

        Returns
        -------
        list[tuple[datetime, float]]
            Chronologically ordered ``(timestamp, cumulative_pnl)`` pairs.
        """
        sorted_trades = sorted(self._trades, key=lambda t: t.fill_time)
        cumulative = 0.0
        result: list[tuple[datetime, float]] = []

        for t in sorted_trades:
            pnl = self._estimate_trade_pnl(t)
            fee = self._estimate_fee(t)
            cumulative += pnl - fee
            result.append((t.fill_time, round(cumulative, 4)))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_period(self, period: str) -> list[_TradeRecord]:
        """Filter trades by time period."""
        if period == "all":
            return list(self._trades)

        now = datetime.now(timezone.utc)
        if period == "daily":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta

            cutoff -= timedelta(days=now.weekday())
        else:
            return list(self._trades)

        return [
            t
            for t in self._trades
            if t.fill_time.tzinfo is not None and t.fill_time >= cutoff
        ]

    def _estimate_trade_pnl(self, trade: _TradeRecord) -> float:
        """Estimate PnL for a single trade.

        If the market has settled, computes exact PnL.  Otherwise uses
        the predicted edge as the expected PnL.
        """
        if trade.settlement_result is not None:
            # Settled: compute exact PnL
            is_yes_side = trade.side == "yes"
            is_buy = trade.action == "buy"

            if is_buy and is_yes_side:
                if trade.settlement_result in ("yes", "all_yes"):
                    return (1.0 - trade.price) * trade.count
                else:
                    return -trade.price * trade.count
            elif is_buy and not is_yes_side:
                if trade.settlement_result in ("no", "all_no"):
                    return (1.0 - (1.0 - trade.price)) * trade.count
                else:
                    return -(1.0 - trade.price) * trade.count
            else:
                # Sell: inverse of buy
                return -self._estimate_trade_pnl(
                    _TradeRecord(
                        trade_id=trade.trade_id,
                        ticker=trade.ticker,
                        side=trade.side,
                        action="buy",
                        count=trade.count,
                        price=trade.price,
                        is_taker=trade.is_taker,
                        fill_time=trade.fill_time,
                        predicted_prob=trade.predicted_prob,
                        predicted_confidence=trade.predicted_confidence,
                        raw_edge=trade.raw_edge,
                        fee_adjusted_edge=trade.fee_adjusted_edge,
                        category=trade.category,
                        settlement_result=trade.settlement_result,
                        settlement_payout=trade.settlement_payout,
                    )
                )

        # Unsettled: use expected PnL from edge
        return trade.fee_adjusted_edge * trade.count

    def _estimate_fee(self, trade: _TradeRecord) -> float:
        """Estimate fee for a trade."""
        if trade.is_taker:
            p = trade.price
            raw = 0.07 * p * (1 - p)
            return min(raw, 0.02) * trade.count
        return 0.0
