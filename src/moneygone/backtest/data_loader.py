"""Historical data loader for backtesting.

Loads market events (ticks, orderbook snapshots, settlements) from the
DataStore in chronological order, ensuring strict timestamp ordering
for deterministic replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from moneygone.data.store import DataStore
from moneygone.exchange.types import (
    Market,
    MarketResult,
    MarketStatus,
    OrderbookLevel,
    OrderbookSnapshot,
    Settlement,
)

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    TICK = "tick"
    ORDERBOOK = "orderbook"
    SETTLEMENT = "settlement"
    TRADE = "trade"


@dataclass(frozen=True, slots=True)
class HistoricalEvent:
    """A single historical event for backtesting replay."""

    timestamp: datetime
    """When the event occurred."""

    event_type: EventType
    """Type of event (tick, orderbook, settlement, trade)."""

    ticker: str
    """Market ticker this event belongs to."""

    data: dict[str, Any]
    """Raw event data."""


class HistoricalDataLoader:
    """Loads historical events from the DataStore in chronological order.

    Merges multiple data streams (market state, orderbook, settlements,
    trades) into a single time-ordered event sequence for replay.

    Parameters
    ----------
    store:
        DataStore to query for historical data.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def load(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> list[HistoricalEvent]:
        """Load all historical events in the given date range.

        Parameters
        ----------
        start_date:
            Start of the replay period (inclusive).
        end_date:
            End of the replay period (inclusive).
        tickers:
            If provided, only load events for these tickers.
        categories:
            If provided, only load events for markets in these categories.

        Returns
        -------
        list[HistoricalEvent]
            Events sorted by timestamp in ascending order.

        Raises
        ------
        ValueError
            If start_date >= end_date.
        """
        if start_date >= end_date:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )

        events: list[HistoricalEvent] = []

        # Load market state snapshots (tick events)
        events.extend(self._load_market_states(start_date, end_date, tickers, categories))

        # Load orderbook snapshots
        events.extend(self._load_orderbooks(start_date, end_date, tickers))

        # Load settlements
        events.extend(self._load_settlements(start_date, end_date, tickers))

        # Load trades
        events.extend(self._load_trades(start_date, end_date, tickers))

        # Sort by timestamp for deterministic replay
        events.sort(key=lambda e: e.timestamp)

        # Validate ordering
        self._validate_ordering(events)

        logger.info(
            "data_loader.loaded",
            n_events=len(events),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            n_tickers=len(tickers) if tickers else "all",
        )

        return events

    # ------------------------------------------------------------------
    # Loaders for each data type
    # ------------------------------------------------------------------

    def _load_market_states(
        self,
        start: datetime,
        end: datetime,
        tickers: list[str] | None,
        categories: list[str] | None,
    ) -> list[HistoricalEvent]:
        """Load market state snapshots as tick events."""
        events: list[HistoricalEvent] = []

        # Query all market states in the time range
        ticker_filter = ""
        params: list[Any] = [start, end]

        if tickers:
            placeholders = ", ".join("?" for _ in tickers)
            ticker_filter = f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        category_filter = ""
        if categories:
            cat_placeholders = ", ".join("?" for _ in categories)
            category_filter = f" AND category IN ({cat_placeholders})"
            params.extend(categories)

        # Use ingested_at OR close_time for time filtering.
        # When data is backfilled, ingested_at is the INSERT time (not event time),
        # so we also check close_time to catch backfilled rows.
        query = f"""
            SELECT *
            FROM market_states
            WHERE (ingested_at >= ? AND ingested_at <= ?)
               OR (close_time >= ? AND close_time <= ?)
            {ticker_filter}
            {category_filter}
            ORDER BY COALESCE(close_time, ingested_at)
        """
        params.extend([start, end])  # For close_time range

        try:
            results = self._store._conn.execute(query, params).fetchall()
            if results:
                columns = [desc[0] for desc in self._store._conn.description]
                for row in results:
                    row_dict = dict(zip(columns, row))
                    ts = row_dict.get("close_time") or row_dict.get("ingested_at", start)
                    events.append(HistoricalEvent(
                        timestamp=ts,
                        event_type=EventType.TICK,
                        ticker=row_dict["ticker"],
                        data=row_dict,
                    ))
        except Exception:
            logger.warning("data_loader.market_states_error", exc_info=True)

        return events

    def _load_orderbooks(
        self,
        start: datetime,
        end: datetime,
        tickers: list[str] | None,
    ) -> list[HistoricalEvent]:
        """Load orderbook snapshots."""
        events: list[HistoricalEvent] = []

        ticker_filter = ""
        params: list[Any] = [start, end]

        if tickers:
            placeholders = ", ".join("?" for _ in tickers)
            ticker_filter = f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        # Use snapshot_time for orderbooks (ingested_at is insertion time,
        # not when the snapshot was taken; backfilled data uses snapshot_time).
        query = f"""
            SELECT *
            FROM orderbook_snapshots
            WHERE (snapshot_time >= ? AND snapshot_time <= ?)
               OR (ingested_at >= ? AND ingested_at <= ?)
            {ticker_filter}
            ORDER BY COALESCE(snapshot_time, ingested_at)
        """
        params.extend([start, end])  # For ingested_at fallback

        try:
            results = self._store._conn.execute(query, params).fetchall()
            if results:
                columns = [desc[0] for desc in self._store._conn.description]
                for row in results:
                    row_dict = dict(zip(columns, row))
                    # Deserialise JSON levels
                    if isinstance(row_dict.get("yes_levels"), str):
                        row_dict["yes_levels"] = json.loads(row_dict["yes_levels"])
                    if isinstance(row_dict.get("no_levels"), str):
                        row_dict["no_levels"] = json.loads(row_dict["no_levels"])

                    ts = row_dict.get("snapshot_time") or row_dict.get("ingested_at", start)
                    events.append(HistoricalEvent(
                        timestamp=ts,
                        event_type=EventType.ORDERBOOK,
                        ticker=row_dict["ticker"],
                        data=row_dict,
                    ))
        except Exception:
            logger.warning("data_loader.orderbooks_error", exc_info=True)

        return events

    def _load_settlements(
        self,
        start: datetime,
        end: datetime,
        tickers: list[str] | None,
    ) -> list[HistoricalEvent]:
        """Load settlement records."""
        events: list[HistoricalEvent] = []

        ticker_filter = ""
        params: list[Any] = [start, end]

        if tickers:
            placeholders = ", ".join("?" for _ in tickers)
            ticker_filter = f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        query = f"""
            SELECT *
            FROM settlements_log
            WHERE settled_time >= ? AND settled_time <= ?
            {ticker_filter}
            ORDER BY settled_time
        """

        try:
            results = self._store._conn.execute(query, params).fetchall()
            if results:
                columns = [desc[0] for desc in self._store._conn.description]
                for row in results:
                    row_dict = dict(zip(columns, row))
                    ts = row_dict.get("settled_time", start)
                    events.append(HistoricalEvent(
                        timestamp=ts,
                        event_type=EventType.SETTLEMENT,
                        ticker=row_dict["ticker"],
                        data=row_dict,
                    ))
        except Exception:
            logger.warning("data_loader.settlements_error", exc_info=True)

        return events

    def _load_trades(
        self,
        start: datetime,
        end: datetime,
        tickers: list[str] | None,
    ) -> list[HistoricalEvent]:
        """Load public trade records."""
        events: list[HistoricalEvent] = []

        if tickers:
            for ticker in tickers:
                trades = self._store.get_trades_between(ticker, start, end)
                for t in trades:
                    events.append(HistoricalEvent(
                        timestamp=t["trade_time"],
                        event_type=EventType.TRADE,
                        ticker=t["ticker"],
                        data=t,
                    ))
        else:
            # Query all trades in the range
            try:
                results = self._store._conn.execute(
                    """
                    SELECT * FROM trades
                    WHERE trade_time >= ? AND trade_time <= ?
                    ORDER BY trade_time
                    """,
                    [start, end],
                ).fetchall()
                if results:
                    columns = [desc[0] for desc in self._store._conn.description]
                    for row in results:
                        row_dict = dict(zip(columns, row))
                        events.append(HistoricalEvent(
                            timestamp=row_dict["trade_time"],
                            event_type=EventType.TRADE,
                            ticker=row_dict["ticker"],
                            data=row_dict,
                        ))
            except Exception:
                logger.warning("data_loader.trades_error", exc_info=True)

        return events

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ordering(events: list[HistoricalEvent]) -> None:
        """Verify events are in strictly non-decreasing timestamp order."""
        for i in range(1, len(events)):
            if events[i].timestamp < events[i - 1].timestamp:
                raise ValueError(
                    f"Event ordering violation at index {i}: "
                    f"{events[i].timestamp} < {events[i-1].timestamp}"
                )
