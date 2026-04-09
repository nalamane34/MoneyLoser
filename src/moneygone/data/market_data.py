"""Market data recorder that buffers WebSocket events and flushes to DuckDB.

The recorder accepts individual events from the Kalshi WebSocket feed,
buffers them in memory, and periodically flushes batches to the
:class:`~moneygone.data.store.DataStore`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog

from moneygone.data.store import DataStore

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocol so we don't hard-depend on an exchange.types module
# that may not exist yet.
# ---------------------------------------------------------------------------


class WSEvent(Protocol):
    """Minimal protocol for a WebSocket event payload."""

    @property
    def type(self) -> str: ...  # noqa: E704

    @property
    def data(self) -> dict[str, Any]: ...  # noqa: E704


# ---------------------------------------------------------------------------
# Internal buffer dataclass
# ---------------------------------------------------------------------------


@dataclass
class _Buffer:
    market_states: list[dict[str, Any]] = field(default_factory=list)
    orderbook_snapshots: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.market_states)
            + len(self.orderbook_snapshots)
            + len(self.trades)
        )

    def clear(self) -> None:
        self.market_states.clear()
        self.orderbook_snapshots.clear()
        self.trades.clear()


# ---------------------------------------------------------------------------
# MarketDataRecorder
# ---------------------------------------------------------------------------


class MarketDataRecorder:
    """Buffers Kalshi WebSocket events and batch-inserts into DuckDB.

    Parameters
    ----------
    store:
        The :class:`DataStore` instance to flush into.
    flush_interval_seconds:
        Maximum seconds between automatic flushes.  Default ``5.0``.
    flush_threshold:
        Maximum buffered events before a flush is triggered.  Default ``100``.
    orderbook_snapshot_interval:
        Minimum seconds between recording orderbook snapshots for the same
        ticker.  Default ``10.0``.
    """

    def __init__(
        self,
        store: DataStore,
        *,
        flush_interval_seconds: float = 5.0,
        flush_threshold: int = 100,
        orderbook_snapshot_interval: float = 10.0,
    ) -> None:
        self._store = store
        self._flush_interval = flush_interval_seconds
        self._flush_threshold = flush_threshold
        self._orderbook_interval = orderbook_snapshot_interval

        self._buffer = _Buffer()
        self._last_flush_time = time.monotonic()
        self._last_orderbook_time: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic flush background task."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())
            logger.info("market_data_recorder.started")

    async def stop(self) -> None:
        """Stop the background flush task and do a final flush."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()
        logger.info("market_data_recorder.stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_ticker_update(self, event: WSEvent | dict[str, Any]) -> None:
        """Handle a ticker / market-state update from the WebSocket."""
        data = event["data"] if isinstance(event, dict) else event.data
        row = {
            "ticker": data["ticker"],
            "event_ticker": data.get("event_ticker", ""),
            "title": data.get("title", ""),
            "status": data.get("status", "unknown"),
            "yes_bid": data.get("yes_bid"),
            "yes_ask": data.get("yes_ask"),
            "last_price": data.get("last_price"),
            "volume": data.get("volume"),
            "open_interest": data.get("open_interest"),
            "close_time": data.get("close_time", datetime.now(tz=timezone.utc).isoformat()),
            "result": data.get("result"),
            "category": data.get("category"),
        }
        async with self._lock:
            self._buffer.market_states.append(row)
        await self._maybe_flush()

    async def on_orderbook_update(self, event: WSEvent | dict[str, Any]) -> None:
        """Handle an orderbook delta / snapshot from the WebSocket.

        Snapshots are rate-limited per ticker to avoid excessive writes.
        """
        data = event["data"] if isinstance(event, dict) else event.data
        ticker = data["ticker"]

        now = time.monotonic()
        last = self._last_orderbook_time.get(ticker, 0.0)
        if now - last < self._orderbook_interval:
            return

        row = {
            "ticker": ticker,
            "yes_levels": data.get("yes_levels", []),
            "no_levels": data.get("no_levels", []),
            "seq": data.get("seq"),
            "snapshot_time": data.get(
                "snapshot_time",
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        }
        async with self._lock:
            self._buffer.orderbook_snapshots.append(row)
            self._last_orderbook_time[ticker] = now
        await self._maybe_flush()

    async def on_trade(self, event: WSEvent | dict[str, Any]) -> None:
        """Handle a trade event from the WebSocket."""
        data = event["data"] if isinstance(event, dict) else event.data
        row = {
            "trade_id": data["trade_id"],
            "ticker": data["ticker"],
            "count": data.get("count", 1),
            "yes_price": data["yes_price"],
            "taker_side": data.get("taker_side", "unknown"),
            "trade_time": data.get(
                "trade_time",
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        }
        async with self._lock:
            self._buffer.trades.append(row)
        await self._maybe_flush()

    # ------------------------------------------------------------------
    # Flush logic
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Flush all buffered events to the DataStore."""
        async with self._lock:
            if self._buffer.total == 0:
                return
            market_states = list(self._buffer.market_states)
            orderbook_snapshots = list(self._buffer.orderbook_snapshots)
            trades = list(self._buffer.trades)
            self._buffer.clear()
            self._last_flush_time = time.monotonic()

        # Run inserts outside the lock so new events can buffer concurrently.
        try:
            if market_states:
                self._store.insert_market_states(market_states)
            if orderbook_snapshots:
                self._store.insert_orderbook_snapshots(orderbook_snapshots)
            if trades:
                self._store.insert_trades(trades)
            total = len(market_states) + len(orderbook_snapshots) + len(trades)
            logger.debug("market_data_recorder.flushed", total=total)
        except Exception:
            logger.exception("market_data_recorder.flush_failed")
            # Re-buffer on failure so data is not lost.
            async with self._lock:
                self._buffer.market_states = market_states + self._buffer.market_states
                self._buffer.orderbook_snapshots = (
                    orderbook_snapshots + self._buffer.orderbook_snapshots
                )
                self._buffer.trades = trades + self._buffer.trades

    async def _maybe_flush(self) -> None:
        elapsed = time.monotonic() - self._last_flush_time
        if (
            self._buffer.total >= self._flush_threshold
            or elapsed >= self._flush_interval
        ):
            await self.flush()

    async def _periodic_flush(self) -> None:
        """Background loop that flushes at the configured interval."""
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.flush()
