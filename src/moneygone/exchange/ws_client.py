"""WebSocket client for Kalshi real-time market data.

Handles authenticated connections, channel subscriptions, automatic
reconnection with exponential backoff, and local orderbook reconstruction
from snapshot + delta messages.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable

import certifi
import structlog
import websockets
import websockets.exceptions

from moneygone.config import ExchangeConfig
from moneygone.exchange.auth import KalshiAuth
from moneygone.exchange.errors import WebSocketError
from moneygone.exchange.types import (
    OrderbookLevel,
    OrderbookSnapshot,
    WSEvent,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EventCallback = Callable[[WSEvent], Awaitable[None] | None]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INITIAL_RECONNECT_DELAY = 1.0
_MAX_RECONNECT_DELAY = 60.0
_RECONNECT_BACKOFF_FACTOR = 2.0
_PING_INTERVAL = 20  # seconds
_PING_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Local orderbook
# ---------------------------------------------------------------------------


class _LocalOrderbook:
    """Maintains a local orderbook from snapshot + delta messages.

    Detects sequence gaps so the caller can re-subscribe for a fresh snapshot.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes_levels: dict[Decimal, Decimal] = {}
        self.no_levels: dict[Decimal, Decimal] = {}
        self.seq: int = -1
        self.last_update: float = 0.0

    def apply_snapshot(self, data: dict[str, Any], seq: int) -> None:
        """Replace the entire book from a snapshot message.

        Handles both legacy format (``yes``/``no``) and real API format
        (``yes_dollars``/``no_dollars`` inside ``orderbook_fp``).
        """
        ob = data.get("orderbook_fp", data)
        yes_raw = ob.get("yes_dollars", data.get("yes", []))
        no_raw = ob.get("no_dollars", data.get("no", []))

        self.yes_levels = {
            Decimal(str(lvl[0])): Decimal(str(lvl[1]))
            for lvl in (yes_raw or [])
        }
        self.no_levels = {
            Decimal(str(lvl[0])): Decimal(str(lvl[1]))
            for lvl in (no_raw or [])
        }
        self.seq = seq
        self.last_update = time.monotonic()

    def apply_delta(self, data: dict[str, Any], seq: int) -> bool:
        """Apply an incremental delta.  Returns ``False`` on sequence gap."""
        if self.seq == -1:
            return False

        if seq != self.seq + 1:
            log.warning(
                "orderbook.seq_gap",
                ticker=self.ticker,
                expected=self.seq + 1,
                received=seq,
            )
            return False

        ob = data.get("orderbook_fp", data)
        yes_raw = ob.get("yes_dollars", data.get("yes", []))
        no_raw = ob.get("no_dollars", data.get("no", []))

        for lvl in (yes_raw or []):
            price = Decimal(str(lvl[0]))
            contracts = Decimal(str(lvl[1]))
            if contracts == 0:
                self.yes_levels.pop(price, None)
            else:
                self.yes_levels[price] = contracts

        for lvl in (no_raw or []):
            price = Decimal(str(lvl[0]))
            contracts = Decimal(str(lvl[1]))
            if contracts == 0:
                self.no_levels.pop(price, None)
            else:
                self.no_levels[price] = contracts

        self.seq = seq
        self.last_update = time.monotonic()
        return True

    def snapshot(self) -> OrderbookSnapshot:
        """Return the current local state as an ``OrderbookSnapshot``."""
        return OrderbookSnapshot(
            ticker=self.ticker,
            yes_bids=tuple(
                OrderbookLevel(price=p, contracts=c)
                for p, c in sorted(self.yes_levels.items())
            ),
            no_bids=tuple(
                OrderbookLevel(price=p, contracts=c)
                for p, c in sorted(self.no_levels.items())
            ),
            seq=self.seq,
            timestamp=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


class KalshiWebSocket:
    """Async WebSocket client for Kalshi real-time feeds.

    Usage::

        ws = KalshiWebSocket(config, on_event=my_handler)
        await ws.connect()
        await ws.subscribe_orderbook(["TICKER-A", "TICKER-B"])
        # ... ws.disconnect() when done
    """

    def __init__(
        self,
        config: ExchangeConfig,
        on_event: EventCallback | None = None,
    ) -> None:
        self._config = config
        self._ws_url = config.ws_url.rstrip("/")
        self._auth = KalshiAuth(config.api_key_id, config.private_key_path)
        self._on_event = on_event

        self._ws: websockets.WebSocketClientProtocol | None = None  # type: ignore[name-defined]
        self._recv_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._should_run = False
        self._reconnect_delay = _INITIAL_RECONNECT_DELAY

        # Subscription tracking: sid -> subscription command dict
        self._subscriptions: dict[int, dict[str, Any]] = {}
        self._next_sid: int = 1

        # Local orderbooks keyed by ticker
        self._orderbooks: dict[str, _LocalOrderbook] = {}

        # Channels that need re-subscribe after reconnect
        self._active_channels: dict[str, list[str]] = defaultdict(list)

        log.info("ws_client.initialized", ws_url=self._ws_url)

    def set_on_event(self, handler: EventCallback | None) -> None:
        """Update the event callback used for dispatched WebSocket messages."""
        self._on_event = handler

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the WebSocket connection and start the receive loop."""
        self._should_run = True
        await self._do_connect()
        self._recv_task = asyncio.create_task(self._recv_loop())
        log.info("ws_client.connected")

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._should_run = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected.clear()
        log.info("ws_client.disconnected")

    async def _do_connect(self) -> None:
        """Open the authenticated WebSocket connection."""
        headers = self._auth.get_headers("GET", "/trade-api/ws/v2")

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                additional_headers=headers,
                ssl=ssl_ctx,
                ping_interval=_PING_INTERVAL,
                ping_timeout=_PING_TIMEOUT,
                max_size=10 * 1024 * 1024,  # 10 MiB
            )
            self._connected.set()
            self._reconnect_delay = _INITIAL_RECONNECT_DELAY
        except Exception as exc:
            raise WebSocketError(f"Failed to connect: {exc}") from exc

    # ------------------------------------------------------------------
    # Receive loop & reconnection
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Receive messages, dispatch events, reconnect on failure."""
        while self._should_run:
            try:
                async for raw_msg in self._ws:  # type: ignore[union-attr]
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        log.warning("ws_client.invalid_json", raw=raw_msg[:200])
                        continue
                    await self._handle_message(msg)

            except websockets.exceptions.ConnectionClosed as exc:
                log.warning(
                    "ws_client.connection_closed",
                    code=exc.code,
                    reason=exc.reason,
                )
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("ws_client.recv_error")

            if not self._should_run:
                return

            # Reconnect with exponential backoff
            self._connected.clear()
            log.info(
                "ws_client.reconnecting",
                delay=self._reconnect_delay,
            )
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * _RECONNECT_BACKOFF_FACTOR,
                _MAX_RECONNECT_DELAY,
            )

            try:
                await self._do_connect()
                await self._resubscribe_all()
            except WebSocketError:
                log.warning("ws_client.reconnect_failed")
                continue

    async def _resubscribe_all(self) -> None:
        """Re-send all active subscriptions after reconnect."""
        for sid, cmd in list(self._subscriptions.items()):
            await self._send(cmd)
        # Reset orderbook state -- need fresh snapshots
        for ob in self._orderbooks.values():
            ob.seq = -1
        log.info(
            "ws_client.resubscribed",
            count=len(self._subscriptions),
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route an incoming message to the appropriate handler."""
        msg_type = msg.get("type", "")
        channel = msg.get("channel", "")
        data = msg.get("msg", msg.get("data", {}))
        seq = int(msg.get("seq", 0))
        ts_str = msg.get("timestamp")

        ts: datetime | None = None
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = None

        # Orderbook-specific processing
        if channel == "orderbook_delta":
            ticker = data.get("market_ticker", "")
            if msg_type == "orderbook_snapshot":
                ob = self._orderbooks.setdefault(ticker, _LocalOrderbook(ticker))
                ob.apply_snapshot(data, seq)
                log.debug("ws_client.orderbook_snapshot", ticker=ticker, seq=seq)
            elif msg_type == "orderbook_delta":
                ob = self._orderbooks.get(ticker)
                if ob is None:
                    log.warning("ws_client.delta_no_book", ticker=ticker)
                elif not ob.apply_delta(data, seq):
                    # Sequence gap detected -- need fresh snapshot.
                    # The server should send one automatically, but log it.
                    log.warning("ws_client.orderbook_gap", ticker=ticker)

        # Dispatch to user callback
        event = WSEvent(
            channel=channel,
            type=msg_type,
            data=data,
            seq=seq,
            timestamp=ts,
        )
        if self._on_event is not None:
            result = self._on_event(event)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Send helper
    # ------------------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        """Serialize and send a JSON message."""
        if self._ws is None:
            raise WebSocketError("Not connected")
        await self._ws.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _alloc_sid(self) -> int:
        sid = self._next_sid
        self._next_sid += 1
        return sid

    async def subscribe_orderbook(self, tickers: list[str]) -> int:
        """Subscribe to orderbook snapshots and deltas for the given tickers."""
        sid = self._alloc_sid()
        cmd: dict[str, Any] = {
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }
        for t in tickers:
            if t not in self._orderbooks:
                self._orderbooks[t] = _LocalOrderbook(t)
        self._subscriptions[sid] = cmd
        await self._send(cmd)
        log.info("ws_client.subscribed_orderbook", tickers=tickers, sid=sid)
        return sid

    async def subscribe_ticker(self, tickers: list[str]) -> int:
        """Subscribe to ticker-level updates (price, volume, status)."""
        sid = self._alloc_sid()
        cmd: dict[str, Any] = {
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": tickers,
            },
        }
        self._subscriptions[sid] = cmd
        await self._send(cmd)
        log.info("ws_client.subscribed_ticker", tickers=tickers, sid=sid)
        return sid

    async def subscribe_trades(self, tickers: list[str]) -> int:
        """Subscribe to public trade events."""
        sid = self._alloc_sid()
        cmd: dict[str, Any] = {
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["trade"],
                "market_tickers": tickers,
            },
        }
        self._subscriptions[sid] = cmd
        await self._send(cmd)
        log.info("ws_client.subscribed_trades", tickers=tickers, sid=sid)
        return sid

    async def subscribe_fills(self) -> int:
        """Subscribe to personal fill notifications."""
        sid = self._alloc_sid()
        cmd: dict[str, Any] = {
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
            },
        }
        self._subscriptions[sid] = cmd
        await self._send(cmd)
        log.info("ws_client.subscribed_fills", sid=sid)
        return sid

    async def subscribe_positions(self) -> int:
        """Subscribe to position update notifications."""
        sid = self._alloc_sid()
        cmd: dict[str, Any] = {
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["market_position"],
            },
        }
        self._subscriptions[sid] = cmd
        await self._send(cmd)
        log.info("ws_client.subscribed_positions", sid=sid)
        return sid

    async def unsubscribe(self, sid: int) -> None:
        """Unsubscribe a previously created subscription by its ID."""
        cmd = self._subscriptions.pop(sid, None)
        if cmd is None:
            log.warning("ws_client.unsubscribe_unknown", sid=sid)
            return

        unsub: dict[str, Any] = {
            "id": sid,
            "cmd": "unsubscribe",
            "params": cmd.get("params", {}),
        }
        await self._send(unsub)
        log.info("ws_client.unsubscribed", sid=sid)

    # ------------------------------------------------------------------
    # Orderbook access
    # ------------------------------------------------------------------

    def get_orderbook(self, ticker: str) -> OrderbookSnapshot | None:
        """Return the latest locally-reconstructed orderbook, or ``None``."""
        ob = self._orderbooks.get(ticker)
        if ob is None or ob.seq == -1:
            return None
        return ob.snapshot()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    async def wait_connected(self, timeout: float = 30.0) -> None:
        """Block until the connection is established or *timeout* expires."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise WebSocketError(
                f"Connection not established within {timeout}s"
            ) from None
