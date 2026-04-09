"""Crypto market data feed using ccxt for exchange connectivity.

Provides unified access to funding rates, open interest, order books, and
recent trades across multiple exchanges.  All public methods are async and
delegate to :mod:`ccxt.async_support`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import ccxt.async_support as ccxt_async
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FundingRate:
    """A single funding-rate observation."""

    exchange: str
    symbol: str
    rate: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class OpenInterestSnapshot:
    """Open-interest snapshot from an exchange."""

    exchange: str
    symbol: str
    value: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class CryptoOrderbook:
    """Snapshot of a limit order book."""

    exchange: str
    symbol: str
    bids: list[list[float]]  # [[price, size], ...]
    asks: list[list[float]]
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class CryptoTrade:
    """A single public trade."""

    exchange: str
    symbol: str
    trade_id: str
    price: float
    amount: float
    side: str  # "buy" | "sell"
    timestamp: datetime


# ---------------------------------------------------------------------------
# CryptoDataFeed
# ---------------------------------------------------------------------------


class CryptoDataFeed:
    """Async crypto data feed backed by ccxt.

    Parameters
    ----------
    exchange_ids:
        List of ccxt exchange identifiers (e.g. ``["binance"]``).
    sandbox:
        Whether to use exchange sandboxes (testnet).
    """

    def __init__(
        self,
        exchange_ids: list[str] | None = None,
        *,
        sandbox: bool = False,
    ) -> None:
        self._exchange_ids = exchange_ids or ["binance"]
        self._sandbox = sandbox
        self._exchanges: dict[str, ccxt_async.Exchange] = {}
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_exchange(self, exchange_id: str) -> ccxt_async.Exchange:
        """Return a connected exchange instance, creating if needed."""
        if exchange_id not in self._exchanges:
            cls = getattr(ccxt_async, exchange_id)
            ex: ccxt_async.Exchange = cls({"enableRateLimit": True})
            if self._sandbox:
                ex.set_sandbox_mode(True)
            await ex.load_markets()
            self._exchanges[exchange_id] = ex
            logger.info("crypto_feed.exchange_loaded", exchange=exchange_id)
        return self._exchanges[exchange_id]

    async def close(self) -> None:
        """Close all exchange connections and cancel streaming."""
        self._streaming = False
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        for ex in self._exchanges.values():
            await ex.close()
        self._exchanges.clear()
        logger.info("crypto_feed.closed")

    # ------------------------------------------------------------------
    # Public data methods
    # ------------------------------------------------------------------

    async def get_funding_rates(
        self, symbols: list[str]
    ) -> list[FundingRate]:
        """Fetch the latest funding rates for the given symbols.

        Each symbol is queried on every configured exchange that supports
        ``fetchFundingRate``.
        """
        results: list[FundingRate] = []
        for exchange_id in self._exchange_ids:
            ex = await self._ensure_exchange(exchange_id)
            if not ex.has.get("fetchFundingRate"):
                logger.debug(
                    "crypto_feed.no_funding_rate_support", exchange=exchange_id
                )
                continue
            for symbol in symbols:
                try:
                    data: dict[str, Any] = await ex.fetch_funding_rate(symbol)
                    rate = data.get("fundingRate")
                    if rate is None:
                        continue
                    ts = data.get("fundingTimestamp") or data.get("timestamp")
                    results.append(
                        FundingRate(
                            exchange=exchange_id,
                            symbol=symbol,
                            rate=float(rate),
                            timestamp=_ts_to_dt(ts),
                        )
                    )
                except Exception:
                    logger.warning(
                        "crypto_feed.funding_rate_error",
                        exchange=exchange_id,
                        symbol=symbol,
                        exc_info=True,
                    )
        return results

    async def get_open_interest(
        self, symbols: list[str]
    ) -> list[OpenInterestSnapshot]:
        """Fetch open interest for the given symbols."""
        results: list[OpenInterestSnapshot] = []
        for exchange_id in self._exchange_ids:
            ex = await self._ensure_exchange(exchange_id)
            if not ex.has.get("fetchOpenInterest"):
                continue
            for symbol in symbols:
                try:
                    data: dict[str, Any] = await ex.fetch_open_interest(symbol)
                    oi = data.get("openInterestAmount") or data.get("openInterest")
                    if oi is None:
                        continue
                    ts = data.get("timestamp")
                    results.append(
                        OpenInterestSnapshot(
                            exchange=exchange_id,
                            symbol=symbol,
                            value=float(oi),
                            timestamp=_ts_to_dt(ts),
                        )
                    )
                except Exception:
                    logger.warning(
                        "crypto_feed.open_interest_error",
                        exchange=exchange_id,
                        symbol=symbol,
                        exc_info=True,
                    )
        return results

    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        exchange_id: str | None = None,
    ) -> CryptoOrderbook:
        """Fetch a limit order book snapshot.

        Parameters
        ----------
        symbol:
            Market symbol (e.g. ``"BTC/USDT"``).
        depth:
            Number of price levels per side.
        exchange_id:
            Specific exchange to query; defaults to the first configured.
        """
        eid = exchange_id or self._exchange_ids[0]
        ex = await self._ensure_exchange(eid)
        ob: dict[str, Any] = await ex.fetch_order_book(symbol, limit=depth)
        return CryptoOrderbook(
            exchange=eid,
            symbol=symbol,
            bids=ob.get("bids", []),
            asks=ob.get("asks", []),
            timestamp=_ts_to_dt(ob.get("timestamp")),
        )

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
        exchange_id: str | None = None,
    ) -> list[CryptoTrade]:
        """Fetch recent public trades for a symbol."""
        eid = exchange_id or self._exchange_ids[0]
        ex = await self._ensure_exchange(eid)
        raw_trades: list[dict[str, Any]] = await ex.fetch_trades(symbol, limit=limit)
        return [
            CryptoTrade(
                exchange=eid,
                symbol=symbol,
                trade_id=str(t.get("id", "")),
                price=float(t["price"]),
                amount=float(t["amount"]),
                side=t.get("side", "unknown"),
                timestamp=_ts_to_dt(t.get("timestamp")),
            )
            for t in raw_trades
        ]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def start_streaming(
        self,
        symbols: list[str],
        callback: Callable[
            [str, str, Any], Coroutine[Any, Any, None]
        ],
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        """Poll data at fixed intervals and invoke *callback*.

        The callback signature is ``callback(event_type, symbol, data)``
        where ``event_type`` is one of ``"funding_rate"``,
        ``"open_interest"``, ``"trades"``.

        Polling is used instead of native WebSocket streams because ccxt
        watch methods are not universally available.
        """
        self._streaming = True

        async def _loop() -> None:
            while self._streaming:
                for symbol in symbols:
                    try:
                        funding = await self.get_funding_rates([symbol])
                        for fr in funding:
                            await callback("funding_rate", symbol, fr)
                    except Exception:
                        logger.warning(
                            "crypto_feed.stream_funding_error",
                            symbol=symbol,
                            exc_info=True,
                        )

                    try:
                        oi = await self.get_open_interest([symbol])
                        for snap in oi:
                            await callback("open_interest", symbol, snap)
                    except Exception:
                        logger.warning(
                            "crypto_feed.stream_oi_error",
                            symbol=symbol,
                            exc_info=True,
                        )

                    try:
                        trades = await self.get_recent_trades(symbol, limit=50)
                        for trade in trades:
                            await callback("trades", symbol, trade)
                    except Exception:
                        logger.warning(
                            "crypto_feed.stream_trades_error",
                            symbol=symbol,
                            exc_info=True,
                        )
                await asyncio.sleep(interval_seconds)

        self._stream_task = asyncio.create_task(_loop())
        logger.info("crypto_feed.streaming_started", symbols=symbols)

    async def stop_streaming(self) -> None:
        """Stop the polling loop started by :meth:`start_streaming`."""
        self._streaming = False
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        logger.info("crypto_feed.streaming_stopped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_to_dt(ts: int | float | None) -> datetime:
    """Convert a millisecond-epoch timestamp to a UTC datetime."""
    if ts is None:
        return datetime.now(tz=timezone.utc)
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
