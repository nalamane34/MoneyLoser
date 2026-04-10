"""Crypto volatility data feeds: realized vol, implied vol (Deribit DVOL), ATR, BRTI proxy.

Data sources:
- Satochi (btcvol.info): free realized volatility, no auth required
- Deribit API: free DVOL (implied volatility index), no auth for public data
- CCXT OHLCV: for computing ATR and realized vol locally from candle data
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt_async = None  # type: ignore[assignment]

import httpx

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VolatilitySnapshot:
    """Point-in-time volatility metrics for a crypto asset."""

    symbol: str
    timestamp: datetime
    realized_vol_24h: float | None  # annualized, from 24h returns
    realized_vol_7d: float | None  # annualized, from 7d returns
    realized_vol_30d: float | None  # annualized, from 30d returns
    implied_vol: float | None  # Deribit DVOL or proxy
    atr_14: float | None  # 14-period ATR (hourly)
    atr_24: float | None  # 24-period ATR (hourly)
    trend_regime: str | None  # "strong_up", "up", "neutral", "down", "strong_down"
    trend_strength: float | None  # 0.0 to 1.0
    brti_price: float | None  # BRTI or proxy (CME reference rate)


@dataclass(frozen=True, slots=True)
class OHLCVCandle:
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Volatility Feed
# ---------------------------------------------------------------------------


class CryptoVolatilityFeed:
    """Fetches and computes crypto volatility metrics from multiple sources.

    Usage::

        feed = CryptoVolatilityFeed()
        snapshot = await feed.get_volatility("BTC/USDT")
    """

    def __init__(
        self,
        exchange_id: str = "binanceus",
        deribit_enabled: bool = True,
    ) -> None:
        self._exchange_id = exchange_id
        self._exchange: Any | None = None
        self._deribit_enabled = deribit_enabled
        self._http: httpx.AsyncClient | None = None

    async def _ensure_exchange(self) -> Any:
        if self._exchange is None and ccxt_async is not None:
            exchange_class = getattr(ccxt_async, self._exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    async def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def get_volatility(self, symbol: str = "BTC/USDT") -> VolatilitySnapshot:
        """Fetch all volatility metrics for a symbol."""
        # Fetch OHLCV candles and external data concurrently
        results = await asyncio.gather(
            self._fetch_ohlcv(symbol, timeframe="1h", limit=168),  # 7 days hourly
            self._fetch_deribit_dvol() if self._deribit_enabled else asyncio.sleep(0),
            self._fetch_satochi_vol(),
            self._fetch_brti_proxy(symbol),
            return_exceptions=True,
        )

        candles = results[0] if not isinstance(results[0], Exception) else []
        dvol = results[1] if not isinstance(results[1], Exception) else None
        satochi = results[2] if not isinstance(results[2], Exception) else None
        brti = results[3] if not isinstance(results[3], Exception) else None

        # Compute from candles
        realized_24h = None
        realized_7d = None
        realized_30d = None
        atr_14 = None
        atr_24 = None
        trend_regime = None
        trend_strength = None

        if candles and len(candles) >= 24:
            closes = np.array([c.close for c in candles])
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])

            realized_24h = self._compute_realized_vol(closes[-24:], periods_per_year=8760)
            if len(closes) >= 168:
                realized_7d = self._compute_realized_vol(closes[-168:], periods_per_year=8760)

            if len(candles) >= 14:
                atr_14 = self._compute_atr(highs, lows, closes, period=14)
            if len(candles) >= 24:
                atr_24 = self._compute_atr(highs, lows, closes, period=24)

            trend_regime, trend_strength = self._compute_trend_regime(closes)

        # Use satochi for 30d realized vol if available
        if satochi and isinstance(satochi, dict):
            realized_30d = satochi.get("vol_30d")

        # Implied vol from Deribit DVOL
        implied_vol = None
        if isinstance(dvol, (int, float)):
            implied_vol = float(dvol)

        return VolatilitySnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            realized_vol_24h=realized_24h,
            realized_vol_7d=realized_7d,
            realized_vol_30d=realized_30d,
            implied_vol=implied_vol,
            atr_14=atr_14,
            atr_24=atr_24,
            trend_regime=trend_regime,
            trend_strength=trend_strength,
            brti_price=brti,
        )

    # ------------------------------------------------------------------
    # OHLCV from exchange
    # ------------------------------------------------------------------

    async def _fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 168,
    ) -> list[OHLCVCandle]:
        """Fetch OHLCV candles via CCXT."""
        exchange = await self._ensure_exchange()
        if exchange is None:
            return []

        try:
            raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            log.warning("ohlcv_fetch_failed", symbol=symbol, exc_info=True)
            return []

        candles = []
        for ts_ms, o, h, l, c, v in raw:
            candles.append(OHLCVCandle(
                timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                open=float(o), high=float(h), low=float(l),
                close=float(c), volume=float(v),
            ))
        return candles

    # ------------------------------------------------------------------
    # Deribit DVOL (implied volatility)
    # ------------------------------------------------------------------

    async def _fetch_deribit_dvol(self) -> float | None:
        """Fetch Deribit DVOL (30-day implied vol index) for BTC.

        Uses public API — no auth required.
        Endpoint: get_volatility_index_data returns OHLC of DVOL.
        We take the most recent close value.
        """
        http = await self._ensure_http()
        try:
            import time as _time
            end_ts = int(_time.time() * 1000)
            start_ts = end_ts - 3600_000  # last hour

            r = await http.get(
                "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                params={
                    "currency": "BTC",
                    "resolution": "3600",
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                },
            )
            if r.status_code == 200:
                data = r.json()
                points = data.get("result", {}).get("data", [])
                if points:
                    # Each point is [timestamp, open, high, low, close]
                    latest = points[-1]
                    dvol_close = float(latest[4])  # close value
                    # DVOL is in vol points (e.g., 52.5 = 52.5%)
                    return dvol_close / 100.0
        except Exception:
            log.warning("deribit_dvol_failed", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Satochi realized volatility
    # ------------------------------------------------------------------

    async def _fetch_satochi_vol(self) -> dict | None:
        """Fetch BTC realized volatility from btcvol.info (free, no auth)."""
        http = await self._ensure_http()
        try:
            r = await http.get("https://btcvol.info/latest")
            if r.status_code == 200:
                data = r.json()
                # Returns 30-day and 60-day realized vol
                return {
                    "vol_30d": float(data.get("Volatility", 0)) / 100.0 if data.get("Volatility") else None,
                }
        except Exception:
            log.warning("satochi_vol_failed", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # BRTI proxy (reference price)
    # ------------------------------------------------------------------

    async def _fetch_brti_proxy(self, symbol: str = "BTC/USDT") -> float | None:
        """BRTI is not freely available. Use exchange mid-price as proxy.

        In production, this could be upgraded to use a CME data feed
        or CF Benchmarks API with a license.
        """
        exchange = await self._ensure_exchange()
        if exchange is None:
            return None

        try:
            ticker = await exchange.fetch_ticker(symbol)
            bid = ticker.get("bid", 0) or 0
            ask = ticker.get("ask", 0) or 0
            if bid and ask:
                return (float(bid) + float(ask)) / 2.0
            return float(ticker.get("last", 0) or 0) or None
        except Exception:
            log.warning("brti_proxy_failed", symbol=symbol, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_realized_vol(
        closes: np.ndarray, periods_per_year: int = 8760,
    ) -> float:
        """Annualized realized volatility from log returns.

        Args:
            closes: array of close prices
            periods_per_year: 8760 for hourly, 365 for daily
        """
        if len(closes) < 2:
            return 0.0
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Average True Range (ATR) as percentage of price.

        Returns ATR normalized by current close price for cross-asset comparability.
        """
        if len(highs) < period + 1:
            return 0.0

        # True range components
        tr1 = highs[1:] - lows[1:]  # high - low
        tr2 = np.abs(highs[1:] - closes[:-1])  # |high - prev_close|
        tr3 = np.abs(lows[1:] - closes[:-1])  # |low - prev_close|
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # Simple moving average of TR for the last `period` bars
        atr_abs = float(np.mean(true_range[-period:]))

        # Normalize by current price
        current_price = float(closes[-1])
        if current_price == 0:
            return 0.0
        return atr_abs / current_price

    @staticmethod
    def _compute_trend_regime(
        closes: np.ndarray,
    ) -> tuple[str, float]:
        """Classify trend regime using multi-timeframe momentum.

        Uses 8h, 24h, and 72h returns to determine regime.
        Returns (regime_label, strength 0-1).
        """
        if len(closes) < 72:
            return "neutral", 0.5

        # Returns at different horizons
        ret_8h = (closes[-1] / closes[-8] - 1) if closes[-8] != 0 else 0
        ret_24h = (closes[-1] / closes[-24] - 1) if closes[-24] != 0 else 0
        ret_72h = (closes[-1] / closes[-72] - 1) if closes[-72] != 0 else 0

        # Composite score: weight shorter timeframes more
        score = 0.5 * ret_8h + 0.3 * ret_24h + 0.2 * ret_72h

        # Classify
        if score > 0.03:
            regime = "strong_up"
        elif score > 0.01:
            regime = "up"
        elif score > -0.01:
            regime = "neutral"
        elif score > -0.03:
            regime = "down"
        else:
            regime = "strong_down"

        # Strength: abs(score) normalized, capped at 1
        strength = min(abs(score) / 0.05, 1.0)

        return regime, float(strength)
