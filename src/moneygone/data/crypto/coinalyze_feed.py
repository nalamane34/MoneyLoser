"""Crypto futures data feed via Coinalyze API.

Provides funding rates, open interest, and liquidation data aggregated
across exchanges.  Free tier, no geo-restrictions, 40 req/min limit.

API docs: https://coinalyze.net (API key required, free signup).

This module implements the same interface as ``CryptoDataFeed`` so it can
be used as a drop-in replacement for the ccxt-based feed's futures data
methods (funding rates + OI).  Spot price/OHLCV still comes from ccxt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from moneygone.data.crypto.ccxt_feed import FundingRate, OpenInterestSnapshot

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.coinalyze.net/v1"

# Map our symbols (BTC/USDT) to Coinalyze perp symbol format.
# Coinalyze uses BTCUSDT_PERP.A (A=Binance), but we can query
# aggregated data without exchange suffix for some endpoints,
# or use multiple exchanges.
_SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BTCUSDT_PERP.A",
    "ETH/USDT": "ETHUSDT_PERP.A",
    "SOL/USDT": "SOLUSDT_PERP.A",
    "DOGE/USDT": "DOGEUSDT_PERP.A",
    "BNB/USDT": "BNBUSDT_PERP.A",
    "XRP/USDT": "XRPUSDT_PERP.A",
    "ADA/USDT": "ADAUSDT_PERP.A",
    "AVAX/USDT": "AVAXUSDT_PERP.A",
    "LTC/USDT": "LTCUSDT_PERP.A",
}


@dataclass(frozen=True, slots=True)
class LiquidationSnapshot:
    """Aggregated liquidation data."""

    symbol: str
    long_liquidations: float
    short_liquidations: float
    timestamp: datetime


class CoinalyzeFeed:
    """Crypto futures data from Coinalyze API.

    Provides funding rates and open interest using the same return types
    as ``CryptoDataFeed``, so the ``CryptoDataProvider`` can use either
    interchangeably.

    Parameters
    ----------
    api_key:
        Coinalyze API key (free signup at coinalyze.net).
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._markets_cache: dict[str, str] | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers={"accept": "application/json"},
            )
        return self._client

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        client = await self._get_client()
        p = dict(params or {})
        p["api_key"] = self._api_key
        resp = await client.get(f"{_BASE_URL}{endpoint}", params=p)
        resp.raise_for_status()
        return resp.json()

    def _resolve_symbol(self, symbol: str) -> str:
        """Convert our symbol format to Coinalyze format."""
        return _SYMBOL_MAP.get(symbol, symbol.replace("/", "") + "_PERP.A")

    # ------------------------------------------------------------------
    # Funding rates
    # ------------------------------------------------------------------

    async def get_funding_rates(self, symbols: list[str]) -> list[FundingRate]:
        """Fetch current funding rates for the given symbols."""
        results: list[FundingRate] = []
        ca_symbols = [self._resolve_symbol(s) for s in symbols]

        try:
            data = await self._request(
                "/funding-rate",
                {"symbols": ",".join(ca_symbols)},
            )
            for item in data:
                ca_sym = item.get("symbol", "")
                # Reverse-map back to our symbol format
                orig_sym = self._reverse_symbol(ca_sym, symbols)
                rate = item.get("value")
                if rate is None:
                    continue
                ts = item.get("timestamp", 0)
                results.append(FundingRate(
                    exchange="coinalyze",
                    symbol=orig_sym,
                    rate=float(rate),
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                ))
        except Exception:
            logger.warning("coinalyze.funding_rate_error", exc_info=True)

        return results

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def get_open_interest(self, symbols: list[str]) -> list[OpenInterestSnapshot]:
        """Fetch current open interest for the given symbols."""
        results: list[OpenInterestSnapshot] = []
        ca_symbols = [self._resolve_symbol(s) for s in symbols]

        try:
            data = await self._request(
                "/open-interest",
                {"symbols": ",".join(ca_symbols)},
            )
            for item in data:
                ca_sym = item.get("symbol", "")
                orig_sym = self._reverse_symbol(ca_sym, symbols)
                oi = item.get("value")
                if oi is None:
                    continue
                ts = item.get("timestamp", 0)
                results.append(OpenInterestSnapshot(
                    exchange="coinalyze",
                    symbol=orig_sym,
                    value=float(oi),
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                ))
        except Exception:
            logger.warning("coinalyze.open_interest_error", exc_info=True)

        return results

    # ------------------------------------------------------------------
    # Liquidations
    # ------------------------------------------------------------------

    async def get_liquidations(
        self,
        symbols: list[str],
        interval: str = "1hour",
    ) -> list[LiquidationSnapshot]:
        """Fetch recent liquidation data."""
        results: list[LiquidationSnapshot] = []
        ca_symbols = [self._resolve_symbol(s) for s in symbols]

        try:
            data = await self._request(
                "/liquidation-history",
                {
                    "symbols": ",".join(ca_symbols),
                    "interval": interval,
                    "limit": 1,
                },
            )
            for item in data:
                ca_sym = item.get("symbol", "")
                orig_sym = self._reverse_symbol(ca_sym, symbols)
                history = item.get("history", [])
                if not history:
                    continue
                latest = history[-1]
                results.append(LiquidationSnapshot(
                    symbol=orig_sym,
                    long_liquidations=float(latest.get("l", 0)),
                    short_liquidations=float(latest.get("s", 0)),
                    timestamp=datetime.fromtimestamp(
                        latest.get("t", 0), tz=timezone.utc
                    ),
                ))
        except Exception:
            logger.warning("coinalyze.liquidation_error", exc_info=True)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reverse_symbol(self, ca_symbol: str, original_symbols: list[str]) -> str:
        """Map a Coinalyze symbol back to our format."""
        for orig, ca in _SYMBOL_MAP.items():
            if ca == ca_symbol and orig in original_symbols:
                return orig
        # Fallback: strip _PERP.X and re-insert /
        base = ca_symbol.split("_")[0]
        if base.endswith("USDT"):
            return base[:-4] + "/USDT"
        return ca_symbol

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
