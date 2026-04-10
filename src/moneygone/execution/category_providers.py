"""Category-specific data providers for the execution engine.

Each provider is an async callable that takes a Market and returns a dict
suitable for populating FeatureContext fields (crypto_snapshot,
weather_ensemble, etc.).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import structlog

from moneygone.exchange.types import Market

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Threshold extraction from Kalshi market titles
# ---------------------------------------------------------------------------

_PRICE_PATTERN = re.compile(
    r"\$?([\d,]+(?:\.\d+)?)\s*(?:or\s+(?:above|more|higher|below|less|lower))?",
    re.IGNORECASE,
)

_ABOVE_PATTERN = re.compile(r"above|over|higher|more than|at least|reach|exceed", re.IGNORECASE)
_BELOW_PATTERN = re.compile(r"below|under|lower|less than|at most|drop", re.IGNORECASE)

_SYMBOL_MAP = {
    "bitcoin": "BTC/USDT",
    "btc": "BTC/USDT",
    "ethereum": "ETH/USDT",
    "eth": "ETH/USDT",
    "solana": "SOL/USDT",
    "sol": "SOL/USDT",
    "dogecoin": "DOGE/USDT",
    "doge": "DOGE/USDT",
    "xrp": "XRP/USDT",
    "ripple": "XRP/USDT",
    "hype": "HYPE/USDT",
    "hyperliquid": "HYPE/USDT",
    "bnb": "BNB/USDT",
    "cardano": "ADA/USDT",
    "ada": "ADA/USDT",
    "avalanche": "AVAX/USDT",
    "avax": "AVAX/USDT",
    "litecoin": "LTC/USDT",
    "ltc": "LTC/USDT",
}


def _extract_threshold(market: Market) -> float | None:
    """Extract the price/value threshold from a market title."""
    text = " ".join(
        v for v in [market.title, market.subtitle, market.yes_sub_title] if v
    )
    prices = _PRICE_PATTERN.findall(text)
    if not prices:
        return None
    # Take the last price found (usually the threshold)
    return float(prices[-1].replace(",", ""))


def _extract_direction(market: Market) -> float:
    """Return 1.0 for 'above' markets, -1.0 for 'below' markets."""
    text = " ".join(
        v for v in [market.title, market.subtitle, market.yes_sub_title] if v
    )
    if _ABOVE_PATTERN.search(text):
        return 1.0
    if _BELOW_PATTERN.search(text):
        return -1.0
    return 1.0  # default: "above"


def _extract_crypto_symbol(market: Market) -> str | None:
    """Extract the crypto trading pair from a market title."""
    text = " ".join(
        v for v in [
            market.ticker, market.title, market.subtitle,
            market.event_ticker, market.series_ticker,
        ] if v
    ).lower()
    for keyword, symbol in _SYMBOL_MAP.items():
        if keyword in text:
            return symbol
    return None


def _hours_to_expiry(market: Market) -> float | None:
    """Calculate hours remaining until market expiry."""
    expiry = getattr(market, "expiration_time", None) or getattr(market, "close_time", None)
    if expiry is None:
        return None
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    now = datetime.now(timezone.utc)
    delta = (expiry - now).total_seconds() / 3600.0
    return max(delta, 0.01)


# ---------------------------------------------------------------------------
# Crypto data provider
# ---------------------------------------------------------------------------


class CryptoDataProvider:
    """Fetches real-time crypto data and builds a feature-ready snapshot."""

    def __init__(self, crypto_feed: Any, volatility_feed: Any) -> None:
        self._feed = crypto_feed
        self._vol_feed = volatility_feed
        self._vol_cache: dict[str, Any] = {}
        self._vol_cache_time: dict[str, datetime] = {}

    async def get_context(self, market: Market | None) -> dict[str, Any] | None:
        if market is None:
            return None

        symbol = _extract_crypto_symbol(market)
        if symbol is None:
            logger.debug("crypto_provider.no_symbol", ticker=market.ticker)
            return None

        threshold = _extract_threshold(market)
        if threshold is None:
            logger.debug("crypto_provider.no_threshold", ticker=market.ticker)
            return None

        hours = _hours_to_expiry(market)
        if hours is None or hours <= 0:
            return None

        snapshot: dict[str, Any] = {
            "threshold_price": threshold,
            "hours_to_expiry": hours,
            "direction": _extract_direction(market),
        }

        # Fetch volatility data (cache for 60s)
        now = datetime.now(timezone.utc)
        cache_age = (now - self._vol_cache_time.get(symbol, datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
        if cache_age > 60 or symbol not in self._vol_cache:
            try:
                vol_snap = await self._vol_feed.get_volatility(symbol)
                if vol_snap is not None:
                    self._vol_cache[symbol] = vol_snap
                    self._vol_cache_time[symbol] = now
            except Exception:
                logger.debug("crypto_provider.vol_fetch_failed", symbol=symbol, exc_info=True)

        vol_snap = self._vol_cache.get(symbol)
        if vol_snap is not None:
            snapshot["current_price"] = vol_snap.brti_price
            snapshot["brti_price"] = vol_snap.brti_price
            snapshot["realized_vol_24h"] = vol_snap.realized_vol_24h
            snapshot["realized_vol_7d"] = vol_snap.realized_vol_7d
            snapshot["realized_vol_30d"] = vol_snap.realized_vol_30d
            snapshot["implied_vol"] = vol_snap.implied_vol
            snapshot["atr_14"] = vol_snap.atr_14
            snapshot["atr_24"] = vol_snap.atr_24
            snapshot["trend_regime"] = (
                {"strong_up": 1.0, "up": 0.5, "neutral": 0.0, "down": -0.5, "strong_down": -1.0}
                .get(vol_snap.trend_regime or "neutral", 0.0)
            )
            snapshot["trend_strength"] = vol_snap.trend_strength
        else:
            return None  # Can't price without vol data

        # Fetch funding rate
        try:
            rates = await self._feed.get_funding_rates([symbol])
            if rates:
                snapshot["funding_rate"] = rates[0].rate
                snapshot["funding_rate_signal"] = rates[0].rate
        except Exception:
            logger.debug("crypto_provider.funding_fetch_failed", symbol=symbol, exc_info=True)

        # Fetch open interest
        try:
            oi_snaps = await self._feed.get_open_interest([symbol])
            if oi_snaps:
                snapshot["open_interest"] = oi_snaps[0].value
        except Exception:
            logger.debug("crypto_provider.oi_fetch_failed", symbol=symbol, exc_info=True)

        logger.debug(
            "crypto_provider.snapshot_built",
            ticker=market.ticker,
            symbol=symbol,
            price=snapshot.get("current_price"),
            threshold=threshold,
            hours=round(hours, 1),
        )
        return snapshot

    async def close(self) -> None:
        if self._feed is not None:
            await self._feed.close()
        if self._vol_feed is not None:
            await self._vol_feed.close()


# ---------------------------------------------------------------------------
# Weather data provider
# ---------------------------------------------------------------------------


class WeatherDataProvider:
    """Fetches NOAA/ECMWF ensemble forecasts and builds a feature-ready snapshot."""

    def __init__(self, noaa_fetcher: Any, ecmwf_fetcher: Any, locations: list[dict[str, Any]]) -> None:
        self._noaa = noaa_fetcher
        self._ecmwf = ecmwf_fetcher
        self._locations = {loc["name"].lower(): loc for loc in locations}
        self._ensemble_cache: dict[str, Any] = {}
        self._cache_time: dict[str, datetime] = {}

    async def get_context(self, market: Market | None) -> dict[str, Any] | None:
        if market is None:
            return None

        # Match market to a location
        text = " ".join(
            v for v in [market.title, market.subtitle, market.event_ticker] if v
        ).lower()

        matched_loc = None
        for name, loc in self._locations.items():
            # Check for city name or common abbreviations
            if name in text or name.split(",")[0] in text:
                matched_loc = loc
                break
            # Check common abbreviations
            abbrevs = {
                "new york": ["nyc", "new york", "manhattan"],
                "chicago": ["chicago", "chi"],
                "los angeles": ["los angeles", "la ", "l.a."],
            }
            for full_name, abbrs in abbrevs.items():
                if full_name in name and any(a in text for a in abbrs):
                    matched_loc = loc
                    break
            if matched_loc:
                break

        if matched_loc is None:
            return None

        threshold = _extract_threshold(market)
        hours = _hours_to_expiry(market)

        # Fetch ensemble (cache for 30 min)
        cache_key = f"{matched_loc['name']}_{market.ticker}"
        now = datetime.now(timezone.utc)
        cache_age = (now - self._cache_time.get(cache_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds()

        if cache_age > 1800 or cache_key not in self._ensemble_cache:
            # Determine weather variable from market title
            variable = "temperature_2m"  # default
            if any(w in text for w in ("rain", "precipitation", "inches of rain")):
                variable = "precipitation"
            elif any(w in text for w in ("wind", "gust")):
                variable = "wind_speed_10m"
            elif any(w in text for w in ("snow", "snowfall")):
                variable = "snowfall"

            try:
                ensemble = await self._noaa.fetch_ensemble(
                    lat=matched_loc["lat"],
                    lon=matched_loc["lon"],
                    variable=variable,
                    location_name=matched_loc["name"],
                )
                if ensemble is not None:
                    self._ensemble_cache[cache_key] = ensemble
                    self._cache_time[cache_key] = now
            except Exception:
                logger.debug("weather_provider.fetch_failed", loc=matched_loc["name"], exc_info=True)

        ensemble = self._ensemble_cache.get(cache_key)
        if ensemble is None:
            return None

        snapshot: dict[str, Any] = {"ensemble": ensemble}
        if threshold is not None:
            snapshot["threshold"] = threshold
        if hours is not None:
            snapshot["hours_to_expiry"] = hours

        # Try ECMWF for model comparison
        try:
            ecmwf_ens = await self._ecmwf.fetch_ensemble(
                lat=matched_loc["lat"],
                lon=matched_loc["lon"],
                variable=variable,
                location_name=matched_loc["name"],
            )
            if ecmwf_ens is not None:
                snapshot["ecmwf_ensemble"] = ecmwf_ens
        except Exception:
            pass

        return snapshot

    async def close(self) -> None:
        pass
