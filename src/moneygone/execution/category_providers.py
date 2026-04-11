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

_ABOVE_PATTERN = re.compile(r"above|over|higher|more than|at least|reach|exceed|or above|>\d", re.IGNORECASE)
_BELOW_PATTERN = re.compile(r"below|under|lower|less than|at most|drop|or below|<\d", re.IGNORECASE)

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


# Ticker suffix patterns for weather thresholds:
#   -T47.99  → threshold 47.99 (temperature or amount)
#   -B40.5   → below 40.5
#   -A80     → above 80
#   -T0      → threshold 0 (any rain/snow)
#   -3       → threshold 3 (e.g., KXRAINNYCM-26APR-3 = 3 inches)
_TICKER_THRESHOLD_RE = re.compile(
    r"-[TBA]?(\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def _extract_threshold(market: Market) -> float | None:
    """Extract the numeric threshold from a market title or ticker suffix.

    Checks ticker suffix first (more reliable), then falls back to
    parsing dollar amounts from the title text.
    """
    # Try ticker suffix first — e.g., KXLOWTNYC-26APR10-B40.5 → 40.5
    if market.ticker:
        m = _TICKER_THRESHOLD_RE.search(market.ticker)
        if m:
            return float(m.group(1))

    # Fallback: parse from title text
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
    """Fetches real-time crypto data and builds a feature-ready snapshot.

    Parameters
    ----------
    crypto_feed:
        ccxt-based feed for spot data (orderbook, trades, OHLCV).
    volatility_feed:
        Volatility feed for realized vol, ATR, trend, price.
    futures_feed:
        Optional Coinalyze (or similar) feed for funding rates and OI.
        When provided, funding/OI queries go here instead of crypto_feed.
    """

    def __init__(
        self,
        crypto_feed: Any,
        volatility_feed: Any,
        futures_feed: Any | None = None,
    ) -> None:
        self._feed = crypto_feed
        self._vol_feed = volatility_feed
        self._futures_feed = futures_feed
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

        # Fetch funding rate (prefer futures feed, fallback to ccxt)
        feed = self._futures_feed or self._feed
        try:
            rates = await feed.get_funding_rates([symbol])
            if rates:
                snapshot["funding_rate"] = rates[0].rate
                snapshot["funding_rate_signal"] = rates[0].rate
        except Exception:
            logger.debug("crypto_provider.funding_fetch_failed", symbol=symbol, exc_info=True)

        # Fetch open interest (prefer futures feed, fallback to ccxt)
        try:
            oi_snaps = await feed.get_open_interest([symbol])
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
        if self._futures_feed is not None:
            await self._futures_feed.close()


# ---------------------------------------------------------------------------
# Weather data provider
# ---------------------------------------------------------------------------


_WEATHER_TICKER_RE = re.compile(
    r"KXLOWT|KXHIGHT|KXTEMP|KXRAIN|KXSNOW|KXWIND|KXHURR|KXPRECIP",
    re.IGNORECASE,
)


class WeatherDataProvider:
    """Fetches NWS observations + NOAA/ECMWF ensemble forecasts for weather markets.

    Data priority:
    1. NWS current observation (ground truth) — free, no key
    2. NWS hourly forecast (deterministic, high-resolution) — free, no key
    3. NOAA GEFS ensemble (probabilistic spread) — Open-Meteo, free
    4. ECMWF IFS ensemble (model comparison) — Open-Meteo, free
    """

    # Common abbreviations for matching market text to locations
    _LOCATION_ALIASES: dict[str, list[str]] = {
        "new york": ["nyc", "new york", "manhattan", "ny"],
        "chicago": ["chicago", "chi"],
        "los angeles": ["los angeles", "la ", "l.a."],
        "miami": ["miami", "mia"],
        "dallas": ["dallas", "dfw"],
        "denver": ["denver", "den"],
        "seattle": ["seattle", "sea"],
        "atlanta": ["atlanta", "atl"],
    }

    def __init__(
        self,
        noaa_fetcher: Any,
        ecmwf_fetcher: Any,
        locations: list[dict[str, Any]],
        nws_fetcher: Any | None = None,
    ) -> None:
        self._noaa = noaa_fetcher
        self._ecmwf = ecmwf_fetcher
        self._nws = nws_fetcher
        self._locations = {loc["name"].lower(): loc for loc in locations}
        self._ensemble_cache: dict[str, Any] = {}
        self._cache_time: dict[str, datetime] = {}

    def _match_location(self, text: str) -> dict[str, Any] | None:
        """Match market text to a configured location."""
        for name, loc in self._locations.items():
            if name in text or name.split(",")[0] in text:
                return loc
            for full_name, abbrs in self._LOCATION_ALIASES.items():
                if full_name in name and any(a in text for a in abbrs):
                    return loc
        return None

    @staticmethod
    def _detect_variable(text: str) -> str:
        """Determine weather variable from market text."""
        if any(w in text for w in ("rain", "precipitation", "inches of rain")):
            return "precipitation"
        if any(w in text for w in ("wind", "gust")):
            return "wind_speed_10m"
        if any(w in text for w in ("snow", "snowfall")):
            return "snowfall"
        return "temperature_2m"

    async def get_context(self, market: Market | None) -> dict[str, Any] | None:
        if market is None:
            return None

        # Only process actual weather markets — ticker must match known patterns
        if not _WEATHER_TICKER_RE.search(market.ticker or ""):
            return None

        # Skip monthly accumulation markets — they tie up capital for weeks
        # and our ensemble model only has hourly precip, not monthly totals.
        # Monthly rain/snow tickers have "M" before the date segment:
        #   KXRAINSEAM-26APR, KXRAINCHIM-26APR, etc.
        ticker_upper = (market.ticker or "").upper()
        if re.search(r"KX(?:RAIN|SNOW|PRECIP)\w*M-\d", ticker_upper):
            logger.debug(
                "weather_provider.monthly_market_skipped",
                ticker=market.ticker,
                msg="Monthly accumulation market — too much capital lockup",
            )
            return None

        # Match market to a location
        text = " ".join(
            v for v in [
                market.title, market.subtitle, market.event_ticker, market.ticker,
            ] if v
        ).lower()

        matched_loc = self._match_location(text)
        if matched_loc is None:
            return None

        threshold = _extract_threshold(market)
        if threshold is None:
            logger.debug(
                "weather_provider.no_threshold",
                ticker=market.ticker,
                msg="Cannot extract threshold — skipping to avoid blind trades",
            )
            return None

        hours = _hours_to_expiry(market)
        variable = self._detect_variable(text)

        # Unit conversion: Kalshi uses °F and inches; NOAA uses °C and mm
        if variable == "temperature_2m":
            # Convert threshold from °F to °C for ensemble comparison
            threshold = (threshold - 32.0) * 5.0 / 9.0
        elif variable == "precipitation":
            if threshold == 0:
                # "Any rain" = use NWS trace cutoff to avoid ensemble noise
                threshold = 0.254  # 0.01 inches in mm
            else:
                # Convert inches → mm for NOAA ensemble data
                threshold = threshold * 25.4
        lat, lon = matched_loc["lat"], matched_loc["lon"]
        loc_name = matched_loc["name"]

        # Fetch ensemble (cache for 30 min per location+variable).
        # The ensemble data is the SAME for all contracts in an event — only
        # the threshold differs.  Keying on ticker caused hundreds of redundant
        # API calls, burning through Open-Meteo's free-tier rate limit.
        cache_key = f"{loc_name}_{variable}"
        now = datetime.now(timezone.utc)
        cache_age = (
            now - self._cache_time.get(cache_key, datetime.min.replace(tzinfo=timezone.utc))
        ).total_seconds()

        if cache_age > 1800 or cache_key not in self._ensemble_cache:
            ensemble = None

            # Primary: NOAA GEFS ensemble (probabilistic spread for calibration)
            try:
                ensemble = await self._noaa.fetch_ensemble(
                    lat=lat, lon=lon,
                    variable=variable,
                    location_name=loc_name,
                )
            except Exception:
                logger.debug("weather_provider.noaa_failed", loc=loc_name, exc_info=True)

            # Fallback: NWS hourly forecast (deterministic, 1-member)
            if (ensemble is None or not ensemble.valid_times) and self._nws is not None:
                try:
                    ensemble = await self._nws.fetch_ensemble(
                        lat=lat, lon=lon,
                        variable=variable,
                        location_name=loc_name,
                    )
                except Exception:
                    logger.debug("weather_provider.nws_failed", loc=loc_name, exc_info=True)

            if ensemble is not None and ensemble.valid_times:
                self._ensemble_cache[cache_key] = ensemble
                self._cache_time[cache_key] = now

        ensemble = self._ensemble_cache.get(cache_key)
        if ensemble is None:
            return None

        direction = _extract_direction(market)

        # Determine weather variable type (high/low) for bias correction
        weather_var = "high"  # default
        if "KXLOW" in ticker_upper or "low" in text:
            weather_var = "low"

        snapshot: dict[str, Any] = {
            "ensemble": ensemble,
            "direction": direction,
            "location_name": loc_name,
            "weather_variable": weather_var,
        }
        if threshold is not None:
            snapshot["threshold"] = threshold
        if hours is not None:
            snapshot["hours_to_expiry"] = hours

        # Get NWS current observation for ground-truth anchoring
        if self._nws is not None:
            try:
                obs = await self._nws.get_current_observation(lat, lon, loc_name)
                if obs is not None:
                    snapshot["nws_observation"] = obs
                    # If temperature market, include current temp for decision context
                    if variable == "temperature_2m" and obs.temperature_f is not None:
                        snapshot["current_temp_f"] = obs.temperature_f
                        snapshot["current_temp_c"] = obs.temperature_c
            except Exception:
                logger.debug("weather_provider.nws_obs_failed", loc=loc_name, exc_info=True)

        # Try ECMWF for model comparison
        try:
            ecmwf_ens = await self._ecmwf.fetch_ensemble(
                lat=lat, lon=lon,
                variable=variable,
                location_name=loc_name,
            )
            if ecmwf_ens is not None:
                snapshot["ecmwf_ensemble"] = ecmwf_ens
        except Exception:
            pass

        logger.info(
            "weather_provider.context_built",
            ticker=market.ticker,
            location=loc_name,
            variable=variable,
            threshold=threshold,
            hours=round(hours, 1) if hours else None,
            has_nws_obs="nws_observation" in snapshot,
            current_temp_f=snapshot.get("current_temp_f"),
        )
        return snapshot

    async def close(self) -> None:
        if self._nws is not None:
            await self._nws.close()
        if self._noaa is not None:
            await self._noaa.close()
        if self._ecmwf is not None:
            await self._ecmwf.close()
