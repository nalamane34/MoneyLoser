"""Shared market discovery and classification service.

Paginates Kalshi ``get_all_markets`` once, classifies each market by
category, and writes the result to a JSON cache file that any worker
process can read.  Eliminates duplicate API calls and halves memory
usage when multiple workers need the same market universe.

Usage::

    # Writer process (market_data worker):
    discovery = MarketDiscoveryService(rest_client, cache_path)
    await discovery.start()   # initial fetch + background refresh

    # Reader process (execution worker):
    markets, refreshed_at = MarketDiscoveryService.load_cache(cache_path)
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market, MarketResult, MarketStatus

log = structlog.get_logger("market_discovery")


# ---------------------------------------------------------------------------
# Market category classification
# ---------------------------------------------------------------------------


class MarketCategory(str, Enum):
    SPORTS = "sports"
    CRYPTO = "crypto"
    WEATHER = "weather"
    ECONOMICS = "economics"
    POLITICS = "politics"
    FINANCIALS = "financials"
    COMPANIES = "companies"
    UNKNOWN = "unknown"


_CRYPTO_PATTERNS = re.compile(
    r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|crypto|\bdoge\b|\bxrp\b|"
    r"coin price|\bbnb\b|\bada\b|\bavax\b|litecoin|\bltc\b",
    re.IGNORECASE,
)
_WEATHER_PATTERNS = re.compile(
    r"temperature|degrees|weather|rain|snow|hurricane|tornado|"
    r"heat wave|cold snap|precipitation|wind speed|forecast",
    re.IGNORECASE,
)
_ECONOMICS_PATTERNS = re.compile(
    r"fed fund|interest rate|cpi|inflation|gdp|unemployment|"
    r"jobs report|nonfarm|payroll|fomc|treasury|yield curve|"
    r"recession|economic|consumer price",
    re.IGNORECASE,
)
_SPORTS_PATTERNS = re.compile(
    r"\b(?:nba|nfl|mlb|nhl|ncaa|moneyline|championship|playoff|soccer|mls|epl|"
    r"tennis|boxing|mma|ufc|golf|pga|lpga|esports|bundesliga|serie a|la liga|"
    r"ligue 1|premier league|kbo|npb|total runs|spread|team total)\b|"
    r"\b(?:vs\.?|at)\b|"
    r"\bkx(?:nba|mlb|nhl|nfl|ncaa|epl|mls|laliga|bundesliga|seriea|ligue1|ucl|"
    r"kbo|npb|elh|pga|lpga|tennis|esports)",
    re.IGNORECASE,
)
_POLITICS_PATTERNS = re.compile(
    r"trump|biden|president|election|congress|senate|house|"
    r"democrat|republican|governor|mayor|vote|ballot|"
    r"political|cabinet|supreme court|scotus|impeach|"
    r"executive order|legislation|bill sign|veto|"
    r"primary|caucus|nominee|approval rating|poll",
    re.IGNORECASE,
)
_FINANCIALS_PATTERNS = re.compile(
    r"\bspy\b|\bs&p\b|s&p 500|nasdaq|\bqqq\b|dow jones|\bdji\b|"
    r"stock market|stock price|index|russell|vix|volatility|"
    r"nyse|wall street|bear market|bull market|"
    r"oil price|gold price|silver price|commodity|"
    r"forex|\beur\b|\busd\b|exchange rate|bond|"
    r"\binx\b|\binxu\b|\bkxsilver\b|\bkxgold\b|\bkxoil\b",
    re.IGNORECASE,
)
_COMPANIES_PATTERNS = re.compile(
    r"apple|google|amazon|meta|microsoft|tesla|nvidia|"
    r"truth social|tiktok|twitter|\bx corp\b|openai|"
    r"spacex|starlink|neuralink|"
    r"earnings|revenue|market cap|ipo|stock split|"
    r"ceo|layoff|merger|acquisition|antitrust",
    re.IGNORECASE,
)

_EXPLICIT_CATEGORY_ALIASES: tuple[tuple[str, MarketCategory], ...] = (
    ("crypto", MarketCategory.CRYPTO),
    ("weather", MarketCategory.WEATHER),
    ("climate", MarketCategory.WEATHER),
    ("econom", MarketCategory.ECONOMICS),
    ("politic", MarketCategory.POLITICS),
    ("election", MarketCategory.POLITICS),
    ("financial", MarketCategory.FINANCIALS),
    ("stock", MarketCategory.FINANCIALS),
    ("compan", MarketCategory.COMPANIES),
    ("business", MarketCategory.COMPANIES),
    ("tech", MarketCategory.COMPANIES),
    ("sport", MarketCategory.SPORTS),
)


def _classify_explicit_category(raw_category: str) -> MarketCategory | None:
    normalized = raw_category.strip().lower()
    if not normalized:
        return None
    for token, category in _EXPLICIT_CATEGORY_ALIASES:
        if token in normalized:
            return category
    return None


def classify_market(market: Market) -> MarketCategory:
    """Classify a Kalshi market into a trading category."""
    explicit = _classify_explicit_category(getattr(market, "category", ""))
    if explicit is not None:
        return explicit

    text = " ".join(
        v for v in [
            market.ticker,
            market.event_ticker,
            market.series_ticker,
            market.title,
            market.subtitle,
            market.yes_sub_title,
        ]
        if v
    )
    if _CRYPTO_PATTERNS.search(text):
        return MarketCategory.CRYPTO
    if _WEATHER_PATTERNS.search(text):
        return MarketCategory.WEATHER
    if _POLITICS_PATTERNS.search(text):
        return MarketCategory.POLITICS
    if _ECONOMICS_PATTERNS.search(text):
        return MarketCategory.ECONOMICS
    if _COMPANIES_PATTERNS.search(text):
        return MarketCategory.COMPANIES
    if _FINANCIALS_PATTERNS.search(text):
        return MarketCategory.FINANCIALS
    if _SPORTS_PATTERNS.search(text):
        return MarketCategory.SPORTS
    return MarketCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _market_to_dict(market: Market, category: MarketCategory) -> dict[str, Any]:
    """Serialize a Market + category to a JSON-safe dict."""
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "series_ticker": market.series_ticker,
        "title": market.title,
        "subtitle": market.subtitle,
        "yes_sub_title": market.yes_sub_title,
        "no_sub_title": market.no_sub_title,
        "status": market.status.value,
        "yes_bid": str(market.yes_bid),
        "yes_ask": str(market.yes_ask),
        "last_price": str(market.last_price),
        "volume": market.volume,
        "open_interest": market.open_interest,
        "close_time": market.close_time.isoformat(),
        "result": market.result.value,
        "category": market.category,
        "market_category": category.value,
        "created_time": market.created_time.isoformat() if market.created_time else None,
        "open_time": market.open_time.isoformat() if market.open_time else None,
        "previous_price": str(market.previous_price),
        "liquidity_dollars": str(market.liquidity_dollars),
        "strike_type": market.strike_type,
        "floor_strike": str(market.floor_strike) if market.floor_strike is not None else None,
        "cap_strike": str(market.cap_strike) if market.cap_strike is not None else None,
        "mve_selected_legs": list(market.mve_selected_legs),
    }


def _dict_to_market(d: dict[str, Any]) -> tuple[Market, MarketCategory]:
    """Deserialize a dict back to a Market + MarketCategory."""
    market = Market(
        ticker=d["ticker"],
        event_ticker=d["event_ticker"],
        series_ticker=d.get("series_ticker", ""),
        title=d["title"],
        subtitle=d.get("subtitle", ""),
        yes_sub_title=d.get("yes_sub_title", ""),
        no_sub_title=d.get("no_sub_title", ""),
        status=MarketStatus(d["status"]),
        yes_bid=Decimal(d["yes_bid"]),
        yes_ask=Decimal(d["yes_ask"]),
        last_price=Decimal(d["last_price"]),
        volume=d["volume"],
        open_interest=d["open_interest"],
        close_time=datetime.fromisoformat(d["close_time"]),
        result=MarketResult(d["result"]),
        category=d.get("category", ""),
        created_time=(
            datetime.fromisoformat(d["created_time"])
            if d.get("created_time")
            else None
        ),
        open_time=(
            datetime.fromisoformat(d["open_time"])
            if d.get("open_time")
            else None
        ),
        previous_price=Decimal(d.get("previous_price", "0")),
        liquidity_dollars=Decimal(d.get("liquidity_dollars", "0")),
        strike_type=d.get("strike_type", ""),
        floor_strike=(
            Decimal(d["floor_strike"])
            if d.get("floor_strike") is not None
            else None
        ),
        cap_strike=(
            Decimal(d["cap_strike"])
            if d.get("cap_strike") is not None
            else None
        ),
        mve_selected_legs=tuple(d.get("mve_selected_legs", []) or []),
    )
    cat = MarketCategory(d["market_category"])
    return market, cat


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------


class MarketDiscoveryService:
    """Discovers all Kalshi markets, classifies them, caches to a JSON file.

    One process (market_data worker) runs the discovery loop and writes
    the cache.  Other processes call ``load_cache()`` to read it.
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        cache_path: Path,
        refresh_interval: float = 120.0,
        max_pages: int = 0,
    ) -> None:
        self._rest = rest_client
        self._cache_path = cache_path
        self._refresh_interval = refresh_interval
        self._max_pages = max_pages
        self._markets: list[tuple[Market, MarketCategory]] = []
        self._task: asyncio.Task | None = None
        self._refreshed_at: datetime | None = None

    # -- lifecycle --

    async def start(self) -> None:
        """Run initial discovery and start background refresh loop."""
        await self.refresh()
        self._task = asyncio.create_task(self._loop(), name="market_discovery")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval)
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("market_discovery.refresh_error")

    # -- core --

    async def refresh(self) -> list[tuple[Market, MarketCategory]]:
        """Fetch all open markets from Kalshi, classify, and write cache."""
        now = datetime.now(timezone.utc)
        markets = await self._rest.get_all_markets(
            limit=1_000,
            max_pages=self._max_pages,
            status="open",
            mve_filter="exclude",  # Skip multivariate combos (KXMVE*)
        )

        # Guard against aliasing bugs or stale cached payloads by keeping
        # only currently tradeable markets here as well.
        classified: list[tuple[Market, MarketCategory]] = []
        category_counts: dict[str, int] = {}
        skipped_closed = 0

        for m in markets:
            if m.status != MarketStatus.OPEN:
                skipped_closed += 1
                continue
            cat = classify_market(m)
            classified.append((m, cat))
            category_counts[cat.value] = category_counts.get(cat.value, 0) + 1

        self._markets = classified
        self._refreshed_at = now
        self._write_cache(classified)

        log.info(
            "market_discovery.refreshed",
            total=len(classified),
            fetched=len(markets),
            skipped_closed=skipped_closed,
            categories=category_counts,
        )
        return classified

    def get_markets(
        self,
        category: MarketCategory | None = None,
    ) -> list[Market]:
        """Return in-memory cached markets, optionally filtered by category."""
        if category is None:
            return [m for m, _ in self._markets]
        return [m for m, c in self._markets if c == category]

    def get_all_classified(self) -> list[tuple[Market, MarketCategory]]:
        """Return all in-memory (Market, MarketCategory) pairs."""
        return list(self._markets)

    def get_tickers(
        self,
        category: MarketCategory | None = None,
    ) -> set[str]:
        """Return set of tickers, optionally filtered by category."""
        if category is None:
            return {m.ticker for m, _ in self._markets}
        return {m.ticker for m, c in self._markets if c == category}

    @property
    def refreshed_at(self) -> datetime | None:
        return self._refreshed_at

    # -- cache I/O --

    def _write_cache(self, classified: list[tuple[Market, MarketCategory]]) -> None:
        """Atomic write: temp file + rename."""
        payload = {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "count": len(classified),
            "markets": [_market_to_dict(m, c) for m, c in classified],
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._cache_path.parent,
            suffix=".tmp",
        )
        try:
            with open(tmp_fd, "w") as f:
                json.dump(payload, f)
            Path(tmp_path).replace(self._cache_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def load_cache(
        cache_path: Path,
    ) -> tuple[list[tuple[Market, MarketCategory]], datetime | None]:
        """Read the discovery cache from disk.

        Safe to call from any process at any time — reads are atomic
        because the writer uses temp-file + rename.

        Returns (classified_markets, refreshed_at) or ([], None) if
        the cache doesn't exist or is corrupt.
        """
        if not cache_path.exists():
            return [], None
        try:
            with open(cache_path) as f:
                payload = json.load(f)
            refreshed_at = datetime.fromisoformat(payload["refreshed_at"])
            classified = [_dict_to_market(d) for d in payload["markets"]]
            return classified, refreshed_at
        except (json.JSONDecodeError, KeyError, ValueError):
            log.warning("market_discovery.cache_corrupt", path=str(cache_path))
            return [], None
