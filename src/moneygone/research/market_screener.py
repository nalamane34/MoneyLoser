"""Market screener for identifying tradeable opportunities on Kalshi.

Scans all markets and ranks them by a composite opportunity score that
considers liquidity, spread tightness, time to expiry, and model coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market, MarketStatus

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MarketOpportunity:
    """A screened market opportunity with scoring."""

    ticker: str
    title: str
    category: str
    volume: int
    open_interest: int
    spread: float
    time_to_expiry_hours: float
    opportunity_score: float


@dataclass
class ScreenerFilters:
    """Filters for the market screener."""

    min_volume: int = 10
    min_oi: int = 5
    max_spread: float = 0.15
    categories: list[str] | None = None
    min_hours_to_expiry: float = 1.0


class MarketScreener:
    """Scans Kalshi markets and ranks them by tradability.

    Scoring components (each normalized to 0-1, then combined):

    * **Liquidity** (40%): ``log(volume * OI + 1)`` normalized
    * **Spread tightness** (30%): ``1 - spread / max_spread``
    * **Time to expiry** (20%): favours 6-72 hours, penalizes extremes
    * **Model coverage** (10%): bonus if category is in known model set

    Parameters
    ----------
    model_categories:
        Set of market categories the system has trained models for.
    """

    def __init__(
        self, model_categories: set[str] | None = None
    ) -> None:
        self._model_categories = model_categories or set()
        log.info(
            "market_screener.initialized",
            model_categories=list(self._model_categories),
        )

    async def scan(
        self,
        rest_client: KalshiRestClient,
        filters: ScreenerFilters | None = None,
    ) -> list[MarketOpportunity]:
        """Fetch all open markets and rank by opportunity score.

        Parameters
        ----------
        rest_client:
            Authenticated Kalshi REST client.
        filters:
            Optional screening filters.  Defaults are applied if ``None``.

        Returns
        -------
        list[MarketOpportunity]
            Markets sorted by descending opportunity score.
        """
        if filters is None:
            filters = ScreenerFilters()

        # Fetch open markets (paginate if necessary)
        all_markets: list[Market] = []
        cursor: str | None = None

        while True:
            kwargs: dict[str, Any] = {"status": "open", "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor

            batch = await rest_client.get_markets(**kwargs)
            if not batch:
                break
            all_markets.extend(batch)

            # Simple pagination heuristic: if we got a full page, there may be more
            if len(batch) < 200:
                break
            # Use last ticker as pseudo-cursor (Kalshi API may use cursor differently)
            cursor = batch[-1].ticker

        log.info("screener.fetched_markets", total=len(all_markets))

        # Filter and score
        opportunities: list[MarketOpportunity] = []
        now = datetime.now(timezone.utc)

        for market in all_markets:
            if market.status != MarketStatus.OPEN:
                continue

            # Compute spread
            spread = float(market.yes_ask - market.yes_bid)
            if spread < 0:
                spread = 0.0

            # Time to expiry
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            hours_to_expiry = max(
                0.0, (close_time - now).total_seconds() / 3600.0
            )

            # Apply filters
            if market.volume < filters.min_volume:
                continue
            if market.open_interest < filters.min_oi:
                continue
            if spread > filters.max_spread:
                continue
            if hours_to_expiry < filters.min_hours_to_expiry:
                continue
            if filters.categories and market.category not in filters.categories:
                continue

            # Compute score
            score = self._compute_score(
                volume=market.volume,
                open_interest=market.open_interest,
                spread=spread,
                hours_to_expiry=hours_to_expiry,
                category=market.category,
                max_spread=filters.max_spread,
            )

            opportunities.append(
                MarketOpportunity(
                    ticker=market.ticker,
                    title=market.title,
                    category=market.category,
                    volume=market.volume,
                    open_interest=market.open_interest,
                    spread=round(spread, 4),
                    time_to_expiry_hours=round(hours_to_expiry, 2),
                    opportunity_score=round(score, 4),
                )
            )

        # Sort by descending score
        opportunities.sort(key=lambda o: o.opportunity_score, reverse=True)
        log.info(
            "screener.scan_complete",
            total_scanned=len(all_markets),
            opportunities=len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        volume: int,
        open_interest: int,
        spread: float,
        hours_to_expiry: float,
        category: str,
        max_spread: float,
    ) -> float:
        """Compute composite opportunity score (0-1)."""
        import math

        # Liquidity score (40%): log-scaled volume * OI
        liq_raw = math.log(volume * open_interest + 1)
        liq_max = math.log(100_000 * 10_000 + 1)  # normalization constant
        liquidity_score = min(1.0, liq_raw / liq_max)

        # Spread tightness (30%): tighter is better
        spread_score = max(0.0, 1.0 - spread / max_spread) if max_spread > 0 else 1.0

        # Time to expiry (20%): bell curve centered at ~24 hours
        # Ideal: 6-72 hours, penalize <1h and >168h (1 week)
        if hours_to_expiry < 1:
            time_score = 0.1
        elif hours_to_expiry < 6:
            time_score = 0.3 + 0.7 * (hours_to_expiry - 1) / 5
        elif hours_to_expiry <= 72:
            time_score = 1.0
        elif hours_to_expiry <= 168:
            time_score = max(0.3, 1.0 - (hours_to_expiry - 72) / 96)
        else:
            time_score = 0.2

        # Model coverage (10%): bonus for known categories
        model_score = 1.0 if category in self._model_categories else 0.3

        # Weighted composite
        score = (
            0.40 * liquidity_score
            + 0.30 * spread_score
            + 0.20 * time_score
            + 0.10 * model_score
        )
        return score
