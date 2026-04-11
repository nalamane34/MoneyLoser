"""Esports odds feed from OddsPapi.

The Odds API has zero esports coverage, so we use OddsPapi for
CS2, LoL, Valorant, Dota 2, Overwatch, R6, and Rocket League odds.

OddsPapi provides Pinnacle + 350 bookmaker odds via a free tier
(250 requests/month).  Sign up at https://oddspapi.io/ and set
``ODDSPAPI_API_KEY`` in your environment.

API structure:
- GET /v4/fixtures?sportId=X&from=...&to=... → list of matches (no odds)
- GET /v4/odds?fixtureId=X → full odds for one match
- Market 171 = Match Winner (moneyline)
- Outcome 171 = participant1, Outcome 172 = participant2
- bookmakerOdds → {bookmaker_slug: {markets: {171: {outcomes: {171/172: ...}}}}}

https://oddspapi.io/docs/endpoints
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncio

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BASE = "https://api.oddspapi.io/v4"

# OddsPapi sport IDs — from GET /v4/sports (sportId field).
ESPORTS_SPORT_IDS: dict[str, int] = {
    "esports_cs2": 17,        # ESport Counter-Strike
    "esports_lol": 18,        # ESport League of Legends
    "esports_valorant": 61,   # ESport Valorant
    "esports_dota2": 16,      # ESport Dota
    "esports_overwatch": 57,  # ESport Overwatch
    "esports_r6": 58,         # ESport Rainbow Six
    "esports_rl": 59,         # ESport Rocket League
}

# Match winner market IDs in OddsPapi.  Different sports use different IDs
# but all are named "Winner".  Outcomes follow a pattern: first=participant1,
# second=participant2 (e.g. 171/172, 181/182, 121/122).
_MATCH_WINNER_MARKETS = {"171", "181", "121", "111", "10728"}

# Sharp bookmaker slugs in OddsPapi (in priority order).
SHARP_BOOKMAKERS = ["pinnacle", "betfair", "singbet", "bwin", "bet365"]


@dataclass(frozen=True, slots=True)
class EsportsOddsLine:
    """A single esports match odds line."""

    fixture_id: str
    home_team: str
    away_team: str
    home_price: float      # decimal odds
    away_price: float      # decimal odds
    bookmaker: str
    commence_time: datetime | None = None
    sport: str = ""


class EsportsOddsFeed:
    """Fetches esports betting odds from OddsPapi.

    Parameters
    ----------
    api_key:
        OddsPapi API key.  Falls back to ``ODDSPAPI_API_KEY`` env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ODDSPAPI_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self._timeout = request_timeout
        self._requests_used = 0

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    @property
    def requests_used(self) -> int:
        return self._requests_used

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"User-Agent": "MoneyGone/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_json(self, url: str, params: dict[str, str] | None = None) -> Any | None:
        if not self._api_key:
            logger.warning("esports_odds.no_api_key")
            return None

        client = await self._get_client()
        full_params: dict[str, str] = {"apiKey": self._api_key}
        if params:
            full_params.update(params)

        try:
            # OddsPapi free tier has aggressive rate limiting.
            await asyncio.sleep(0.5)
            resp = await client.get(url, params=full_params)
            self._requests_used += 1
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "esports_odds.http_error",
                url=url,
                status=exc.response.status_code,
                body=exc.response.text[:200],
            )
            return None
        except httpx.TransportError as exc:
            logger.warning("esports_odds.transport_error", url=url, error=str(exc))
            return None

    async def get_fixtures(self, sport: str, days_ahead: int = 3) -> list[dict]:
        """Get upcoming fixtures for an esports title.

        Costs 1 API request.
        """
        sport_id = ESPORTS_SPORT_IDS.get(sport)
        if sport_id is None:
            return []

        now = datetime.now(timezone.utc)
        from_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

        data = await self._get_json(
            f"{_BASE}/fixtures",
            params={
                "sportId": str(sport_id),
                "from": from_str,
                "to": to_str,
            },
        )
        if data is None or not isinstance(data, list):
            return []

        # Filter to fixtures that have odds
        return [f for f in data if f.get("hasOdds")]

    async def get_fixture_odds(self, fixture_id: str) -> dict | None:
        """Get odds for a single fixture.

        Costs 1 API request.
        """
        return await self._get_json(
            f"{_BASE}/odds",
            params={"fixtureId": fixture_id},
        )

    async def get_odds(self, sport: str, max_fixtures: int = 20) -> list[EsportsOddsLine]:
        """Fetch moneyline odds for an esports title.

        Costs 1 + N API requests (1 for fixtures, N for individual odds).
        With 250 req/month budget, be conservative with max_fixtures.

        Parameters
        ----------
        sport:
            Our sport key, e.g. ``"esports_cs2"``.
        max_fixtures:
            Maximum fixtures to fetch odds for (controls API usage).
        """
        fixtures = await self.get_fixtures(sport)
        if not fixtures:
            logger.info("esports_odds.no_fixtures", sport=sport)
            return []

        # Sort by start time, take only upcoming ones
        fixtures.sort(key=lambda f: f.get("startTime", ""))
        fixtures = fixtures[:max_fixtures]

        all_lines: list[EsportsOddsLine] = []
        for fixture in fixtures:
            fixture_id = fixture.get("fixtureId", "")
            if not fixture_id:
                continue

            odds_data = await self.get_fixture_odds(fixture_id)
            if odds_data is None:
                continue

            # Carry participant names from fixtures (odds response omits them)
            odds_data.setdefault("participant1Name", fixture.get("participant1Name", ""))
            odds_data.setdefault("participant2Name", fixture.get("participant2Name", ""))
            odds_data.setdefault("startTime", fixture.get("startTime"))

            lines = self._parse_odds_response(odds_data, sport)
            all_lines.extend(lines)

        logger.info(
            "esports_odds.fetched",
            sport=sport,
            fixtures_checked=len(fixtures),
            lines=len(all_lines),
            requests_used=self._requests_used,
        )
        return all_lines

    async def get_all_esports_odds(
        self,
        max_fixtures_per_sport: int = 10,
        sports: list[str] | None = None,
    ) -> dict[str, list[EsportsOddsLine]]:
        """Fetch odds for all supported esports titles.

        With 250 req/month budget:
        - 7 sports × (1 fixture call + 10 odds calls) = 77 requests per run
        - Can run ~3× per month, or reduce max_fixtures_per_sport

        For the collector running every 30 min (1440/month):
        - Use max_fixtures_per_sport=3 → 7×4=28 req/run
        - 250/28 ≈ 8 runs/month → run esports collection every ~4 days
        - OR collect only CS2+LoL+Valorant (3 sports) → 3×4=12/run → 20 runs

        Parameters
        ----------
        max_fixtures_per_sport:
            Max fixtures to fetch per sport.
        sports:
            Optional subset of sports to fetch. Defaults to all.
        """
        target_sports = sports or list(ESPORTS_SPORT_IDS.keys())
        result: dict[str, list[EsportsOddsLine]] = {}

        for sport in target_sports:
            lines = await self.get_odds(sport, max_fixtures=max_fixtures_per_sport)
            if lines:
                result[sport] = lines

        return result

    def build_line_history_rows(
        self,
        lines: list[EsportsOddsLine],
        captured_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Convert esports odds lines into sportsbook_game_lines rows."""
        captured_at = captured_at or datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for line in lines:
            rows.append({
                "event_id": line.fixture_id,
                "sport": line.sport,
                "home_team": line.home_team,
                "away_team": line.away_team,
                "bookmaker": line.bookmaker,
                "commence_time": line.commence_time,
                "home_price": line.home_price,
                "away_price": line.away_price,
                "draw_price": None,
                "spread_home": None,
                "total": None,
                "captured_at": captured_at,
            })
        return rows

    @staticmethod
    def _parse_odds_response(
        data: dict[str, Any],
        sport: str,
    ) -> list[EsportsOddsLine]:
        """Parse a full odds response from OddsPapi."""
        lines: list[EsportsOddsLine] = []
        fixture_id = str(data.get("fixtureId", ""))
        p1_name = str(data.get("participant1Name", data.get("participant1", "")))
        p2_name = str(data.get("participant2Name", data.get("participant2", "")))

        if not p1_name or not p2_name or not fixture_id:
            return []

        # Parse commence time
        commence = None
        start_str = data.get("startTime")
        if start_str:
            try:
                commence = datetime.fromisoformat(
                    str(start_str).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        bm_odds = data.get("bookmakerOdds", {})
        if not isinstance(bm_odds, dict):
            return []

        for bm_slug, bm_data in bm_odds.items():
            if not isinstance(bm_data, dict):
                continue

            markets = bm_data.get("markets", {})
            if not isinstance(markets, dict):
                continue

            # Find the match winner market (could be 171, 181, 121, etc.)
            match_winner = None
            for mkt_id in _MATCH_WINNER_MARKETS:
                if mkt_id in markets:
                    match_winner = markets[mkt_id]
                    break

            if not match_winner or not isinstance(match_winner, dict):
                continue

            outcomes = match_winner.get("outcomes", {})
            if not isinstance(outcomes, dict):
                continue

            # Outcomes use paired IDs: first=participant1, second=participant2
            # e.g. 171/172, 181/182, 121/122.  Take the two sorted outcome keys.
            outcome_keys = sorted(outcomes.keys())
            if len(outcome_keys) < 2:
                continue

            p1_outcome = outcomes.get(outcome_keys[0], {})
            p2_outcome = outcomes.get(outcome_keys[1], {})

            p1_price = _extract_price(p1_outcome)
            p2_price = _extract_price(p2_outcome)

            if p1_price is not None and p2_price is not None and p1_price > 1.0 and p2_price > 1.0:
                lines.append(EsportsOddsLine(
                    fixture_id=fixture_id,
                    home_team=p1_name,
                    away_team=p2_name,
                    home_price=p1_price,
                    away_price=p2_price,
                    bookmaker=bm_slug,
                    commence_time=commence,
                    sport=sport,
                ))

        return lines


def _extract_price(outcome: dict) -> float | None:
    """Extract decimal price from an OddsPapi outcome."""
    if not isinstance(outcome, dict):
        return None

    # Try players → 0 → price
    players = outcome.get("players", {})
    if isinstance(players, dict):
        first = players.get("0", {})
        if isinstance(first, dict) and first.get("active"):
            try:
                return float(first["price"])
            except (KeyError, ValueError, TypeError):
                pass

    # Fallback: direct price field
    try:
        return float(outcome.get("price", 0)) or None
    except (ValueError, TypeError):
        return None
