"""Sportsbook odds feed from SportsGameOdds API.

Drop-in alternative to The Odds API for fetching Pinnacle and consensus
sportsbook lines.  Produces the same ``GameOdds`` data model and
``sportsbook_game_lines`` table rows.

https://sportsgameodds.com/docs
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from moneygone.data.sports.odds import GameOdds

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.sportsgameodds.com/v2"

# Mapping from our league shorthand to SportsGameOdds leagueID.
LEAGUE_IDS: dict[str, str] = {
    "nba": "NBA",
    "nfl": "NFL",
    "nhl": "NHL",
    "mlb": "MLB",
    "ncaab": "NCAAB",
    "ncaaf": "NCAAF",
    "soccer_epl": "EPL",
    "soccer_usa_mls": "MLS",
    "soccer_spain_la_liga": "LA_LIGA",
    "soccer_germany_bundesliga": "BUNDESLIGA",
    "soccer_italy_serie_a": "SERIE_A",
    "soccer_france_ligue_one": "LIGUE_1",
    "ufc": "UFC",
    "mma": "UFC",
    "tennis_atp": "ATP",
    "tennis_wta": "WTA",
}

# OddID components for each market type.
_MARKET_ODD_IDS: dict[str, list[str]] = {
    "h2h": [
        "points-home-game-ml-home",
        "points-away-game-ml-away",
    ],
    "spreads": [
        "points-home-game-sp-home",
        "points-away-game-sp-away",
    ],
    "totals": [
        "points-all-game-ou-over",
        "points-all-game-ou-under",
    ],
}

# Soccer needs 3-way moneyline (draw).
_SOCCER_EXTRA_ODD_IDS = [
    "points-home-game-ml3way-home",
    "points-away-game-ml3way-away",
    "points-home-game-ml3way-draw",
]

_SOCCER_LEAGUES = {
    "EPL", "MLS", "LA_LIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1",
    "UEFA_CHAMPIONS_LEAGUE",
}


def _american_to_decimal(american: float | str | None) -> float | None:
    """Convert American odds to decimal format.

    +150 → 2.50,  -200 → 1.50
    """
    if american is None:
        return None
    try:
        val = float(american)
    except (ValueError, TypeError):
        return None
    if val == 0:
        return None
    if val > 0:
        return round(val / 100 + 1, 4)
    return round(100 / abs(val) + 1, 4)


class SportsGameOddsFeed:
    """Fetches sportsbook odds from SportsGameOdds API.

    Parameters
    ----------
    api_key:
        SportsGameOdds API key.  Falls back to
        ``SPORTSGAMEODDS_API_KEY`` env var.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("SPORTSGAMEODDS_API_KEY", "")
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout
        self._requests_used: int = 0
        self._last_requests_remaining: int | None = None

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    @property
    def requests_remaining(self) -> int | None:
        """Estimated remaining requests, if tracked."""
        return self._last_requests_remaining

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "User-Agent": "MoneyGone/1.0",
                    "x-api-key": self._api_key,
                },
            )
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    async def _get_json(
        self, url: str, params: dict[str, str] | None = None,
    ) -> Any | None:
        if not self._api_key:
            logger.warning("sportsgameodds.no_api_key", url=url)
            return None

        client = await self._get_client()
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            self._requests_used += 1

            # Track remaining from response headers if available.
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                try:
                    self._last_requests_remaining = int(remaining)
                except ValueError:
                    pass

            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "sportsgameodds.http_error",
                url=url,
                status=exc.response.status_code,
                body=exc.response.text[:300],
            )
            return None
        except httpx.TransportError as exc:
            logger.warning("sportsgameodds.transport_error", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # League key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _league_id(league: str) -> str:
        """Resolve a league shorthand to SportsGameOdds leagueID."""
        key = league.lower()
        if key in LEAGUE_IDS:
            return LEAGUE_IDS[key]
        # Assume already a valid leagueID.
        return key.upper()

    # ------------------------------------------------------------------
    # Public API — mirrors OddsAPIFeed interface
    # ------------------------------------------------------------------

    async def get_upcoming_games(
        self,
        league: str,
        *,
        bookmakers: list[str] | None = None,
        markets: list[str] | None = None,
    ) -> list[GameOdds]:
        """Fetch upcoming games with bookmaker odds.

        Parameters
        ----------
        league:
            League shorthand (e.g. ``"nba"``) or SportsGameOdds leagueID.
        bookmakers:
            Optional bookmaker IDs to filter (e.g. ``["pinnacle"]``).
            Not directly filterable in the API — we filter client-side.
        markets:
            Market types: ``"h2h"``, ``"spreads"``, ``"totals"``.
        """
        league_id = self._league_id(league)
        market_keys = markets or ["h2h"]

        # Build oddID filter for efficient API response.
        odd_ids: list[str] = []
        for mkt in market_keys:
            odd_ids.extend(_MARKET_ODD_IDS.get(mkt, []))

        # Add 3-way moneyline for soccer leagues.
        if league_id in _SOCCER_LEAGUES and "h2h" in market_keys:
            odd_ids.extend(_SOCCER_EXTRA_ODD_IDS)

        params: dict[str, str] = {
            "leagueID": league_id,
            "oddsAvailable": "true",
            "oddID": ",".join(odd_ids),
            "limit": "50",
        }

        data = await self._get_json(f"{_BASE_URL}/events", params=params)
        if data is None:
            return []

        events = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(events, list):
            events = [events] if isinstance(events, dict) else []

        bookmaker_filter = set(b.lower() for b in bookmakers) if bookmakers else None

        games: list[GameOdds] = []
        for evt in events:
            game = self._parse_event(evt, bookmaker_filter, market_keys)
            if game is not None:
                games.append(game)

        logger.info(
            "sportsgameodds.fetched",
            league=league_id,
            games=len(games),
            requests_used=self._requests_used,
        )
        return games

    async def get_active_tennis_keys(self) -> list[str]:
        """Return active tennis leagueIDs.

        SportsGameOdds uses flat leagueIDs for tennis (ATP, WTA),
        so we just return them directly.
        """
        return ["ATP", "WTA"]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_event(
        self,
        evt: dict[str, Any],
        bookmaker_filter: set[str] | None,
        market_keys: list[str],
    ) -> GameOdds | None:
        """Parse a SportsGameOdds event into a GameOdds object."""
        event_id = evt.get("eventID", "")
        if not event_id:
            return None

        # Extract team names.
        teams = evt.get("teams", {})
        home_info = teams.get("home", {})
        away_info = teams.get("away", {})
        home_team = home_info.get("names", {}).get("short") or home_info.get("teamID", "")
        away_team = away_info.get("names", {}).get("short") or away_info.get("teamID", "")

        # Extract commence time.
        status = evt.get("status", {})
        commence_time = status.get("startsAt", "")

        # Skip events that have already ended.
        if status.get("ended") or status.get("finalized"):
            return None

        # Parse odds into the bookmaker list format expected by GameOdds.
        odds_data = evt.get("odds", {})
        bookmakers_list = self._build_bookmakers_list(
            odds_data, home_team, away_team, bookmaker_filter, market_keys,
        )

        if not bookmakers_list:
            return None

        return GameOdds(
            event_id=event_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            bookmakers=bookmakers_list,
        )

    def _build_bookmakers_list(
        self,
        odds_data: dict[str, Any],
        home_team: str,
        away_team: str,
        bookmaker_filter: set[str] | None,
        market_keys: list[str],
    ) -> list[dict[str, Any]]:
        """Transform SportsGameOdds odds structure into OddsAPI-compatible bookmakers list."""
        # Collect per-bookmaker data across all oddIDs.
        bookmaker_markets: dict[str, dict[str, list[dict[str, Any]]]] = {}

        for odd_id, odd_info in odds_data.items():
            if not isinstance(odd_info, dict):
                continue
            by_bookmaker = odd_info.get("byBookmaker", {})
            if not by_bookmaker:
                continue

            for bm_id, bm_data in by_bookmaker.items():
                if not isinstance(bm_data, dict):
                    continue
                if bookmaker_filter and bm_id.lower() not in bookmaker_filter:
                    continue
                if not bm_data.get("available", True):
                    continue

                american_odds = bm_data.get("odds")
                decimal_odds = _american_to_decimal(american_odds)
                if decimal_odds is None:
                    continue

                spread_val = bm_data.get("spread")
                ou_val = bm_data.get("overUnder")

                # Determine which market this oddID belongs to and build outcome.
                market_key, outcome = self._classify_odd(
                    odd_id, decimal_odds, home_team, away_team,
                    spread_val, ou_val,
                )
                if market_key is None or market_key not in market_keys:
                    continue

                bm_lower = bm_id.lower()
                if bm_lower not in bookmaker_markets:
                    bookmaker_markets[bm_lower] = {}
                if market_key not in bookmaker_markets[bm_lower]:
                    bookmaker_markets[bm_lower][market_key] = []
                bookmaker_markets[bm_lower][market_key].append(outcome)

        # Assemble into the list format.
        result: list[dict[str, Any]] = []
        for bm_key, markets in bookmaker_markets.items():
            market_list = []
            for mkt_key, outcomes in markets.items():
                market_list.append({"key": mkt_key, "outcomes": outcomes})
            result.append({
                "key": bm_key,
                "title": bm_key,
                "markets": market_list,
            })

        return result

    @staticmethod
    def _classify_odd(
        odd_id: str,
        decimal_odds: float,
        home_team: str,
        away_team: str,
        spread_val: str | None,
        ou_val: str | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Classify an oddID into market type and build an outcome dict."""
        parts = odd_id.split("-")
        if len(parts) < 5:
            return None, {}

        bet_type = parts[3]   # ml, sp, ou, ml3way
        side = parts[4]       # home, away, over, under, draw

        if bet_type in ("ml", "ml3way"):
            if side == "home":
                return "h2h", {"name": home_team, "price": decimal_odds}
            elif side == "away":
                return "h2h", {"name": away_team, "price": decimal_odds}
            elif side == "draw":
                return "h2h", {"name": "Draw", "price": decimal_odds}

        elif bet_type == "sp":
            point = None
            if spread_val is not None:
                try:
                    point = float(spread_val)
                except (ValueError, TypeError):
                    pass
            if side == "home":
                return "spreads", {
                    "name": home_team,
                    "price": decimal_odds,
                    "point": point,
                }
            elif side == "away":
                return "spreads", {
                    "name": away_team,
                    "price": decimal_odds,
                    "point": point,
                }

        elif bet_type == "ou":
            point = None
            if ou_val is not None:
                try:
                    point = float(ou_val)
                except (ValueError, TypeError):
                    pass
            if side == "over":
                return "totals", {
                    "name": "Over",
                    "price": decimal_odds,
                    "point": point,
                }
            elif side == "under":
                return "totals", {
                    "name": "Under",
                    "price": decimal_odds,
                    "point": point,
                }

        return None, {}

    # ------------------------------------------------------------------
    # Line history rows — same output contract as OddsAPIFeed
    # ------------------------------------------------------------------

    def build_line_history_rows(
        self,
        league: str,
        games: list[GameOdds],
        *,
        captured_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Convert game odds into datastore rows for line-history persistence."""
        captured_at = captured_at or datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []

        for game in games:
            for bookmaker in game.bookmakers:
                line = self._extract_bookmaker_line(game, bookmaker)
                home_price = line["home_price"]
                away_price = line["away_price"]
                if home_price is None or away_price is None:
                    continue
                rows.append(
                    {
                        "event_id": game.event_id,
                        "sport": league.lower(),
                        "home_team": game.home_team,
                        "away_team": game.away_team,
                        "bookmaker": str(
                            bookmaker.get("key", bookmaker.get("title", ""))
                        ).lower(),
                        "commence_time": self._parse_timestamp(game.commence_time),
                        "home_price": home_price,
                        "away_price": away_price,
                        "draw_price": line.get("draw_price"),
                        "spread_home": line["spread_home"],
                        "total": line["total"],
                        "captured_at": captured_at,
                    }
                )

        return rows

    @staticmethod
    def _extract_bookmaker_line(
        game: GameOdds,
        bookmaker: dict[str, Any],
    ) -> dict[str, float | None]:
        """Extract one bookmaker's line snapshot for a game."""
        home_price: float | None = None
        away_price: float | None = None
        draw_price: float | None = None
        spread_home: float | None = None
        total: float | None = None

        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h":
                for outcome in outcomes:
                    try:
                        price = float(outcome.get("price", 0))
                    except (ValueError, TypeError):
                        continue
                    if outcome.get("name") == game.home_team:
                        home_price = price
                    elif outcome.get("name") == game.away_team:
                        away_price = price
                    elif outcome.get("name") == "Draw":
                        draw_price = price

            elif key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") != game.home_team:
                        continue
                    try:
                        spread_home = float(outcome.get("point", 0))
                    except (ValueError, TypeError):
                        spread_home = None

            elif key == "totals":
                for outcome in outcomes:
                    if str(outcome.get("name", "")).lower() != "over":
                        continue
                    try:
                        total = float(outcome.get("point", 0))
                    except (ValueError, TypeError):
                        total = None

        return {
            "home_price": home_price,
            "away_price": away_price,
            "draw_price": draw_price,
            "spread_home": spread_home,
            "total": total,
        }

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse an ISO timestamp into a timezone-aware datetime."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
