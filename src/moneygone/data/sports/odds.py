"""Sportsbook odds and player props feed from The Odds API.

Fetches consensus sportsbook lines for player prop markets so the
model can compare Kalshi prices against DraftKings / FanDuel / BetMGM
consensus.  Requires an API key (set via ``ODDS_API_KEY`` env var).

https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

_ODDS_BASE = "https://api.the-odds-api.com/v4"

# Mapping from our league shorthand to The Odds API sport keys.
SPORT_KEYS: dict[str, str] = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
    "ncaab": "basketball_ncaab",
    "ncaaf": "americanfootball_ncaaf",
}

# Common player prop market keys.
PLAYER_PROP_MARKETS: dict[str, list[str]] = {
    "nba": [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_points_rebounds_assists",
    ],
    "nfl": [
        "player_pass_yds",
        "player_pass_tds",
        "player_rush_yds",
        "player_receptions",
        "player_reception_yds",
    ],
    "nhl": [
        "player_points",
        "player_goals",
        "player_assists",
        "player_shots_on_goal",
    ],
    "mlb": [
        "batter_hits",
        "batter_total_bases",
        "batter_rbis",
        "pitcher_strikeouts",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PropLine:
    """A single player prop line from one bookmaker."""

    player_name: str
    market: str  # e.g. "player_points", "player_rebounds"
    line: float  # the over/under threshold
    over_price: float  # decimal odds for over
    under_price: float  # decimal odds for under
    bookmaker: str


@dataclass(frozen=True, slots=True)
class MoneylineOdds:
    """Head-to-head moneyline odds from one bookmaker."""

    home_team: str
    away_team: str
    home_price: float  # decimal odds
    away_price: float  # decimal odds
    bookmaker: str


@dataclass(frozen=True, slots=True)
class GameOdds:
    """Odds data for a single upcoming game."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: str  # ISO 8601
    bookmakers: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------


class OddsAPIFeed:
    """Fetches sportsbook odds and player props from The Odds API.

    Parameters
    ----------
    api_key:
        The Odds API key.  Falls back to ``ODDS_API_KEY`` env var.
    client:
        Optional pre-configured ``httpx.AsyncClient``.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"User-Agent": "MoneyGone/1.0"},
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

    async def _get_json(self, url: str, params: dict[str, str] | None = None) -> Any | None:
        if not self._api_key:
            logger.warning("odds.no_api_key", url=url)
            return None

        client = await self._get_client()
        full_params = {"apiKey": self._api_key}
        if params:
            full_params.update(params)

        try:
            resp = await client.get(url, params=full_params)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                logger.debug("odds.quota", remaining=remaining)
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "odds.http_error",
                url=url,
                status=exc.response.status_code,
            )
            return None
        except httpx.TransportError as exc:
            logger.warning("odds.transport_error", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Sport key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _sport_key(league: str) -> str:
        """Resolve a league shorthand to The Odds API sport key."""
        key = league.lower()
        if key in SPORT_KEYS:
            return SPORT_KEYS[key]
        # Assume already a full sport key.
        return key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_upcoming_games(self, league: str) -> list[GameOdds]:
        """Fetch upcoming games with bookmaker odds.

        Parameters
        ----------
        league:
            League shorthand (e.g. ``"nba"``) or full Odds API sport key.
        """
        sport_key = self._sport_key(league)
        url = f"{_ODDS_BASE}/sports/{sport_key}/odds"
        data = await self._get_json(
            url,
            params={
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
            },
        )
        if data is None or not isinstance(data, list):
            return []

        games: list[GameOdds] = []
        for evt in data:
            home = away = ""
            for team_key in ("home_team", "homeTeam"):
                if team_key in evt:
                    home = evt[team_key]
                    break
            for team_key in ("away_team", "awayTeam"):
                if team_key in evt:
                    away = evt[team_key]
                    break

            games.append(
                GameOdds(
                    event_id=str(evt.get("id", "")),
                    home_team=home,
                    away_team=away,
                    commence_time=evt.get("commence_time", ""),
                    bookmakers=evt.get("bookmakers", []),
                )
            )
        return games

    async def get_player_props(
        self,
        league: str,
        event_id: str,
        markets: list[str] | None = None,
    ) -> list[PropLine]:
        """Fetch player prop lines for a specific event.

        Parameters
        ----------
        league:
            League shorthand or full Odds API sport key.
        event_id:
            The Odds API event ID (from :meth:`get_upcoming_games`).
        markets:
            List of player prop market keys.  Defaults to the standard
            markets for the league.
        """
        sport_key = self._sport_key(league)
        if markets is None:
            markets = PLAYER_PROP_MARKETS.get(league.lower(), ["player_points"])

        url = f"{_ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds"
        data = await self._get_json(
            url,
            params={
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
            },
        )
        if data is None:
            return []

        return self._parse_player_props(data)

    async def get_moneyline_odds(self, league: str) -> list[MoneylineOdds]:
        """Fetch head-to-head moneyline odds for upcoming games.

        Parameters
        ----------
        league:
            League shorthand or full Odds API sport key.
        """
        sport_key = self._sport_key(league)
        url = f"{_ODDS_BASE}/sports/{sport_key}/odds"
        data = await self._get_json(
            url,
            params={
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )
        if data is None or not isinstance(data, list):
            return []

        results: list[MoneylineOdds] = []
        for evt in data:
            home = evt.get("home_team", "")
            away = evt.get("away_team", "")
            for bm in evt.get("bookmakers", []):
                bm_name = bm.get("title", bm.get("key", ""))
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    outcomes = mkt.get("outcomes", [])
                    home_price = away_price = 0.0
                    for outcome in outcomes:
                        if outcome.get("name") == home:
                            home_price = float(outcome.get("price", 0))
                        elif outcome.get("name") == away:
                            away_price = float(outcome.get("price", 0))
                    if home_price > 0 and away_price > 0:
                        results.append(
                            MoneylineOdds(
                                home_team=home,
                                away_team=away,
                                home_price=home_price,
                                away_price=away_price,
                                bookmaker=bm_name,
                            )
                        )
        return results

    async def get_spreads(self, league: str) -> dict[str, float]:
        """Return a mapping of ``event_id`` -> consensus spread (home perspective).

        Positive spread means the home team is an underdog.
        """
        games = await self.get_upcoming_games(league)
        spreads: dict[str, float] = {}
        for game in games:
            spread_values: list[float] = []
            for bm in game.bookmakers:
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "spreads":
                        continue
                    for outcome in mkt.get("outcomes", []):
                        if outcome.get("name") == game.home_team:
                            try:
                                spread_values.append(float(outcome.get("point", 0)))
                            except (ValueError, TypeError):
                                pass
            if spread_values:
                spreads[game.event_id] = sum(spread_values) / len(spread_values)
        return spreads

    async def get_game_winner_snapshots(
        self,
        league: str,
        opening_lines: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """Build game-winner feature snapshot dicts for all upcoming games.

        Each dict can be set directly as ``context.sports_snapshot`` when
        building game-winner features.

        Parameters
        ----------
        league:
            League shorthand (e.g. ``"nba"``) or full Odds API sport key.
        opening_lines:
            Optional mapping of ``event_id`` -> ``{"home": decimal_odds,
            "away": decimal_odds}`` representing the opening line.  Used
            to compute :class:`~moneygone.features.game_winner_features.MoneylineMovement`.
            When not provided those snapshot keys will be ``None``.

        Returns
        -------
        list[dict]
            One dict per game, with keys matching ``game_winner_features.py``
            snapshot contract.
        """
        games = await self.get_upcoming_games(league)
        snapshots: list[dict] = []

        for game in games:
            h2h_home: list[float] = []
            h2h_away: list[float] = []
            spread_home: list[float] = []
            totals: list[float] = []

            for bm in game.bookmakers:
                for mkt in bm.get("markets", []):
                    key = mkt.get("key")
                    outcomes = mkt.get("outcomes", [])
                    if key == "h2h":
                        for o in outcomes:
                            try:
                                price = float(o.get("price", 0))
                            except (ValueError, TypeError):
                                continue
                            if o.get("name") == game.home_team:
                                h2h_home.append(price)
                            elif o.get("name") == game.away_team:
                                h2h_away.append(price)
                    elif key == "spreads":
                        for o in outcomes:
                            if o.get("name") == game.home_team:
                                try:
                                    spread_home.append(float(o.get("point", 0)))
                                except (ValueError, TypeError):
                                    pass
                    elif key == "totals":
                        for o in outcomes:
                            if o.get("name", "").lower() == "over":
                                try:
                                    totals.append(float(o.get("point", 0)))
                                except (ValueError, TypeError):
                                    pass

            curr_home_odds = sum(h2h_home) / len(h2h_home) if h2h_home else None
            curr_away_odds = sum(h2h_away) / len(h2h_away) if h2h_away else None

            # Consensus normalised win probabilities (remove overround).
            sb_home_win_prob = None
            if curr_home_odds and curr_away_odds:
                raw_h = 1.0 / curr_home_odds
                raw_a = 1.0 / curr_away_odds
                total_implied = raw_h + raw_a
                if total_implied > 0:
                    sb_home_win_prob = raw_h / total_implied

            opening = (opening_lines or {}).get(game.event_id, {})

            snap: dict = {
                "event_id": game.event_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "commence_time": game.commence_time,
                "sport": league.lower(),
                # Sportsbook win probability.
                "sportsbook_home_win_prob": sb_home_win_prob,
                # Raw moneyline odds (current).
                "current_moneyline_home": curr_home_odds,
                "current_moneyline_away": curr_away_odds,
                # Opening line for movement calculation.
                "opening_moneyline_home": opening.get("home"),
                "opening_moneyline_away": opening.get("away"),
                # Spread and total.
                "spread": sum(spread_home) / len(spread_home) if spread_home else None,
                "total": sum(totals) / len(totals) if totals else None,
                # Placeholders populated by downstream (injury feed, ESPN).
                "home_key_injuries": None,
                "away_key_injuries": None,
                "home_injury_severity": None,
                "away_injury_severity": None,
                "home_team_rating": None,
                "away_team_rating": None,
                "public_pct_home": None,
                "public_pct_away": None,
                # Set by the caller to orient features toward home or away.
                "is_home_team": None,
                "kalshi_implied_prob": None,
            }
            snapshots.append(snap)

        return snapshots

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_player_props(data: dict[str, Any] | list) -> list[PropLine]:
        """Parse The Odds API player props response."""
        props: list[PropLine] = []
        try:
            bookmakers = []
            if isinstance(data, dict):
                bookmakers = data.get("bookmakers", [])
            elif isinstance(data, list):
                bookmakers = data

            for bm in bookmakers:
                bm_name = bm.get("title", bm.get("key", ""))
                for mkt in bm.get("markets", []):
                    market_key = mkt.get("key", "")
                    outcomes = mkt.get("outcomes", [])

                    # Player props come in pairs: Over/Under for each player.
                    # Group by player + line.
                    player_lines: dict[tuple[str, float], dict[str, float]] = {}
                    for outcome in outcomes:
                        player = outcome.get("description", outcome.get("name", ""))
                        line = float(outcome.get("point", 0))
                        price = float(outcome.get("price", 0))
                        side = outcome.get("name", "").lower()

                        key = (player, line)
                        if key not in player_lines:
                            player_lines[key] = {"over": 0.0, "under": 0.0}

                        if side == "over":
                            player_lines[key]["over"] = price
                        elif side == "under":
                            player_lines[key]["under"] = price

                    for (player, line), prices in player_lines.items():
                        if player and (prices["over"] > 0 or prices["under"] > 0):
                            props.append(
                                PropLine(
                                    player_name=player,
                                    market=market_key,
                                    line=line,
                                    over_price=prices["over"],
                                    under_price=prices["under"],
                                    bookmaker=bm_name,
                                )
                            )
        except Exception:
            logger.warning("odds.parse_props_error", exc_info=True)

        return props
