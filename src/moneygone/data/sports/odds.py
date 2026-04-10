"""Sportsbook odds and player props feed from The Odds API.

Fetches consensus sportsbook lines for player prop markets so the
model can compare Kalshi prices against DraftKings / FanDuel / BetMGM
consensus.  Requires an API key (set via ``ODDS_API_KEY`` env var).

https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from moneygone.data.store import DataStore
    from moneygone.data.sports.power_ratings import ESPNPowerRatings, TeamRating
    from moneygone.data.sports.stats import PlayerStatsFeed, TeamInjurySummary

_ODDS_BASE = "https://api.the-odds-api.com/v4"

# Mapping from our league shorthand to The Odds API sport keys.
SPORT_KEYS: dict[str, str] = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
    "ncaab": "basketball_ncaab",
    "ncaaf": "americanfootball_ncaaf",
    "soccer_epl": "soccer_epl",
    "soccer_usa_mls": "soccer_usa_mls",
    "soccer_spain_la_liga": "soccer_spain_la_liga",
    "soccer_germany_bundesliga": "soccer_germany_bundesliga",
    "soccer_italy_serie_a": "soccer_italy_serie_a",
    "soccer_france_ligue_one": "soccer_france_ligue_one",
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
        self._last_requests_remaining: int | None = None

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    @property
    def requests_remaining(self) -> int | None:
        """Last seen Odds API remaining-request header, if available."""
        return self._last_requests_remaining

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
                try:
                    self._last_requests_remaining = int(remaining)
                except ValueError:
                    self._last_requests_remaining = None
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
            League shorthand (e.g. ``"nba"``) or full Odds API sport key.
        bookmakers:
            Optional Odds API bookmaker keys to filter the response.
        markets:
            Optional Odds API market keys. Defaults to ``["h2h", "spreads",
            "totals"]`` for richer game-winner snapshots, but collectors can
            pass ``["h2h"]`` to minimise credit spend.
        """
        sport_key = self._sport_key(league)
        url = f"{_ODDS_BASE}/sports/{sport_key}/odds"
        market_keys = markets or ["h2h", "spreads", "totals"]
        params = {
            "regions": "us",
            "markets": ",".join(market_keys),
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        data = await self._get_json(
            url,
            params=params,
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
        *,
        store: DataStore | None = None,
        stats_feed: PlayerStatsFeed | None = None,
        power_ratings: ESPNPowerRatings | None = None,
        key_minutes_threshold: float = 20.0,
        movement_bookmaker: str = "pinnacle",
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
        store:
            Optional datastore containing previously recorded sportsbook
            snapshots. When available, opening lines are loaded from the
            earliest stored row for ``movement_bookmaker``.
        stats_feed:
            Optional ESPN stats feed.  When omitted, the method creates
            a temporary one to enrich snapshots with injury data.
        power_ratings:
            Optional ESPN power-rating feed.  When omitted, the method
            creates a temporary one to enrich snapshots with team ratings.
        key_minutes_threshold:
            Minimum average minutes per game for an injured player to
            count as a "key" absence.
        movement_bookmaker:
            Bookmaker to use for opening/current line-movement fields.
            Defaults to Pinnacle, the sharpest book available in this stack.

        Returns
        -------
        list[dict]
            One dict per game, with keys matching ``game_winner_features.py``
            snapshot contract.
        """
        created_stats_feed = False
        created_power_ratings = False

        if stats_feed is None:
            from moneygone.data.sports.stats import PlayerStatsFeed

            stats_feed = PlayerStatsFeed()
            created_stats_feed = True

        if power_ratings is None:
            from moneygone.data.sports.power_ratings import ESPNPowerRatings

            power_ratings = ESPNPowerRatings()
            created_power_ratings = True

        try:
            games = await self.get_upcoming_games(
                league,
                bookmakers=["pinnacle", "draftkings", "fanduel"],
            )
            if not games:
                games = await self.get_upcoming_games(league)
            snapshots: list[dict] = []
            event_ids = [game.event_id for game in games if game.event_id]
            stored_opening_lines: dict[str, dict[str, Any]] = {}
            stored_latest_lines: dict[str, dict[str, Any]] = {}
            if store is not None and event_ids:
                stored_opening_lines = store.get_opening_sportsbook_lines(
                    bookmaker=movement_bookmaker,
                    sport=league.lower(),
                    event_ids=event_ids,
                )
                stored_latest_lines = store.get_latest_sportsbook_lines(
                    bookmaker=movement_bookmaker,
                    sport=league.lower(),
                    event_ids=event_ids,
                )
            ratings = await power_ratings.get_ratings(league)
            injury_summaries = await self._load_team_injury_summaries(
                league,
                games,
                stats_feed,
                power_ratings,
                ratings,
                key_minutes_threshold=key_minutes_threshold,
            )

            for game in games:
                moneyline_data = self._extract_moneyline_data(game)
                current_home_odds = moneyline_data["consensus_home"]
                current_away_odds = moneyline_data["consensus_away"]
                pinnacle_home_odds = moneyline_data["pinnacle_home"]
                pinnacle_away_odds = moneyline_data["pinnacle_away"]

                sb_home_win_prob = self._normalised_home_probability(
                    current_home_odds,
                    current_away_odds,
                )
                pinnacle_home_win_prob = self._normalised_home_probability(
                    pinnacle_home_odds,
                    pinnacle_away_odds,
                )

                opening = (opening_lines or {}).get(game.event_id, {})
                stored_opening = stored_opening_lines.get(game.event_id, {})
                stored_latest = stored_latest_lines.get(game.event_id, {})
                if not opening and stored_opening:
                    opening = {
                        "home": stored_opening.get("home_price"),
                        "away": stored_opening.get("away_price"),
                    }

                movement_home_odds = (
                    pinnacle_home_odds
                    or stored_latest.get("home_price")
                    or current_home_odds
                )
                movement_away_odds = (
                    pinnacle_away_odds
                    or stored_latest.get("away_price")
                    or current_away_odds
                )
                movement_source = (
                    movement_bookmaker
                    if (
                        (pinnacle_home_odds is not None and pinnacle_away_odds is not None)
                        or stored_latest
                    )
                    else "consensus"
                )
                home_rating = power_ratings.lookup(game.home_team, ratings)
                away_rating = power_ratings.lookup(game.away_team, ratings)
                home_injury = injury_summaries.get(game.home_team)
                away_injury = injury_summaries.get(game.away_team)

                snap: dict[str, Any] = {
                    "event_id": game.event_id,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "commence_time": game.commence_time,
                    "sport": league.lower(),
                    # Consensus sportsbook probability across available books.
                    "sportsbook_home_win_prob": sb_home_win_prob,
                    # Pinnacle-only line, our primary sharp-money proxy.
                    "pinnacle_home_win_prob": pinnacle_home_win_prob,
                    "pinnacle_moneyline_home": pinnacle_home_odds,
                    "pinnacle_moneyline_away": pinnacle_away_odds,
                    # Raw moneyline odds used for line-movement calculations.
                    # Prefer the sharp book so opening/current compare like-for-like.
                    "current_moneyline_home": movement_home_odds,
                    "current_moneyline_away": movement_away_odds,
                    # Opening line for movement calculation, ideally from stored
                    # pre-game Pinnacle history.
                    "opening_moneyline_home": opening.get("home"),
                    "opening_moneyline_away": opening.get("away"),
                    "movement_line_source": movement_source,
                    "opening_line_captured_at": stored_opening.get("captured_at"),
                    "latest_line_captured_at": stored_latest.get("captured_at"),
                    # Spread and total.
                    "spread": moneyline_data["spread"],
                    "total": moneyline_data["total"],
                    # Injury and power-rating enrichment.
                    "home_key_injuries": home_injury.key_injuries if home_injury else None,
                    "away_key_injuries": away_injury.key_injuries if away_injury else None,
                    "home_injury_severity": home_injury.injury_severity if home_injury else None,
                    "away_injury_severity": away_injury.injury_severity if away_injury else None,
                    "home_team_rating": home_rating.rating if home_rating else None,
                    "away_team_rating": away_rating.rating if away_rating else None,
                    "home_team_win_pct": home_rating.win_pct if home_rating else None,
                    "away_team_win_pct": away_rating.win_pct if away_rating else None,
                    "public_pct_home": None,
                    "public_pct_away": None,
                    # Set by the caller to orient features toward home or away.
                    "is_home_team": None,
                    "kalshi_implied_prob": None,
                }
                snapshots.append(snap)

            return snapshots
        finally:
            if created_stats_feed and stats_feed is not None:
                await stats_feed.close()
            if created_power_ratings and power_ratings is not None:
                await power_ratings.close()

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

    @staticmethod
    def _normalised_home_probability(
        home_odds: float | None,
        away_odds: float | None,
    ) -> float | None:
        """Convert decimal moneyline odds into an overround-removed home win prob."""
        if home_odds is None or away_odds is None or home_odds <= 1.0 or away_odds <= 1.0:
            return None

        raw_h = 1.0 / home_odds
        raw_a = 1.0 / away_odds
        total = raw_h + raw_a
        if total <= 0:
            return None
        return raw_h / total

    @staticmethod
    def _extract_moneyline_data(game: GameOdds) -> dict[str, float | None]:
        """Extract consensus and Pinnacle-specific line data for a game."""
        h2h_home: list[float] = []
        h2h_away: list[float] = []
        spread_home: list[float] = []
        totals: list[float] = []
        pinnacle_home: float | None = None
        pinnacle_away: float | None = None

        for bookmaker in game.bookmakers:
            bm_key = str(bookmaker.get("key", "")).lower()
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    home_price: float | None = None
                    away_price: float | None = None
                    for outcome in outcomes:
                        try:
                            price = float(outcome.get("price", 0))
                        except (ValueError, TypeError):
                            continue
                        if outcome.get("name") == game.home_team:
                            h2h_home.append(price)
                            home_price = price
                        elif outcome.get("name") == game.away_team:
                            h2h_away.append(price)
                            away_price = price

                    if bm_key == "pinnacle":
                        pinnacle_home = home_price
                        pinnacle_away = away_price

                elif key == "spreads":
                    for outcome in outcomes:
                        if outcome.get("name") == game.home_team:
                            try:
                                spread_home.append(float(outcome.get("point", 0)))
                            except (ValueError, TypeError):
                                continue

                elif key == "totals":
                    for outcome in outcomes:
                        if str(outcome.get("name", "")).lower() == "over":
                            try:
                                totals.append(float(outcome.get("point", 0)))
                            except (ValueError, TypeError):
                                continue

        return {
            "consensus_home": sum(h2h_home) / len(h2h_home) if h2h_home else None,
            "consensus_away": sum(h2h_away) / len(h2h_away) if h2h_away else None,
            "pinnacle_home": pinnacle_home,
            "pinnacle_away": pinnacle_away,
            "spread": sum(spread_home) / len(spread_home) if spread_home else None,
            "total": sum(totals) / len(totals) if totals else None,
        }

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
                    elif str(outcome.get("name", "")).lower() == "draw":
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
                        "bookmaker": str(bookmaker.get("key", bookmaker.get("title", ""))).lower(),
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
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse a basic ISO timestamp into a timezone-aware datetime."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    async def _load_team_injury_summaries(
        self,
        league: str,
        games: list[GameOdds],
        stats_feed: PlayerStatsFeed,
        power_ratings: ESPNPowerRatings,
        ratings: dict[str, TeamRating],
        *,
        key_minutes_threshold: float,
    ) -> dict[str, TeamInjurySummary]:
        """Fetch injury summaries for all teams present in ``games``."""
        sport_lookup = {
            "nba": "basketball",
            "ncaab": "basketball",
            "nfl": "football",
            "ncaaf": "football",
            "nhl": "hockey",
            "mlb": "baseball",
            "soccer_epl": "soccer",
            "soccer_usa_mls": "soccer",
            "soccer_spain_la_liga": "soccer",
            "soccer_germany_bundesliga": "soccer",
            "soccer_italy_serie_a": "soccer",
            "soccer_france_ligue_one": "soccer",
        }
        sport = sport_lookup.get(league.lower())
        if sport is None:
            return {}

        tasks: dict[str, asyncio.Task[TeamInjurySummary]] = {}
        team_names = {
            team_name
            for game in games
            for team_name in (game.home_team, game.away_team)
            if team_name
        }

        for team_name in team_names:
            rating = power_ratings.lookup(team_name, ratings)
            if rating is None or not rating.team_id:
                continue
            tasks[team_name] = asyncio.create_task(
                stats_feed.get_team_injury_summary(
                    sport,
                    league,
                    rating.team_id,
                    key_minutes_threshold=key_minutes_threshold,
                )
            )

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        summaries: dict[str, TeamInjurySummary] = {}
        for team_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(
                    "odds.injury_enrichment_failed",
                    team=team_name,
                    error=str(result),
                )
                continue
            summaries[team_name] = result
        return summaries
