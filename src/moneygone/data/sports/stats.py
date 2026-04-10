"""Player statistics feed from ESPN's public API.

Provides season stats, game logs, rosters, and injury reports across
NBA, NFL, NHL, and MLB.  All data is fetched asynchronously via httpx
from ESPN's free JSON endpoints -- no API key required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_CORE_API_BASE = "https://sports.core.api.espn.com/v2/sports"
_WEB_API_BASE = "https://site.web.api.espn.com/apis/common/v3/sports"

# ESPN uses "{sport}/{league}" path segments.
_LEAGUE_PATHS: dict[str, str] = {
    "nba": "basketball/nba",
    "nfl": "football/nfl",
    "nhl": "hockey/nhl",
    "mlb": "baseball/mlb",
    "ncaab": "basketball/mens-college-basketball",
    "ncaaf": "football/college-football",
    "soccer_epl": "soccer/eng.1",
    "soccer_usa_mls": "soccer/usa.1",
    "soccer_spain_la_liga": "soccer/esp.1",
    "soccer_germany_bundesliga": "soccer/ger.1",
    "soccer_italy_serie_a": "soccer/ita.1",
    "soccer_france_ligue_one": "soccer/fra.1",
}

# Sport/league split for ESPN scoreboard checks (used by collector).
_SPORT_MAP: dict[str, dict[str, str]] = {
    k: {"sport": v.split("/")[0], "league": v.split("/")[1]}
    for k, v in _LEAGUE_PATHS.items()
}

_INJURY_STATUS_WEIGHTS: dict[str, float] = {
    "out": 1.0,
    "inactive": 1.0,
    "doubtful": 0.75,
    "questionable": 0.5,
    "day-to-day": 0.4,
    "day to day": 0.4,
    "probable": 0.15,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerSeasonStats:
    """Aggregated season statistics for a single player."""

    player_id: str
    name: str
    team: str
    position: str
    games_played: int
    stats: dict[str, float] = field(default_factory=dict)
    """Stat name -> per-game value, e.g. {"points_per_game": 27.3, ...}."""


@dataclass(frozen=True, slots=True)
class GameLogEntry:
    """A single game's stat line for one player."""

    game_id: str
    date: str
    opponent: str
    minutes: float
    stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlayerInfo:
    """Basic biographical / roster info for a player."""

    player_id: str
    name: str
    team_id: str
    team_name: str
    position: str
    jersey: str
    status: str  # "Active", "Injured", "Day-To-Day", etc.


@dataclass(frozen=True, slots=True)
class InjuryReport:
    """Injury entry for a single player on a game."""

    player_id: str
    name: str
    status: str  # "Out", "Questionable", "Probable", "Doubtful"
    description: str


@dataclass(frozen=True, slots=True)
class TeamInjurySummary:
    """Severity-weighted injury summary for a team."""

    team_id: str
    team_name: str
    key_injuries: int
    injury_severity: float
    impacted_players: tuple[dict[str, Any], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------


class PlayerStatsFeed:
    """Fetches player statistics from ESPN's public API.

    Parameters
    ----------
    client:
        Optional pre-configured ``httpx.AsyncClient``.  A new one is
        created internally when ``None``.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout
        self._player_minutes_cache: dict[tuple[str, str, str], float | None] = {}

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

    async def _get_json(self, url: str) -> dict[str, Any] | None:
        """GET *url* and return parsed JSON, or ``None`` on failure."""
        client = await self._get_client()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "stats.http_error",
                url=url,
                status=exc.response.status_code,
            )
            return None
        except httpx.TransportError as exc:
            logger.warning("stats.transport_error", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # League path helper
    # ------------------------------------------------------------------

    @staticmethod
    def _league_path(sport: str, league: str) -> str:
        """Return the ESPN URL segment for the given sport/league.

        Accepts either the canonical league key (``"nba"``) or explicit
        ``sport``/``league`` pair (``"basketball"``, ``"nba"``).
        """
        key = league.lower()
        if key in _LEAGUE_PATHS:
            return _LEAGUE_PATHS[key]
        return f"{sport.lower()}/{league.lower()}"

    @staticmethod
    def _web_athlete_url(sport: str, league: str, player_id: str) -> str:
        return f"{_WEB_API_BASE}/{PlayerStatsFeed._league_path(sport, league)}/athletes/{player_id}"

    @staticmethod
    def _web_gamelog_url(sport: str, league: str, player_id: str) -> str:
        return (
            f"{_WEB_API_BASE}/{PlayerStatsFeed._league_path(sport, league)}"
            f"/athletes/{player_id}/gamelog"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_player_season_stats(
        self,
        sport: str,
        league: str,
        player_id: str,
    ) -> PlayerSeasonStats | None:
        """Fetch a player's season-level stats.

        Tries the ``/athletes/{id}/statistics`` endpoint first, then
        falls back to ``/athletes/{id}`` which sometimes includes an
        inline statistics block.
        """
        web_data = await self._get_json(
            self._web_athlete_url(sport, league, player_id)
        )
        if web_data is not None:
            parsed = self._parse_web_athlete_summary(web_data, player_id)
            if parsed is not None:
                return parsed

        base = f"{_ESPN_BASE}/{self._league_path(sport, league)}"

        # Attempt 1: dedicated statistics endpoint.
        data = await self._get_json(f"{base}/athletes/{player_id}/statistics")
        if data is not None:
            return self._parse_season_stats(data, player_id)

        # Attempt 2: main athlete endpoint (may embed stats).
        data = await self._get_json(f"{base}/athletes/{player_id}")
        if data is not None:
            return self._parse_athlete_with_stats(data, player_id)

        return None

    async def get_player_game_log(
        self,
        sport: str,
        league: str,
        player_id: str,
        last_n: int = 10,
    ) -> list[GameLogEntry]:
        """Fetch a player's recent game-by-game stat lines.

        Tries the core API ``statisticslog`` endpoint first for richer
        structured data, then falls back to the site API ``gamelog``.
        """
        web_data = await self._get_json(
            self._web_gamelog_url(sport, league, player_id)
        )
        if web_data is not None:
            entries = self._parse_game_log(web_data, last_n)
            if entries:
                return entries

        lp = self._league_path(sport, league)
        sport_name, league_name = lp.split("/", 1)

        # Attempt 1: core API statisticslog (better structured data).
        core_url = (
            f"{_CORE_API_BASE}/{sport_name}/leagues/{league_name}"
            f"/athletes/{player_id}/statisticslog"
        )
        data = await self._get_json(core_url)
        if data is not None:
            entries = self._parse_game_log(data, last_n)
            if entries:
                return entries

        # Attempt 2: site API gamelog (original fallback).
        site_url = f"{_ESPN_BASE}/{lp}/athletes/{player_id}/gamelog"
        data = await self._get_json(site_url)
        if data is None:
            return []
        return self._parse_game_log(data, last_n)

    async def get_team_roster(
        self,
        sport: str,
        league: str,
        team_id: str,
    ) -> list[PlayerInfo]:
        """Return the roster for a team."""
        base = f"{_ESPN_BASE}/{self._league_path(sport, league)}"
        data = await self._get_json(f"{base}/teams/{team_id}/roster")
        if data is None:
            return []
        return self._parse_roster(data, team_id)

    async def get_game_injuries(
        self,
        sport: str,
        league: str,
        game_id: str | None = None,
    ) -> list[InjuryReport]:
        """Return injuries from the scoreboard for today's games.

        If *game_id* is provided, filters to that game.  Otherwise
        returns all injuries on the current scoreboard.
        """
        base = f"{_ESPN_BASE}/{self._league_path(sport, league)}"
        data = await self._get_json(f"{base}/scoreboard")
        if data is None:
            return []
        return self._parse_scoreboard_injuries(data, game_id)

    async def search_player(
        self,
        sport: str,
        league: str,
        name: str,
    ) -> PlayerInfo | None:
        """Search for a player by name using ESPN's athlete search.

        Returns the first matching result or ``None``.
        """
        base = f"{_ESPN_BASE}/{self._league_path(sport, league)}"
        # ESPN search endpoint (undocumented but widely used).
        url = f"https://site.api.espn.com/apis/common/v3/search?query={name}&type=player&sport={sport}&league={league}&limit=5"
        data = await self._get_json(url)
        if data is None:
            return None

        # Try the results/items array.
        items = data.get("results", data.get("items", []))
        if isinstance(items, list) and len(items) > 0:
            # Each item may have an $ref or id field.
            for item in items:
                pid = str(item.get("id", item.get("uid", "")))
                if not pid:
                    continue
                # Fetch full athlete detail.
                athlete_data = await self._get_json(f"{base}/athletes/{pid}")
                if athlete_data is not None:
                    return self._parse_player_info(athlete_data, pid)

        return None

    async def get_scoreboard(
        self,
        sport: str,
        league: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """Return raw scoreboard JSON, optionally for a specific date.

        *date* should be in ``YYYYMMDD`` format.
        """
        base = f"{_ESPN_BASE}/{self._league_path(sport, league)}"
        url = f"{base}/scoreboard"
        if date:
            url += f"?dates={date}"
        return await self._get_json(url)

    async def get_player_splits(
        self,
        sport: str,
        league: str,
        player_id: str,
    ) -> dict[str, Any] | None:
        """Fetch statistical splits (home/away/vs opponent) for a player.

        Uses the site API ``/athletes/{id}/splits`` endpoint.
        """
        lp = self._league_path(sport, league)
        url = f"{_ESPN_BASE}/{lp}/athletes/{player_id}/splits"
        return await self._get_json(url)

    async def get_team_injuries(
        self,
        sport: str,
        league: str,
        team_id: str,
    ) -> list[InjuryReport]:
        """Fetch the full injury report for a team.

        Uses the dedicated ``/teams/{id}/injuries`` endpoint which is
        more complete than extracting injuries from the scoreboard.
        """
        lp = self._league_path(sport, league)
        url = f"{_ESPN_BASE}/{lp}/teams/{team_id}/injuries"
        data = await self._get_json(url)
        if data is None:
            return []

        injuries: list[InjuryReport] = []
        try:
            for item in data.get("items", data.get("injuries", [])):
                athlete = item.get("athlete", {})
                pid = str(athlete.get("id", item.get("playerId", "")))
                pname = athlete.get("displayName", "")
                status = item.get("status", item.get("type", {}).get("description", "Unknown"))
                desc = item.get("details", item.get("longComment", item.get("shortComment", "")))
                if pid:
                    injuries.append(
                        InjuryReport(
                            player_id=pid,
                            name=pname,
                            status=status,
                            description=desc,
                        )
                    )
        except Exception:
            logger.warning("stats.parse_team_injuries_error", team_id=team_id, exc_info=True)

        if injuries:
            return injuries

        roster_url = f"{_ESPN_BASE}/{lp}/teams/{team_id}/roster"
        roster_data = await self._get_json(roster_url)
        if roster_data is None:
            return []

        try:
            for athlete in roster_data.get("athletes", []):
                for item in athlete.get("injuries", []):
                    pid = str(athlete.get("id", ""))
                    if not pid:
                        continue
                    injuries.append(
                        InjuryReport(
                            player_id=pid,
                            name=athlete.get("displayName", athlete.get("fullName", "")),
                            status=item.get("status", "Unknown"),
                            description=item.get("date", ""),
                        )
                    )
        except Exception:
            logger.warning("stats.parse_team_roster_injuries_error", team_id=team_id, exc_info=True)

        return injuries

    async def get_player_avg_minutes(
        self,
        sport: str,
        league: str,
        player_id: str,
    ) -> float | None:
        """Return a player's season-average minutes per game."""
        cache_key = (sport.lower(), league.lower(), player_id)
        if cache_key in self._player_minutes_cache:
            return self._player_minutes_cache[cache_key]

        data = await self._get_json(self._web_gamelog_url(sport, league, player_id))
        minutes = self._parse_avg_minutes_from_gamelog(data) if data is not None else None
        self._player_minutes_cache[cache_key] = minutes
        return minutes

    async def get_team_injury_summary(
        self,
        sport: str,
        league: str,
        team_id: str,
        *,
        key_minutes_threshold: float = 20.0,
    ) -> TeamInjurySummary:
        """Return severity-weighted injury impact for a team.

        Severity is a normalized 0-1 proxy built from injury status and
        average minutes per game.  This is intentionally lightweight:
        we care more about distinguishing a rotation player from a bench
        player than about perfect medical modeling.
        """
        roster = await self.get_team_roster(sport, league, team_id)
        if not roster:
            return TeamInjurySummary(
                team_id=team_id,
                team_name="",
                key_injuries=0,
                injury_severity=0.0,
            )

        impacted = [
            player for player in roster
            if self._injury_status_weight(player.status) > 0.0
        ]
        if not impacted:
            team_name = roster[0].team_name if roster else ""
            return TeamInjurySummary(
                team_id=team_id,
                team_name=team_name,
                key_injuries=0,
                injury_severity=0.0,
            )

        minute_tasks = [
            self.get_player_avg_minutes(sport, league, player.player_id)
            for player in impacted
        ]
        minute_results = await asyncio.gather(*minute_tasks)

        team_name = impacted[0].team_name if impacted else ""
        key_injuries = 0
        weighted_role_sum = 0.0
        impacted_players: list[dict[str, Any]] = []

        for player, avg_minutes in zip(impacted, minute_results):
            status_weight = self._injury_status_weight(player.status)
            role_weight = min((avg_minutes or 0.0) / 36.0, 1.0) if avg_minutes is not None else 0.15
            player_severity = status_weight * role_weight

            if avg_minutes is not None and avg_minutes >= key_minutes_threshold and status_weight >= 0.4:
                key_injuries += 1

            weighted_role_sum += player_severity
            impacted_players.append(
                {
                    "player_id": player.player_id,
                    "name": player.name,
                    "status": player.status,
                    "avg_minutes": avg_minutes,
                    "severity": round(player_severity, 4),
                }
            )

        # Roughly, three starter-level outs should push the team toward 1.0.
        team_severity = min(weighted_role_sum / 3.0, 1.0)

        return TeamInjurySummary(
            team_id=team_id,
            team_name=team_name,
            key_injuries=key_injuries,
            injury_severity=round(team_severity, 4),
            impacted_players=tuple(impacted_players),
        )

    async def get_league_stats(
        self,
        sport: str,
        league: str,
        category: str | None = None,
        sort: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch league-wide stats for normalization / comparison.

        Uses the web API ``byathlete`` endpoint which returns stat
        leaders across the league.

        Parameters
        ----------
        category:
            Stat category filter (e.g. ``"scoring"``, ``"passing"``).
        sort:
            Stat name to sort by (e.g. ``"points"``, ``"touchdowns"``).
        """
        lp = self._league_path(sport, league)
        url = f"{_WEB_API_BASE}/{lp}/statistics/byathlete"
        params: dict[str, str] = {}
        if category:
            params["category"] = category
        if sort:
            params["sort"] = sort

        client = await self._get_client()
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("stats.league_stats_http_error", url=url, status=exc.response.status_code)
            return None
        except httpx.TransportError as exc:
            logger.warning("stats.league_stats_transport_error", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_season_stats(
        data: dict[str, Any],
        player_id: str,
    ) -> PlayerSeasonStats | None:
        """Parse the ``/athletes/{id}/statistics`` response."""
        try:
            # The structure varies but typically has "splits" -> "categories".
            splits = data.get("splits", data.get("statistics", {}))
            if isinstance(splits, list) and splits:
                splits = splits[0]

            categories = splits.get("categories", [])
            stats: dict[str, float] = {}
            for cat in categories:
                cat_name = cat.get("name", "")
                for stat_entry in cat.get("stats", []):
                    stat_name = stat_entry.get("name", stat_entry.get("abbreviation", ""))
                    val = stat_entry.get("value") or stat_entry.get("displayValue")
                    if stat_name and val is not None:
                        try:
                            stats[stat_name] = float(val)
                        except (ValueError, TypeError):
                            pass

            athlete = data.get("athlete", {})
            name = athlete.get("displayName", f"Player {player_id}")
            team = athlete.get("team", {}).get("displayName", "")
            position = athlete.get("position", {}).get("abbreviation", "")
            gp = int(stats.pop("gamesPlayed", stats.pop("GP", 0)))

            return PlayerSeasonStats(
                player_id=player_id,
                name=name,
                team=team,
                position=position,
                games_played=gp,
                stats=stats,
            )
        except Exception:
            logger.warning("stats.parse_season_error", player_id=player_id, exc_info=True)
            return None

    @staticmethod
    def _parse_web_athlete_summary(
        data: dict[str, Any],
        player_id: str,
    ) -> PlayerSeasonStats | None:
        """Parse the site.web athlete summary endpoint."""
        try:
            athlete = data.get("athlete", data)
            summary = athlete.get("statsSummary", {})
            raw_stats = summary.get("statistics", [])
            stats: dict[str, float] = {}

            for entry in raw_stats:
                name = entry.get("name", entry.get("abbreviation", ""))
                value = entry.get("value", entry.get("displayValue"))
                if not name or value is None:
                    continue
                try:
                    stats[name] = float(value)
                except (ValueError, TypeError):
                    continue

            return PlayerSeasonStats(
                player_id=player_id,
                name=athlete.get("displayName", athlete.get("fullName", f"Player {player_id}")),
                team=athlete.get("team", {}).get("displayName", ""),
                position=athlete.get("position", {}).get("abbreviation", ""),
                games_played=0,
                stats=stats,
            )
        except Exception:
            logger.warning("stats.parse_web_athlete_error", player_id=player_id, exc_info=True)
            return None

    @staticmethod
    def _parse_athlete_with_stats(
        data: dict[str, Any],
        player_id: str,
    ) -> PlayerSeasonStats | None:
        """Parse the ``/athletes/{id}`` response that may embed stats."""
        try:
            name = data.get("displayName", f"Player {player_id}")
            team_info = data.get("team", {})
            team = team_info.get("displayName", "")
            position = data.get("position", {}).get("abbreviation", "")

            # Stats sometimes nested under "statistics" key.
            stats_block = data.get("statistics", [])
            stats: dict[str, float] = {}
            gp = 0

            if isinstance(stats_block, list):
                for stat_group in stats_block:
                    for cat in stat_group.get("categories", stat_group.get("splits", {}).get("categories", [])):
                        for entry in cat.get("stats", []):
                            sname = entry.get("name", entry.get("abbreviation", ""))
                            val = entry.get("value") or entry.get("displayValue")
                            if sname and val is not None:
                                try:
                                    stats[sname] = float(val)
                                except (ValueError, TypeError):
                                    pass

            gp = int(stats.pop("gamesPlayed", stats.pop("GP", 0)))

            return PlayerSeasonStats(
                player_id=player_id,
                name=name,
                team=team,
                position=position,
                games_played=gp,
                stats=stats,
            )
        except Exception:
            logger.warning("stats.parse_athlete_error", player_id=player_id, exc_info=True)
            return None

    @staticmethod
    def _parse_game_log(data: dict[str, Any], last_n: int) -> list[GameLogEntry]:
        """Parse the ``/athletes/{id}/gamelog`` response."""
        entries: list[GameLogEntry] = []
        try:
            if isinstance(data.get("seasonTypes"), list):
                names = data.get("names", [])
                season_types = data.get("seasonTypes", [])
                events_lookup = data.get("events", {})
                collected: list[GameLogEntry] = []

                for season_type in season_types:
                    for category in season_type.get("categories", []):
                        for event in category.get("events", []):
                            stats_list = event.get("stats", [])
                            if not stats_list:
                                continue

                            game_id = str(event.get("eventId", ""))
                            metadata = events_lookup.get(game_id, {})
                            opponent = metadata.get("opponent", {}).get("displayName", "")
                            date = metadata.get("gameDate", "")

                            game_stats: dict[str, float] = {}
                            for idx, raw_val in enumerate(stats_list):
                                if idx >= len(names):
                                    continue
                                parsed_val = PlayerStatsFeed._parse_numeric_stat(raw_val)
                                if parsed_val is not None:
                                    game_stats[names[idx]] = parsed_val

                            minutes = game_stats.pop("minutes", 0.0)
                            collected.append(
                                GameLogEntry(
                                    game_id=game_id,
                                    date=date,
                                    opponent=opponent,
                                    minutes=minutes,
                                    stats=game_stats,
                                )
                            )

                if collected:
                    return collected[-last_n:]

            # gamelog typically has "events", "labels", and "stats" arrays.
            events = data.get("events", [])
            labels = data.get("labels", [])
            seasontype = data.get("seasonType", {})
            categories = data.get("categories", [])

            # Newer ESPN format: "events" is a list of dicts.
            if events and isinstance(events[0], dict):
                for evt in events[-last_n:]:
                    game_id = str(evt.get("id", evt.get("eventId", "")))
                    date = evt.get("gameDate", evt.get("date", ""))
                    opponent = evt.get("opponent", {}).get("displayName", "")
                    game_stats: dict[str, float] = {}
                    minutes = 0.0
                    for cat in evt.get("categories", evt.get("stats", [])):
                        if isinstance(cat, dict):
                            for s in cat.get("stats", []):
                                sname = s.get("name", s.get("abbreviation", ""))
                                val = s.get("value") or s.get("displayValue")
                                if sname and val is not None:
                                    try:
                                        game_stats[sname] = float(val)
                                    except (ValueError, TypeError):
                                        pass
                    minutes = game_stats.pop("minutes", game_stats.pop("MIN", 0.0))
                    entries.append(
                        GameLogEntry(
                            game_id=game_id,
                            date=date,
                            opponent=opponent,
                            minutes=minutes,
                            stats=game_stats,
                        )
                    )

            # Older format: parallel arrays.
            elif "stats" in data and isinstance(data["stats"], list):
                raw_stats = data["stats"]
                for i, row in enumerate(raw_stats[-last_n:]):
                    game_id = str(events[i]) if i < len(events) else ""
                    game_stats_dict: dict[str, float] = {}
                    if isinstance(row, list):
                        for j, val in enumerate(row):
                            label = labels[j] if j < len(labels) else f"stat_{j}"
                            try:
                                game_stats_dict[label] = float(val)
                            except (ValueError, TypeError):
                                pass
                    minutes_val = game_stats_dict.pop("MIN", game_stats_dict.pop("minutes", 0.0))
                    entries.append(
                        GameLogEntry(
                            game_id=game_id,
                            date="",
                            opponent="",
                            minutes=minutes_val,
                            stats=game_stats_dict,
                        )
                    )
        except Exception:
            logger.warning("stats.parse_gamelog_error", exc_info=True)

        return entries[-last_n:]

    @staticmethod
    def _parse_roster(data: dict[str, Any], team_id: str) -> list[PlayerInfo]:
        """Parse the ``/teams/{id}/roster`` response."""
        players: list[PlayerInfo] = []
        try:
            # May be under "athletes" grouped by position, or flat list.
            athlete_groups = data.get("athletes", [])
            team_name = data.get("team", {}).get("displayName", "")

            for group in athlete_groups:
                # group may be a dict with "items" or itself a list.
                if isinstance(group, dict) and "items" in group and isinstance(group["items"], list):
                    items = group["items"]
                else:
                    items = [group]
                for athlete in items:
                    pid = str(athlete.get("id", ""))
                    name = athlete.get("displayName", athlete.get("fullName", ""))
                    pos = athlete.get("position", {})
                    pos_abbr = pos.get("abbreviation", "") if isinstance(pos, dict) else str(pos)
                    jersey = str(athlete.get("jersey", ""))
                    status_val = athlete.get("status", {})
                    status_name = status_val.get("type", status_val.get("name", "Active")) if isinstance(status_val, dict) else str(status_val)
                    injuries = athlete.get("injuries", [])
                    if injuries and isinstance(injuries, list):
                        status_name = str(injuries[0].get("status", status_name))

                    if pid and name:
                        players.append(
                            PlayerInfo(
                                player_id=pid,
                                name=name,
                                team_id=team_id,
                                team_name=team_name,
                                position=pos_abbr,
                                jersey=jersey,
                                status=status_name,
                            )
                        )
        except Exception:
            logger.warning("stats.parse_roster_error", team_id=team_id, exc_info=True)

        return players

    @staticmethod
    def _parse_scoreboard_injuries(
        data: dict[str, Any],
        game_id: str | None,
    ) -> list[InjuryReport]:
        """Extract injuries from scoreboard JSON."""
        injuries: list[InjuryReport] = []
        try:
            for event in data.get("events", []):
                eid = str(event.get("id", ""))
                if game_id is not None and eid != game_id:
                    continue
                for comp in event.get("competitions", []):
                    for competitor in comp.get("competitors", []):
                        for inj in competitor.get("injuries", []):
                            pid = str(inj.get("playerId", inj.get("athlete", {}).get("id", "")))
                            pname = inj.get("athlete", {}).get("displayName", "")
                            status = inj.get("status", inj.get("type", {}).get("description", "Unknown"))
                            desc = inj.get("details", inj.get("longComment", inj.get("shortComment", "")))
                            if pid:
                                injuries.append(
                                    InjuryReport(
                                        player_id=pid,
                                        name=pname,
                                        status=status,
                                        description=desc,
                                    )
                                )
        except Exception:
            logger.warning("stats.parse_injuries_error", exc_info=True)

        return injuries

    @staticmethod
    def _parse_player_info(data: dict[str, Any], player_id: str) -> PlayerInfo:
        """Build a PlayerInfo from an athlete detail response."""
        name = data.get("displayName", data.get("fullName", f"Player {player_id}"))
        team_info = data.get("team", {})
        team_id = str(team_info.get("id", ""))
        team_name = team_info.get("displayName", "")
        pos = data.get("position", {})
        pos_abbr = pos.get("abbreviation", "") if isinstance(pos, dict) else str(pos)
        jersey = str(data.get("jersey", ""))
        status_val = data.get("status", {})
        status_name = status_val.get("type", "Active") if isinstance(status_val, dict) else str(status_val)

        return PlayerInfo(
            player_id=player_id,
            name=name,
            team_id=team_id,
            team_name=team_name,
            position=pos_abbr,
            jersey=jersey,
            status=status_name,
        )

    @staticmethod
    def _parse_numeric_stat(value: Any) -> float | None:
        """Parse ESPN stat strings into floats.

        Handles plain numerics (``"22.1"``) and minute-like values
        (``"18:30"`` -> ``18.5``).
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                try:
                    mins = float(parts[0])
                    secs = float(parts[1])
                except ValueError:
                    return None
                return mins + secs / 60.0
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_avg_minutes_from_gamelog(data: dict[str, Any]) -> float | None:
        """Extract average minutes from the site.web gamelog payload."""
        if not data:
            return None

        names = data.get("names", [])
        labels = data.get("labels", [])
        minute_idx = None

        for idx, name in enumerate(names):
            lowered = str(name).lower()
            if lowered in {"minutes", "timeonice"} or "minutes" in lowered:
                minute_idx = idx
                break

        if minute_idx is None:
            for idx, label in enumerate(labels):
                if str(label).upper() in {"MIN", "TOI"}:
                    minute_idx = idx
                    break

        if minute_idx is None:
            return None

        for season_type in data.get("seasonTypes", []):
            summary = season_type.get("summary", {})
            for stat_block in summary.get("stats", []):
                if stat_block.get("type") != "avg":
                    continue
                stats = stat_block.get("stats", [])
                if minute_idx >= len(stats):
                    continue
                return PlayerStatsFeed._parse_numeric_stat(stats[minute_idx])

        return None

    @staticmethod
    def _injury_status_weight(status: str) -> float:
        lowered = status.strip().lower()
        for key, weight in _INJURY_STATUS_WEIGHTS.items():
            if key in lowered:
                return weight
        return 0.0
