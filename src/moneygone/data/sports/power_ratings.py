"""Team power ratings from ESPN standings.

Fetches win percentage and point differential for all teams in a league.
Used as a model-based win probability signal in game winner features.

No API key required — ESPN public CDN endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# CDN standings endpoints by league
_STANDINGS_URLS: dict[str, str] = {
    "nba":   "https://cdn.espn.com/core/nba/standings?xhr=1",
    "nfl":   "https://cdn.espn.com/core/nfl/standings?xhr=1",
    "nhl":   "https://cdn.espn.com/core/nhl/standings?xhr=1",
    "mlb":   "https://cdn.espn.com/core/mlb/standings?xhr=1",
    "ncaab": "https://cdn.espn.com/core/mens-college-basketball/standings?xhr=1",
    "ncaaf": "https://cdn.espn.com/core/college-football/standings?xhr=1",
}

# ESPN teams list endpoints (for ID lookups)
_TEAMS_URLS: dict[str, str] = {
    "nba":   "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50",
    "nfl":   "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams?limit=50",
    "nhl":   "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams?limit=50",
    "mlb":   "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams?limit=50",
}


@dataclass(frozen=True, slots=True)
class TeamRating:
    """Power rating snapshot for one team."""

    team_id: str
    display_name: str
    abbreviation: str
    win_pct: float           # 0.0 – 1.0
    wins: int
    losses: int
    avg_pts_for: float       # avg points/goals scored per game
    avg_pts_against: float   # avg points/goals allowed per game
    point_differential: float  # avg margin (positive = better team)

    @property
    def rating(self) -> float:
        """Single numeric rating: point differential is most predictive."""
        return self.point_differential


class ESPNPowerRatings:
    """Fetches and caches ESPN standings for team power ratings.

    Usage::

        ratings = ESPNPowerRatings()
        nba = await ratings.get_ratings("nba")
        lakers = ratings.lookup("Los Angeles Lakers", nba)
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None
        self._cache: dict[str, dict[str, TeamRating]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                headers={"User-Agent": "MoneyGone/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_ratings(self, league: str) -> dict[str, TeamRating]:
        """Return ``{display_name: TeamRating}`` for all teams in a league.

        Results are cached in-process. Call :meth:`invalidate` to refresh.
        """
        league = league.lower()
        if league in self._cache:
            return self._cache[league]

        ratings = await self._fetch_ratings(league)
        self._cache[league] = ratings
        return ratings

    def invalidate(self, league: str | None = None) -> None:
        """Invalidate cached ratings (call before each trading session)."""
        if league:
            self._cache.pop(league.lower(), None)
        else:
            self._cache.clear()

    async def _fetch_ratings(self, league: str) -> dict[str, TeamRating]:
        url = _STANDINGS_URLS.get(league)
        if url is None:
            logger.warning("power_ratings.unsupported_league", league=league)
            return {}

        client = await self._get_client()
        try:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        except Exception:
            logger.warning("power_ratings.fetch_failed", league=league, exc_info=True)
            return {}

        return self._parse_standings(data)

    @staticmethod
    def _parse_standings(data: dict[str, Any]) -> dict[str, TeamRating]:
        """Parse ESPN CDN standings JSON into TeamRating objects."""
        ratings: dict[str, TeamRating] = {}
        try:
            groups = data.get("content", {}).get("standings", {}).get("groups", [])
            for group in groups:
                entries = group.get("standings", {}).get("entries", [])
                for entry in entries:
                    team = entry.get("team", {})
                    display_name = team.get("displayName", "")
                    if not display_name:
                        continue

                    stats: dict[str, Any] = {}
                    for s in entry.get("stats", []):
                        name = s.get("name", "")
                        val = s.get("value")
                        if val is None:
                            try:
                                val = float(s.get("displayValue", "0").replace("%", ""))
                            except (ValueError, TypeError):
                                val = 0.0
                        stats[name] = val

                    win_pct = float(stats.get("winPercent", 0) or 0)
                    wins = int(float(stats.get("wins", 0) or 0))
                    losses = int(float(stats.get("losses", 0) or 0))
                    pts_for = float(stats.get("avgPointsFor", 0) or 0)
                    pts_against = float(stats.get("avgPointsAgainst", 0) or 0)
                    differential = float(stats.get("differential", pts_for - pts_against) or 0)

                    ratings[display_name] = TeamRating(
                        team_id=team.get("id", ""),
                        display_name=display_name,
                        abbreviation=team.get("abbreviation", ""),
                        win_pct=win_pct,
                        wins=wins,
                        losses=losses,
                        avg_pts_for=pts_for,
                        avg_pts_against=pts_against,
                        point_differential=differential,
                    )
        except Exception:
            logger.warning("power_ratings.parse_failed", exc_info=True)

        return ratings

    @staticmethod
    def lookup(team_name: str, ratings: dict[str, TeamRating]) -> TeamRating | None:
        """Fuzzy-match a team name to its rating.

        Handles mismatches between Odds API names (e.g. "LA Lakers") and
        ESPN display names (e.g. "Los Angeles Lakers").
        """
        if not team_name or not ratings:
            return None

        normalized_target = ESPNPowerRatings._normalize_team_name(team_name)

        # Exact match first
        if team_name in ratings:
            return ratings[team_name]
        for display_name, rating in ratings.items():
            if ESPNPowerRatings._normalize_team_name(display_name) == normalized_target:
                return rating

        name_lower = team_name.lower()

        # Substring match — ESPN name contains Odds API name or vice versa
        for display, rating in ratings.items():
            dl = display.lower()
            if name_lower in dl or dl in name_lower:
                return rating

        # Word overlap — match on the city or mascot word
        name_words = set(name_lower.split())
        best_overlap = 0
        best_rating: TeamRating | None = None
        for display, rating in ratings.items():
            dl_words = set(display.lower().split())
            overlap = len(name_words & dl_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rating = rating

        return best_rating if best_overlap >= 1 else None

    @staticmethod
    def _normalize_team_name(team_name: str) -> str:
        """Normalize common display-name variants across feeds."""
        cleaned = team_name.lower().replace(".", "")
        aliases = {
            "la ": "los angeles ",
            "ny ": "new york ",
            "st ": "saint ",
        }
        for old, new in aliases.items():
            if cleaned.startswith(old):
                cleaned = new + cleaned[len(old):]
        return " ".join(cleaned.split())

    async def get_team_id(self, league: str, team_name: str) -> str | None:
        """Return ESPN team ID for a given team name."""
        ratings = await self.get_ratings(league)
        match = self.lookup(team_name, ratings)
        return match.team_id if match else None
