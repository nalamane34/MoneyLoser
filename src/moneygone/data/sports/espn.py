"""ESPN live score feed for real-time game outcome detection.

Uses ESPN's public scoreboard API to poll live scores across major sports
leagues.  When a game reaches final status, an :class:`OutcomeSignal` is
emitted for the resolution sniping strategy to act on.

Supported sport/league combinations:
    - football / nfl
    - basketball / nba
    - baseball / mlb
    - hockey / nhl
    - soccer / usa.1 (MLS)
    - mma / ufc
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

import httpx
import structlog

from moneygone.utils.time import now_utc

logger = structlog.get_logger(__name__)

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Canonical mapping: (sport, league) pairs that ESPN supports.
SUPPORTED_LEAGUES: dict[str, str] = {
    "nfl": "football/nfl",
    "nba": "basketball/nba",
    "mlb": "baseball/mlb",
    "nhl": "hockey/nhl",
    "mls": "soccer/usa.1",
    "ufc": "mma/ufc",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameState:
    """Snapshot of a single game's current state."""

    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str  # "pre", "in", "post"
    period: str  # quarter, inning, half, round, etc.
    clock: str  # time remaining, e.g. "2:30" or ""
    is_final: bool
    winner: str | None  # team name if final, else None
    detail: str  # e.g. "Final", "4th Quarter 2:30", "Halftime"
    sport: str = ""
    league: str = ""


@dataclass(frozen=True, slots=True)
class OutcomeSignal:
    """Signal that a real-world outcome has been determined."""

    game_id: str
    outcome: str  # "home_win", "away_win", "draw"
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    confidence: float  # 0-1, typically 1.0 for final games
    source: str  # "espn"
    sport: str
    league: str
    detected_at: datetime


# ---------------------------------------------------------------------------
# ESPN Live Feed
# ---------------------------------------------------------------------------


class ESPNLiveFeed:
    """Async client for ESPN's public scoreboard API.

    Polls live scores and detects when game outcomes become known.
    Designed for low-latency outcome detection in the resolution sniping
    pipeline.

    Parameters
    ----------
    client:
        Optional pre-configured ``httpx.AsyncClient``.  If ``None``,
        a new client is created internally.
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
        # Track previously seen game states to detect transitions.
        self._previous_states: dict[str, GameState] = {}
        self._polling_tasks: list[asyncio.Task[None]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"User-Agent": "MoneyGone/1.0"},
            )
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Cancel polling tasks and close the HTTP client."""
        for task in self._polling_tasks:
            task.cancel()
        self._polling_tasks.clear()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Score fetching
    # ------------------------------------------------------------------

    async def get_live_scores(self, sport: str, league: str) -> list[GameState]:
        """Fetch current scoreboard for a sport/league.

        Parameters
        ----------
        sport:
            ESPN sport slug (e.g. ``"basketball"``).
        league:
            ESPN league slug (e.g. ``"nba"``).

        Returns
        -------
        list[GameState]
            All games on today's scoreboard.
        """
        url = f"{_ESPN_BASE}/{sport}/{league}/scoreboard"
        client = await self._get_client()

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "espn.http_error",
                sport=sport,
                league=league,
                status=exc.response.status_code,
            )
            return []
        except httpx.TransportError as exc:
            logger.error(
                "espn.transport_error",
                sport=sport,
                league=league,
                error=str(exc),
            )
            return []

        events = payload.get("events", [])
        games: list[GameState] = []

        for event in events:
            game = self._parse_event(event, sport, league)
            if game is not None:
                games.append(game)

        logger.debug(
            "espn.scores_fetched",
            sport=sport,
            league=league,
            games=len(games),
        )
        return games

    # ------------------------------------------------------------------
    # Outcome detection
    # ------------------------------------------------------------------

    def detect_outcome(self, game: GameState) -> OutcomeSignal | None:
        """Check whether a game has a determined outcome.

        Returns an :class:`OutcomeSignal` if the game is final.  For games
        still in progress, returns ``None``.

        Parameters
        ----------
        game:
            Current game state snapshot.

        Returns
        -------
        OutcomeSignal | None
        """
        if not game.is_final:
            return None

        if game.home_score > game.away_score:
            outcome = "home_win"
        elif game.away_score > game.home_score:
            outcome = "away_win"
        else:
            outcome = "draw"

        return OutcomeSignal(
            game_id=game.game_id,
            outcome=outcome,
            home_team=game.home_team,
            away_team=game.away_team,
            home_score=game.home_score,
            away_score=game.away_score,
            confidence=1.0,
            source="espn",
            sport=game.sport,
            league=game.league,
            detected_at=now_utc(),
        )

    # ------------------------------------------------------------------
    # Continuous polling
    # ------------------------------------------------------------------

    async def start_polling(
        self,
        sport: str,
        league: str,
        interval_seconds: float = 10.0,
        callback: Callable[[list[GameState], list[OutcomeSignal]], Coroutine[Any, Any, None]] | None = None,
    ) -> asyncio.Task[None]:
        """Start continuous polling for a sport/league.

        Polls the scoreboard at the given interval, detects new outcomes
        by comparing against previous state, and invokes the callback
        with changed games and any new outcome signals.

        Parameters
        ----------
        sport:
            ESPN sport slug.
        league:
            ESPN league slug.
        interval_seconds:
            Seconds between polls.
        callback:
            Async callable ``(changed_games, new_outcomes) -> None``.

        Returns
        -------
        asyncio.Task
            The background polling task.  Cancel to stop.
        """
        task = asyncio.create_task(
            self._poll_loop(sport, league, interval_seconds, callback),
            name=f"espn_poll_{sport}_{league}",
        )
        self._polling_tasks.append(task)
        return task

    async def _poll_loop(
        self,
        sport: str,
        league: str,
        interval: float,
        callback: Callable[[list[GameState], list[OutcomeSignal]], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Internal polling loop."""
        poll_key = f"{sport}/{league}"
        logger.info(
            "espn.polling_started",
            sport=sport,
            league=league,
            interval=interval,
        )

        while True:
            try:
                games = await self.get_live_scores(sport, league)
                changed: list[GameState] = []
                new_outcomes: list[OutcomeSignal] = []

                for game in games:
                    prev = self._previous_states.get(game.game_id)
                    state_changed = self._has_state_changed(prev, game)

                    if state_changed:
                        changed.append(game)

                        # Check for newly-final games.
                        was_final = prev.is_final if prev else False
                        if game.is_final and not was_final:
                            signal = self.detect_outcome(game)
                            if signal is not None:
                                new_outcomes.append(signal)
                                logger.info(
                                    "espn.outcome_detected",
                                    game_id=game.game_id,
                                    outcome=signal.outcome,
                                    home=game.home_team,
                                    away=game.away_team,
                                    score=f"{game.home_score}-{game.away_score}",
                                )

                    self._previous_states[game.game_id] = game

                if callback and (changed or new_outcomes):
                    await callback(changed, new_outcomes)

            except asyncio.CancelledError:
                logger.info("espn.polling_stopped", key=poll_key)
                raise
            except Exception:
                logger.exception("espn.poll_error", key=poll_key)

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_event(event: dict[str, Any], sport: str, league: str) -> GameState | None:
        """Parse a single ESPN event dict into a GameState."""
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            # ESPN puts home team first by homeAway field.
            home = away = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                elif c.get("homeAway") == "away":
                    away = c

            if home is None or away is None:
                # Fallback: first is home, second is away.
                home, away = competitors[0], competitors[1]

            home_team_info = home.get("team", {})
            away_team_info = away.get("team", {})

            home_name = (
                home_team_info.get("displayName")
                or home_team_info.get("shortDisplayName")
                or home_team_info.get("name", "Home")
            )
            away_name = (
                away_team_info.get("displayName")
                or away_team_info.get("shortDisplayName")
                or away_team_info.get("name", "Away")
            )

            home_score = int(home.get("score", "0"))
            away_score = int(away.get("score", "0"))

            # Status parsing.
            status_data = comp.get("status", event.get("status", {}))
            status_type = status_data.get("type", {})
            status_state = status_type.get("state", "pre")  # pre, in, post
            detail = status_type.get("detail", "")
            short_detail = status_type.get("shortDetail", detail)
            is_final = status_type.get("completed", False)

            # Period / clock.
            period_val = str(status_data.get("period", "0"))
            clock = status_data.get("displayClock", "")

            return GameState(
                game_id=str(event.get("id", "")),
                home_team=home_name,
                away_team=away_name,
                home_score=home_score,
                away_score=away_score,
                status=status_state,
                period=period_val,
                clock=clock,
                is_final=is_final,
                winner=home_name if is_final and home_score > away_score
                else away_name if is_final and away_score > home_score
                else None,
                detail=short_detail or detail,
                sport=sport,
                league=league,
            )
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "espn.parse_error",
                event_id=event.get("id"),
                exc_info=True,
            )
            return None

    @staticmethod
    def _has_state_changed(prev: GameState | None, curr: GameState) -> bool:
        """Return True if the game state has materially changed."""
        if prev is None:
            return True
        return (
            prev.home_score != curr.home_score
            or prev.away_score != curr.away_score
            or prev.status != curr.status
            or prev.is_final != curr.is_final
            or prev.period != curr.period
        )
