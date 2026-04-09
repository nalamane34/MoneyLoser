"""Sports prop features for player performance prediction markets.

All features read from ``context.sports_snapshot``, a dict populated by
the sports data pipeline before feature computation.  When data is
unavailable the feature returns ``None`` gracefully.

Expected ``sports_snapshot`` keys (populated upstream):

    player_season_stats : dict
        Season averages keyed by stat name (e.g. ``"points_per_game"``).
    player_game_log : list[dict]
        Recent game-by-game stat dicts, newest last.
    player_minutes_avg : float
        Average minutes per game this season.
    player_minutes_trend : list[float]
        Minutes played in last N games, newest last.
    team_pace : float
        Team pace (possessions per game for NBA, plays per game for NFL).
    opponent_def_rank : float
        Opponent's defensive ranking against the relevant stat.
        Higher value = worse defense = more stats allowed.
    league_avg_def : float
        League-average defensive stat allowed (for normalisation).
    spread : float
        Game spread (home perspective).  Positive = home is underdog.
    teammate_injuries : int
        Number of key teammates on the injury report.
    prop_stat_key : str
        The stat key this prop market corresponds to (e.g. ``"PTS"``,
        ``"REB"``, ``"3PM"``).
    prop_lines : list[dict]
        Sportsbook prop lines, each with ``line``, ``over_price``,
        ``under_price``, ``bookmaker``.
    kalshi_implied_prob : float
        Kalshi market implied probability (yes price / 100).
    opening_line : float | None
        Opening prop line from earliest available snapshot.
    current_line : float | None
        Current consensus prop line.
    usage_rate : float | None
        NBA usage rate (estimated possessions used / team possessions).
"""

from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Any

import structlog

from moneygone.features.base import Feature, FeatureContext

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sports(ctx: FeatureContext, key: str, default: Any = None) -> Any:
    """Safely extract a value from the sports snapshot."""
    snap = ctx.sports_snapshot
    if snap is None:
        return default
    return snap.get(key, default)


def _get_sports_float(ctx: FeatureContext, key: str) -> float | None:
    """Extract a float from the sports snapshot, returning None on failure."""
    val = _get_sports(ctx, key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _game_log_stat_values(ctx: FeatureContext, stat_key: str | None = None) -> list[float]:
    """Extract a list of stat values from the player's game log.

    Uses ``prop_stat_key`` from the snapshot when *stat_key* is not
    provided.
    """
    game_log = _get_sports(ctx, "player_game_log")
    if not game_log or not isinstance(game_log, list):
        return []

    if stat_key is None:
        stat_key = _get_sports(ctx, "prop_stat_key")
    if not stat_key:
        return []

    values: list[float] = []
    for game in game_log:
        if not isinstance(game, dict):
            continue
        stats = game.get("stats", game)
        val = stats.get(stat_key)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                pass
    return values


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class PlayerMean(Feature):
    """Player's season average for the stat in question.

    This is the single most important feature -- the baseline
    expectation against which all other signals are measured.
    """

    name = "player_mean"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        season_stats = _get_sports(context, "player_season_stats")
        stat_key = _get_sports(context, "prop_stat_key")
        if not season_stats or not stat_key:
            return None
        if isinstance(season_stats, dict):
            val = season_stats.get(stat_key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
        return None


class PlayerVariance(Feature):
    """Standard deviation of a player's recent game-by-game performance.

    High variance means the player is less predictable, and the model
    should be less confident in point estimates.  Uses the last N games
    from the game log.
    """

    name = "player_variance"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        values = _game_log_stat_values(context)
        if len(values) < 3:
            return None
        try:
            return statistics.stdev(values)
        except statistics.StatisticsError:
            return None


class PlayerRecentForm(Feature):
    """Last-5-game average vs season average.

    A ratio > 1.0 indicates the player is on a hot streak; < 1.0
    means they are underperforming their season baseline.
    """

    name = "player_recent_form"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, recent_n: int = 5) -> None:
        self._recent_n = recent_n

    def compute(self, context: FeatureContext) -> float | None:
        values = _game_log_stat_values(context)
        if len(values) < self._recent_n:
            return None

        recent_avg = sum(values[-self._recent_n :]) / self._recent_n

        # Get season mean for comparison.
        season_stats = _get_sports(context, "player_season_stats")
        stat_key = _get_sports(context, "prop_stat_key")
        if not season_stats or not stat_key:
            return None

        season_avg = season_stats.get(stat_key)
        if season_avg is None:
            return None
        try:
            season_avg = float(season_avg)
        except (ValueError, TypeError):
            return None

        if season_avg == 0:
            return None

        return recent_avg / season_avg


class UsageRate(Feature):
    """Estimated usage rate for NBA players.

    Usage rate approximates the percentage of team possessions a player
    uses while on the floor.  Higher usage = more opportunities for
    stats.  Returns ``None`` for non-NBA markets.
    """

    name = "usage_rate"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_sports_float(context, "usage_rate")


class GameScript(Feature):
    """Expected game competitiveness derived from the spread.

    Blowouts (large absolute spread) mean starters play fewer minutes
    and garbage-time dynamics change stat distributions.  Tight games
    (small spread) mean stars play full minutes.

    Returns the absolute spread value -- larger = less competitive.
    """

    name = "game_script"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        spread = _get_sports_float(context, "spread")
        if spread is None:
            return None
        # Return absolute value; the direction matters less than magnitude
        # for predicting whether starters play full minutes.
        return abs(spread)


class MatchupEffect(Feature):
    """Opponent defensive ranking against the relevant stat category.

    Normalised against the league average so that:
      > 0 means opponent allows more stats than average (good matchup)
      < 0 means opponent allows fewer stats (tough matchup)

    Computed as ``(opponent_allowed - league_avg) / league_avg``.
    """

    name = "matchup_effect"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        opp_rank = _get_sports_float(context, "opponent_def_rank")
        league_avg = _get_sports_float(context, "league_avg_def")
        if opp_rank is None or league_avg is None:
            return None
        if league_avg == 0:
            return None
        return (opp_rank - league_avg) / league_avg


class TeamPace(Feature):
    """Team pace of play.

    For NBA: possessions per game.  For NFL: plays per game.
    Higher pace = more statistical opportunities.
    """

    name = "team_pace"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_sports_float(context, "team_pace")


class InjuryImpact(Feature):
    """Count of key teammates on the injury report.

    More injuries to teammates often leads to higher usage for the
    remaining starters (especially in the NBA) -- they absorb the
    missing player's touches, shots, and minutes.
    """

    name = "injury_impact"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        count = _get_sports(context, "teammate_injuries")
        if count is None:
            return None
        try:
            return float(int(count))
        except (ValueError, TypeError):
            return None


class MinutesExpected(Feature):
    """Player's expected minutes per game.

    Returns the season average.  When a minutes trend is available,
    also checks if minutes are trending up or down -- but the raw
    average is the primary output used for modelling.
    """

    name = "minutes_expected"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_sports_float(context, "player_minutes_avg")


class PropLineVsMarket(Feature):
    """Sportsbook consensus line vs Kalshi implied probability.

    Computes the consensus over-probability from sportsbook lines
    (converting decimal odds to implied probability) and compares it
    to the Kalshi market price.  A positive value means sportsbooks
    imply a higher over probability than Kalshi -- potential buy signal.

    Output: ``sportsbook_implied_over - kalshi_implied_prob``.
    """

    name = "prop_line_vs_market"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        prop_lines = _get_sports(context, "prop_lines")
        kalshi_prob = _get_sports_float(context, "kalshi_implied_prob")
        if not prop_lines or kalshi_prob is None:
            return None

        # Average the implied over probability across bookmakers.
        over_probs: list[float] = []
        for line_data in prop_lines:
            if not isinstance(line_data, dict):
                continue
            over_price = line_data.get("over_price")
            if over_price is not None:
                try:
                    over_dec = float(over_price)
                    if over_dec > 0:
                        # Decimal odds -> implied probability.
                        over_probs.append(1.0 / over_dec)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

        if not over_probs:
            return None

        avg_implied_over = sum(over_probs) / len(over_probs)
        return avg_implied_over - kalshi_prob


class SharpMoneyIndicator(Feature):
    """Line movement direction as a sharp-money signal.

    If the opening line was 22.5 and the current line is 24.5, sharp
    money has pushed the over.  Returns the signed movement:
      > 0 means line moved up (sharp on the over)
      < 0 means line moved down (sharp on the under)

    Normalised by the opening line so the magnitude is comparable
    across different stat categories.
    """

    name = "sharp_money_indicator"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        opening = _get_sports_float(context, "opening_line")
        current = _get_sports_float(context, "current_line")
        if opening is None or current is None:
            return None
        if opening == 0:
            return None
        return (current - opening) / opening
