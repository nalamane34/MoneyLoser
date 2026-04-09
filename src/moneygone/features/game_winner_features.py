"""Game winner features for head-to-head prediction markets.

All features read from ``context.sports_snapshot``, a dict populated by
the sports data pipeline before feature computation.  When data is
unavailable the feature returns ``None`` gracefully.

Expected ``sports_snapshot`` keys for game-winner context:

    home_team : str
        Home team name.
    away_team : str
        Away team name.
    is_home_team : bool | int
        Whether the Kalshi market is on the HOME team winning (1 = home, 0 = away).
        Used to orient all features from the perspective of the team Kalshi is pricing.
    kalshi_implied_prob : float
        Kalshi market implied probability for the team (yes price / 100).

    # Sportsbook consensus
    sportsbook_home_win_prob : float | None
        Consensus sportsbook implied win probability for home team.
        Derived by averaging 1/decimal_odds across bookmakers, normalised
        to remove the overround.
    opening_moneyline_home : float | None
        Opening decimal odds for home team at line open (earliest snapshot).
    current_moneyline_home : float | None
        Current consensus decimal odds for home team (latest snapshot).

    # Public / sharp money
    public_pct_home : float | None
        Percentage of public bets placed on the home team (0.0 - 1.0).
        Source: Action Network, DraftKings public splits, or Odds API.
    public_pct_away : float | None
        Percentage of public bets placed on the away team.

    # Power ratings
    home_team_rating : float | None
        Home team Elo / net-rating / power index (higher = better team).
    away_team_rating : float | None
        Away team Elo / net-rating / power index.

    # Injuries
    home_key_injuries : int | None
        Number of key players (rotation / starter) OUT for the home team.
    away_key_injuries : int | None
        Number of key players OUT for the away team.
    home_injury_severity : float | None
        Aggregate salary / usage share of injured home players (0.0 - 1.0).
    away_injury_severity : float | None
        Aggregate salary / usage share of injured away players (0.0 - 1.0).

    # Situation
    spread : float | None
        Consensus spread, home perspective (positive = home underdog).
    total : float | None
        Consensus game total (over/under).
    is_playoff : bool | int | None
        Whether the game is a playoff game.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import structlog

from moneygone.features.base import Feature, FeatureContext

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(ctx: FeatureContext, key: str, default: Any = None) -> Any:
    snap = ctx.sports_snapshot
    if snap is None:
        return default
    return snap.get(key, default)


def _getf(ctx: FeatureContext, key: str) -> float | None:
    val = _get(ctx, key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _is_home(ctx: FeatureContext) -> bool | None:
    """Return True if the Kalshi market prices the HOME team."""
    val = _get(ctx, "is_home_team")
    if val is None:
        return None
    return bool(int(val))


def _team_perspective(home_val: float | None, away_val: float | None, ctx: FeatureContext) -> float | None:
    """Return the value for the team the Kalshi market is pricing."""
    is_home = _is_home(ctx)
    if is_home is None:
        return None
    return home_val if is_home else away_val


def _decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to raw implied probability."""
    if decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


def _consensus_implied_prob(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Normalise raw implied probabilities to remove overround.

    Returns ``(home_prob, away_prob)`` that sum to 1.0.
    """
    raw_home = _decimal_to_implied_prob(home_odds)
    raw_away = _decimal_to_implied_prob(away_odds)
    total = raw_home + raw_away
    if total <= 0:
        return 0.5, 0.5
    return raw_home / total, raw_away / total


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class SportsbookWinProbability(Feature):
    """Consensus sportsbook implied win probability for the priced team.

    Derived from head-to-head moneyline odds across bookmakers, normalised
    to remove the house overround.  This is the single most important
    signal for game winner markets: Kalshi tends to closely track
    sportsbook consensus but lags on line moves.
    """

    name = "sportsbook_win_prob"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        sb_home = _getf(context, "sportsbook_home_win_prob")
        if sb_home is not None:
            # Already computed and normalised upstream.
            away_prob = 1.0 - sb_home
            return _team_perspective(sb_home, away_prob, context)

        # Compute from raw moneyline odds.
        home_odds = _getf(context, "current_moneyline_home")
        away_odds = _getf(context, "current_moneyline_away")
        if home_odds is None or away_odds is None:
            return None

        home_prob, away_prob = _consensus_implied_prob(home_odds, away_odds)
        return _team_perspective(home_prob, away_prob, context)


class KalshiVsSportsbookEdge(Feature):
    """Difference between Kalshi implied probability and sportsbook consensus.

    ``sportsbook_win_prob - kalshi_implied_prob``

    Positive → Kalshi is underpricing the team relative to sportsbooks.
    Negative → Kalshi overprices relative to sportsbooks.

    This is the core arbitrage signal: Kalshi prediction markets often
    lag sportsbook line moves, especially after injury news or sharp action.
    """

    name = "kalshi_vs_sportsbook_edge"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        sb_prob_feat = SportsbookWinProbability()
        sb_prob = sb_prob_feat.compute(context)
        kalshi_prob = _getf(context, "kalshi_implied_prob")
        if sb_prob is None or kalshi_prob is None:
            return None
        return sb_prob - kalshi_prob


class MoneylineMovement(Feature):
    """Opening vs current moneyline change for the priced team.

    Measures how much the line has moved since open (normalised):
      ``(current_implied_prob - opening_implied_prob)``

    Positive → line moved toward the team (money coming in, possible
    sharp action or injury news favouring them).
    Negative → line moved away (public or sharp fade).

    Line movement is a proxy for sharp money: professional bettors
    move lines; public money typically does not.
    """

    name = "moneyline_movement"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        open_home = _getf(context, "opening_moneyline_home")
        curr_home = _getf(context, "current_moneyline_home")
        open_away = _getf(context, "opening_moneyline_away")
        curr_away = _getf(context, "current_moneyline_away")

        if open_home is None or curr_home is None:
            return None
        # Need away odds to properly normalise.
        if open_away is None or curr_away is None:
            return None

        open_home_prob, open_away_prob = _consensus_implied_prob(open_home, open_away)
        curr_home_prob, curr_away_prob = _consensus_implied_prob(curr_home, curr_away)

        home_move = curr_home_prob - open_home_prob
        away_move = curr_away_prob - open_away_prob
        return _team_perspective(home_move, away_move, context)


class SharpVsPublicBias(Feature):
    """Mismatch between public betting % and the consensus win probability.

    When public bettors heavily back a team (high ``public_pct``), but
    the line doesn't move in that direction, sharp money is opposing
    them.  This "steam" is a key indicator of professional action.

    ``sportsbook_win_prob - public_betting_pct``

    Positive → sportsbooks imply a higher win prob than the public
    consensus → sharps may be on the other side.
    Negative → public aligns with or exceeds sportsbook implied probability.
    """

    name = "sharp_vs_public_bias"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        public_home = _getf(context, "public_pct_home")
        public_away = _getf(context, "public_pct_away")

        # Derive away public % if only home is provided.
        if public_home is not None and public_away is None:
            public_away = 1.0 - public_home
        elif public_away is not None and public_home is None:
            public_home = 1.0 - public_away

        public_pct = _team_perspective(public_home, public_away, context)
        if public_pct is None:
            return None

        sb_prob_feat = SportsbookWinProbability()
        sb_prob = sb_prob_feat.compute(context)
        if sb_prob is None:
            return None

        return sb_prob - public_pct


class PowerRatingEdge(Feature):
    """Team power rating differential (priced team minus opponent).

    Uses Elo ratings, ESPN BPI, Ken Pomeroy ratings, or any numeric
    team rating where a higher value indicates a stronger team.

    Positive → the priced team is stronger by the rating model.
    Negative → the priced team is the underdog by model.

    This is a model-based signal: markets may underprice teams that
    are strong by analytics but weak by public perception.
    """

    name = "power_rating_edge"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        home_rating = _getf(context, "home_team_rating")
        away_rating = _getf(context, "away_team_rating")

        if home_rating is None or away_rating is None:
            return None

        diff_home = home_rating - away_rating
        diff_away = away_rating - home_rating
        return _team_perspective(diff_home, diff_away, context)


class HomeFieldAdvantage(Feature):
    """Home field / court / ice advantage indicator.

    Returns 1.0 if the Kalshi market prices the HOME team, -1.0 if the
    AWAY team, and 0.0 if unknown.  In the NBA, home court advantage is
    worth ~3 points.  In the NFL, ~2.5 points.  In the NHL, ~0.05 in
    win probability terms.

    This encodes the structural edge that home teams win more often,
    independent of market pricing.
    """

    name = "home_field_advantage"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        is_home = _is_home(context)
        if is_home is None:
            return 0.0
        return 1.0 if is_home else -1.0


class TeamInjuryImpact(Feature):
    """Net injury load differential between the two teams.

    ``(away_injury_severity - home_injury_severity)`` from the perspective
    of the priced team.

    Positive → opponent has worse injuries (good for the priced team).
    Negative → the priced team is more injured.

    Uses severity-weighted injury scores when available (salary share or
    usage share of injured players), falling back to raw counts.
    """

    name = "team_injury_impact"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        home_sev = _getf(context, "home_injury_severity")
        away_sev = _getf(context, "away_injury_severity")

        if home_sev is not None and away_sev is not None:
            # Net advantage for the priced team: opponent injuries - own injuries.
            advantage_home = away_sev - home_sev
            advantage_away = home_sev - away_sev
            return _team_perspective(advantage_home, advantage_away, context)

        # Fall back to raw counts.
        home_count = _getf(context, "home_key_injuries")
        away_count = _getf(context, "away_key_injuries")
        if home_count is None and away_count is None:
            return None
        home_count = home_count or 0.0
        away_count = away_count or 0.0

        advantage_home = away_count - home_count
        advantage_away = home_count - away_count
        return _team_perspective(advantage_home, advantage_away, context)


class InjuryAdjustedSpread(Feature):
    """Spread adjusted for late-breaking injury news.

    When key players are ruled out after the line was set, the closing
    spread can be stale.  Computes an approximate adjustment:
      ``raw_spread + injury_adjustment``

    Injury adjustment: each key player out adds/subtracts an estimated
    point value (NBA ~3 pts/star, NFL ~5 pts/QB).  Positive = home
    is bigger underdog than the raw spread implies.
    """

    name = "injury_adjusted_spread"
    dependencies = ()
    lookback = timedelta(0)

    # Points-per-key-player by sport (rough rule of thumb).
    _PTS_PER_PLAYER: dict[str, float] = {
        "nba": 3.0,
        "nfl": 5.0,
        "nhl": 1.5,
        "mlb": 2.0,
        "ncaab": 4.0,
        "ncaaf": 5.0,
    }
    _DEFAULT_PTS = 2.5

    def compute(self, context: FeatureContext) -> float | None:
        spread = _getf(context, "spread")
        if spread is None:
            return None

        home_injuries = _getf(context, "home_key_injuries") or 0.0
        away_injuries = _getf(context, "away_key_injuries") or 0.0

        sport = _get(context, "sport", "").lower()
        pts = self._PTS_PER_PLAYER.get(sport, self._DEFAULT_PTS)

        # More home injuries → home weaker → spread moves in away team's favour.
        adjustment = (home_injuries - away_injuries) * pts
        return spread + adjustment


class SpreadImpliedWinProb(Feature):
    """Convert the point spread to an approximate win probability.

    Uses the standard normal approximation:
      ``P(home wins) = Φ(-spread / sigma)``

    where ``sigma`` is the standard deviation of final margin,
    typically ~13.45 for NBA and ~13.86 for NFL.

    This gives a model-based win probability from spread alone,
    which can be compared to Kalshi pricing.
    """

    name = "spread_implied_win_prob"
    dependencies = ()
    lookback = timedelta(0)

    _SIGMA: dict[str, float] = {
        "nba": 13.45,
        "nfl": 13.86,
        "nhl": 1.35,   # in goals
        "mlb": 1.55,   # in runs
        "ncaab": 11.0,
        "ncaaf": 14.0,
    }
    _DEFAULT_SIGMA = 13.0

    def compute(self, context: FeatureContext) -> float | None:
        spread = _getf(context, "spread")
        if spread is None:
            return None

        sport = _get(context, "sport", "").lower()
        sigma = self._SIGMA.get(sport, self._DEFAULT_SIGMA)

        # spread < 0 → home is favourite → P(home wins) > 0.5
        home_win_prob = 0.5 * math.erfc(spread / (sigma * math.sqrt(2)))
        away_win_prob = 1.0 - home_win_prob
        return _team_perspective(home_win_prob, away_win_prob, context)


class PublicBettingLoad(Feature):
    """Raw public betting percentage for the priced team.

    High public load (> 0.65) often creates value on the other side
    because sportsbooks shade their lines against the public.  When
    public load is high AND the line hasn't moved, sharps are fading
    the public -- strong contrary signal.

    Returns the fraction of public bets on the priced team (0.0 - 1.0).
    """

    name = "public_betting_load"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        public_home = _getf(context, "public_pct_home")
        public_away = _getf(context, "public_pct_away")

        if public_home is not None and public_away is None:
            public_away = 1.0 - public_home
        elif public_away is not None and public_home is None:
            public_home = 1.0 - public_away

        return _team_perspective(public_home, public_away, context)
