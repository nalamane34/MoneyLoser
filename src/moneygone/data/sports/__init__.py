"""Real-time sports data feeds for resolution sniping and prop prediction.

Provides live score tracking via ESPN's public API, player statistics,
injury reports, and sportsbook odds for the sports prop trading pipeline.
"""

from moneygone.data.sports.espn import (
    ESPNLiveFeed,
    GameState,
    OutcomeSignal,
)
from moneygone.data.sports.odds import (
    GameOdds,
    MoneylineOdds,
    OddsAPIFeed,
    PropLine,
)
from moneygone.data.sports.stats import (
    GameLogEntry,
    InjuryReport,
    PlayerInfo,
    PlayerSeasonStats,
    PlayerStatsFeed,
)

__all__ = [
    # ESPN live feed
    "ESPNLiveFeed",
    "GameState",
    "OutcomeSignal",
    # Player stats feed
    "PlayerStatsFeed",
    "PlayerSeasonStats",
    "GameLogEntry",
    "PlayerInfo",
    "InjuryReport",
    # Odds feed
    "OddsAPIFeed",
    "PropLine",
    "MoneylineOdds",
    "GameOdds",
]
