"""Trading strategies for the MoneyGone automated Kalshi trading system."""

from moneygone.strategies.cross_market_arb import (
    ArbConfig,
    ArbLeg,
    ArbOpportunity,
    CrossMarketArbitrage,
)
from moneygone.strategies.live_event_edge import (
    CryptoPriceEdgeEstimator,
    LiveEdgeConfig,
    LiveEdgeSignal,
    LiveEventEdge,
    SportsProbabilityEstimator,
)
from moneygone.strategies.market_maker import (
    MarketMaker,
    MMConfig,
    MMQuote,
    MMState,
)
from moneygone.strategies.resolution_sniper import (
    ContractMapping,
    ResolutionSniper,
    SnipeConfig,
    SnipeOpportunity,
    SnipeRecord,
)

__all__ = [
    "ArbConfig",
    "ArbLeg",
    "ArbOpportunity",
    "ContractMapping",
    "CrossMarketArbitrage",
    "CryptoPriceEdgeEstimator",
    "LiveEdgeConfig",
    "LiveEdgeSignal",
    "LiveEventEdge",
    "MarketMaker",
    "MMConfig",
    "MMQuote",
    "MMState",
    "ResolutionSniper",
    "SnipeConfig",
    "SnipeOpportunity",
    "SnipeRecord",
    "SportsProbabilityEstimator",
]
