"""Feature engineering pipeline for prediction market models."""

from moneygone.features.base import Feature, FeatureContext
from moneygone.features.crypto_features import (
    BasisSpread,
    CryptoOrderbookImbalance,
    FundingRateSignal,
    FundingRateZScore,
    OpenInterestChange,
    VolatilityRegime,
    WhaleFlowIndicator,
)
from moneygone.features.market_features import (
    BidAskSpread,
    DepthRatio,
    MidPrice,
    OrderbookImbalance,
    PriceMomentum,
    PriceVelocity,
    TimeToExpiry,
    TradeFlowImbalance,
    VolumeProfile,
    WeightedMidPrice,
)
from moneygone.features.game_winner_features import (
    HomeFieldAdvantage,
    InjuryAdjustedSpread,
    KalshiVsSportsbookEdge,
    MoneylineMovement,
    PinnacleVsMarketEdge,
    PinnacleWinProbability,
    PowerRatingEdge,
    PublicBettingLoad,
    SharpVsPublicBias,
    SpreadImpliedWinProb,
    SportsbookWinProbability,
    TeamInjuryImpact,
)
from moneygone.features.pipeline import FeaturePipeline
from moneygone.features.registry import FeatureRegistry
from moneygone.features.temporal import DayOfWeek, HourOfDay, IsWeekend, TimeToExpiryHours
from moneygone.features.sports_features import (
    GameScript,
    InjuryImpact,
    MatchupEffect,
    MinutesExpected,
    PlayerMean,
    PlayerRecentForm,
    PlayerVariance,
    PropLineVsMarket,
    SharpMoneyIndicator,
    TeamPace,
    UsageRate,
)
from moneygone.features.weather_features import (
    ClimatologicalAnomaly,
    EnsembleExceedanceProb,
    EnsembleMean,
    EnsembleSpread,
    ForecastHorizon,
    ForecastRevisionDirection,
    ForecastRevisionMagnitude,
    ModelDisagreement,
    StationBiasExceedance,
)

__all__ = [
    # Base
    "Feature",
    "FeatureContext",
    "FeaturePipeline",
    "FeatureRegistry",
    # Market
    "BidAskSpread",
    "MidPrice",
    "OrderbookImbalance",
    "WeightedMidPrice",
    "DepthRatio",
    "TradeFlowImbalance",
    "VolumeProfile",
    "TimeToExpiry",
    "PriceVelocity",
    "PriceMomentum",
    # Weather
    "EnsembleMean",
    "EnsembleSpread",
    "EnsembleExceedanceProb",
    "ForecastRevisionMagnitude",
    "ForecastRevisionDirection",
    "ModelDisagreement",
    "ForecastHorizon",
    "ClimatologicalAnomaly",
    # Crypto
    "FundingRateSignal",
    "FundingRateZScore",
    "OpenInterestChange",
    "CryptoOrderbookImbalance",
    "WhaleFlowIndicator",
    "VolatilityRegime",
    "BasisSpread",
    # Sports props
    "PlayerMean",
    "PlayerVariance",
    "PlayerRecentForm",
    "UsageRate",
    "GameScript",
    "MatchupEffect",
    "TeamPace",
    "InjuryImpact",
    "MinutesExpected",
    "PropLineVsMarket",
    "SharpMoneyIndicator",
    # Game winners
    "SportsbookWinProbability",
    "PinnacleWinProbability",
    "KalshiVsSportsbookEdge",
    "PinnacleVsMarketEdge",
    "MoneylineMovement",
    "SharpVsPublicBias",
    "PowerRatingEdge",
    "HomeFieldAdvantage",
    "TeamInjuryImpact",
    "InjuryAdjustedSpread",
    "SpreadImpliedWinProb",
    "PublicBettingLoad",
    # Temporal
    "TimeToExpiryHours",
    "DayOfWeek",
    "HourOfDay",
    "IsWeekend",
]
