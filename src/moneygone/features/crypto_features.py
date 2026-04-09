"""Crypto market features for crypto-driven prediction markets."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import structlog

from moneygone.features.base import Feature, FeatureContext

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_crypto(context: FeatureContext, key: str, default: float | None = None) -> float | None:
    """Safely extract a value from the crypto snapshot dict."""
    snap = context.crypto_snapshot
    if snap is None:
        return default
    val = snap.get(key)
    if val is None:
        return default
    return float(val)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class FundingRateSignal(Feature):
    """Current perpetual futures funding rate.

    Positive funding means longs pay shorts (bullish bias), negative
    means shorts pay longs (bearish bias).
    """

    name = "funding_rate_signal"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "funding_rate")


class FundingRateZScore(Feature):
    """Z-score of the current funding rate over a lookback window.

    Queries the DataStore for historical funding rates to compute
    the mean and standard deviation.
    """

    name = "funding_rate_zscore"
    dependencies = ()

    def __init__(self, lookback_hours: float = 168.0) -> None:  # 7 days default
        self.lookback = timedelta(hours=lookback_hours)
        self._lookback_hours = lookback_hours

    def compute(self, context: FeatureContext) -> float | None:
        current_rate = _get_crypto(context, "funding_rate")
        if current_rate is None or context.store is None:
            return None

        symbol = context.crypto_snapshot.get("symbol", "") if context.crypto_snapshot else ""
        exchange = context.crypto_snapshot.get("exchange", "") if context.crypto_snapshot else ""
        cutoff = context.observation_time - self.lookback

        try:
            result = context.store.query(
                "SELECT rate FROM funding_rates "
                "WHERE symbol = $symbol "
                "  AND exchange = $exchange "
                "  AND timestamp >= $cutoff "
                "  AND timestamp <= $obs_time "
                "ORDER BY timestamp ASC",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "cutoff": cutoff,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("funding_zscore_query_failed", symbol=symbol)
            return None

        if result is None or len(result) < 2:
            return 0.0

        rates = np.array([
            float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "rate", 0.0))
            for row in result
        ])
        mean = rates.mean()
        std = rates.std(ddof=1)
        if std < 1e-12:
            return 0.0
        return float((current_rate - mean) / std)


class OpenInterestChange(Feature):
    """Percentage change in open interest over a lookback window.

    Rising OI with rising price = new longs entering (bullish).
    Rising OI with falling price = new shorts entering (bearish).
    """

    name = "open_interest_change"
    dependencies = ()

    def __init__(self, lookback_hours: float = 24.0) -> None:
        self.lookback = timedelta(hours=lookback_hours)
        self._lookback_hours = lookback_hours

    def compute(self, context: FeatureContext) -> float | None:
        current_oi = _get_crypto(context, "open_interest")
        if current_oi is None or context.store is None:
            return None

        symbol = context.crypto_snapshot.get("symbol", "") if context.crypto_snapshot else ""
        exchange = context.crypto_snapshot.get("exchange", "") if context.crypto_snapshot else ""
        cutoff = context.observation_time - self.lookback

        try:
            result = context.store.query(
                "SELECT value FROM open_interest "
                "WHERE symbol = $symbol "
                "  AND exchange = $exchange "
                "  AND timestamp >= $cutoff "
                "  AND timestamp <= $obs_time "
                "ORDER BY timestamp ASC "
                "LIMIT 1",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "cutoff": cutoff,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("oi_change_query_failed", symbol=symbol)
            return None

        if result is None or len(result) == 0:
            return 0.0

        row = result[0]
        old_oi = float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "value", current_oi))

        if old_oi == 0:
            return 0.0
        return (current_oi - old_oi) / old_oi


class CryptoOrderbookImbalance(Feature):
    """Bid/ask imbalance in the crypto orderbook.

    Computed as (bid_depth - ask_depth) / (bid_depth + ask_depth).
    Expects ``crypto_snapshot`` to contain ``bid_depth`` and ``ask_depth``.
    """

    name = "crypto_orderbook_imbalance"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        bid = _get_crypto(context, "bid_depth")
        ask = _get_crypto(context, "ask_depth")
        if bid is None or ask is None:
            return None

        total = bid + ask
        if total == 0:
            return 0.0
        return (bid - ask) / total


class WhaleFlowIndicator(Feature):
    """Ratio of large-trade volume to total volume.

    "Large" is defined by a configurable ``whale_threshold`` (in base
    currency units).  Expects ``crypto_snapshot`` to contain
    ``whale_volume`` and ``total_volume``.
    """

    name = "whale_flow_indicator"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, whale_threshold: float = 100_000.0) -> None:
        self.whale_threshold = whale_threshold

    def compute(self, context: FeatureContext) -> float | None:
        whale_vol = _get_crypto(context, "whale_volume")
        total_vol = _get_crypto(context, "total_volume")

        if whale_vol is None or total_vol is None:
            return None
        if total_vol == 0:
            return 0.0
        return whale_vol / total_vol


class VolatilityRegime(Feature):
    """Categorises the current realised volatility into a regime.

    Uses a lookback window to compute realised vol, then compares
    against configurable percentile thresholds to output:
    0.0 = low vol, 0.5 = medium vol, 1.0 = high vol.

    Falls back to the ``realized_vol`` key in crypto_snapshot if store
    is unavailable.
    """

    name = "volatility_regime"
    dependencies = ()

    def __init__(
        self,
        lookback_hours: float = 168.0,
        low_pct: float = 25.0,
        high_pct: float = 75.0,
    ) -> None:
        self.lookback = timedelta(hours=lookback_hours)
        self._lookback_hours = lookback_hours
        self.low_pct = low_pct
        self.high_pct = high_pct

    def compute(self, context: FeatureContext) -> float | None:
        # Try direct value first
        realized_vol = _get_crypto(context, "realized_vol")
        if realized_vol is None:
            return None

        vol_history_str = context.crypto_snapshot.get("vol_history") if context.crypto_snapshot else None
        if vol_history_str is not None and isinstance(vol_history_str, (list, np.ndarray)):
            history = np.array(vol_history_str, dtype=float)
            if len(history) >= 5:
                low_threshold = float(np.percentile(history, self.low_pct))
                high_threshold = float(np.percentile(history, self.high_pct))
                if realized_vol <= low_threshold:
                    return 0.0
                elif realized_vol >= high_threshold:
                    return 1.0
                else:
                    return 0.5

        # Fallback: use simple thresholds on the raw value
        # Typical annualised crypto vol: <40% low, >80% high
        if realized_vol < 0.4:
            return 0.0
        elif realized_vol > 0.8:
            return 1.0
        return 0.5


class BasisSpread(Feature):
    """Futures premium or discount relative to spot price.

    Computed as (futures_price - spot_price) / spot_price.
    Positive = contango (bullish), negative = backwardation (bearish).
    """

    name = "basis_spread"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        futures = _get_crypto(context, "futures_price")
        spot = _get_crypto(context, "spot_price")

        if futures is None or spot is None:
            return None
        if spot == 0:
            return None
        return (futures - spot) / spot


# ---------------------------------------------------------------------------
# Advanced volatility & trend features
# ---------------------------------------------------------------------------


class ATR14(Feature):
    """14-period Average True Range normalized by price.

    Higher ATR = more volatile, wider price swings.
    Normalized so it's comparable across price levels.
    Expects ``crypto_snapshot`` to contain ``atr_14`` from CryptoVolatilityFeed.
    """

    name = "atr_14"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "atr_14")


class ATR24(Feature):
    """24-period (24h hourly) ATR normalized by price."""

    name = "atr_24"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "atr_24")


class RealizedVol24h(Feature):
    """Annualized realized volatility from 24h of hourly returns.

    Computed as std(log_returns) * sqrt(8760).
    """

    name = "realized_vol_24h"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "realized_vol_24h")


class RealizedVol7d(Feature):
    """Annualized realized volatility from 7 days of hourly returns."""

    name = "realized_vol_7d"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "realized_vol_7d")


class RealizedVol30d(Feature):
    """30-day realized volatility from Satochi (btcvol.info)."""

    name = "realized_vol_30d"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "realized_vol_30d")


class ImpliedVolatility(Feature):
    """Implied volatility proxy from Deribit DVOL index.

    Measures the market's expectation of 30-day BTC volatility,
    derived from options prices. When IV > RV, market expects
    increased volatility (risk-off). When IV < RV, market is
    complacent (potential for surprises).
    """

    name = "implied_volatility"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "implied_vol")


class VolSpread(Feature):
    """Implied vol minus realized vol (vol risk premium).

    Positive = market expects more vol than realized (fear).
    Negative = market is complacent relative to recent moves.
    This spread is a key signal for crypto prediction markets:
    high vol spread often precedes directional moves.
    """

    name = "vol_spread"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        iv = _get_crypto(context, "implied_vol")
        rv = _get_crypto(context, "realized_vol_24h")
        if iv is None or rv is None:
            return None
        return iv - rv


class TrendRegime(Feature):
    """Multi-timeframe trend regime classification.

    Uses 8h, 24h, 72h returns to classify into:
    strong_down=-1.0, down=-0.5, neutral=0.0, up=0.5, strong_up=1.0
    """

    name = "trend_regime"
    dependencies = ()
    lookback = timedelta(0)

    _REGIME_MAP = {
        "strong_down": -1.0,
        "down": -0.5,
        "neutral": 0.0,
        "up": 0.5,
        "strong_up": 1.0,
    }

    def compute(self, context: FeatureContext) -> float | None:
        regime = context.crypto_snapshot.get("trend_regime") if context.crypto_snapshot else None
        if regime is None:
            return None
        return self._REGIME_MAP.get(str(regime), 0.0)


class TrendStrength(Feature):
    """Strength of the current trend (0.0 = no trend, 1.0 = strong trend).

    Direction-agnostic magnitude of the trend.
    """

    name = "trend_strength"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "trend_strength")


class BRTIPrice(Feature):
    """Bitcoin Real-Time Index price (or exchange mid-price proxy).

    BRTI is the CME CF Bitcoin Reference Rate calculated in real-time.
    Since BRTI requires a commercial license, we use the exchange
    mid-price as a proxy. In production, upgrade to CF Benchmarks feed.
    """

    name = "brti_price"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return _get_crypto(context, "brti_price")


class BRTIDistanceToThreshold(Feature):
    """Distance from BRTI price to the market's strike threshold.

    Normalized as (brti - threshold) / brti.
    Positive = price above threshold, negative = below.
    The magnitude indicates how far the current price is from the strike.
    """

    name = "brti_distance_to_threshold"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        brti = _get_crypto(context, "brti_price")
        if brti is None or brti == 0:
            return None

        # Try to get threshold from market data
        if context.market_state is None:
            return None

        # floor_strike is stored in market_state for threshold markets
        threshold = getattr(context.market_state, "floor_strike", None)
        if threshold is None:
            return None

        return (brti - float(threshold)) / brti
