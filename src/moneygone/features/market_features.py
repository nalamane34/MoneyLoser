"""Orderbook microstructure features for Kalshi prediction markets."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import numpy as np
import structlog

from moneygone.exchange.types import OrderbookLevel, OrderbookSnapshot
from moneygone.features.base import Feature, FeatureContext

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yes_bid_ask(ob: OrderbookSnapshot) -> tuple[float, float] | None:
    """Extract best yes bid and ask from an orderbook snapshot.

    Yes bids come from yes_levels (sorted descending by price -- best bid first).
    Yes asks come from no_levels converted: yes_ask = 1 - no_bid.
    """
    if not ob.yes_levels and not ob.no_levels:
        return None

    yes_bid = float(ob.yes_levels[0].price) if ob.yes_levels else 0.0
    # The ask side for yes contracts comes from the best no bid:
    # a no bid at price P implies a yes ask at 1-P (in cents).
    yes_ask = (1.0 - float(ob.no_levels[0].price)) if ob.no_levels else 1.0

    if yes_bid <= 0 and yes_ask >= 1.0:
        return None
    return yes_bid, yes_ask


def _levels_to_arrays(
    levels: tuple[OrderbookLevel, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert orderbook levels to numpy arrays of (prices, sizes)."""
    if not levels:
        return np.array([], dtype=float), np.array([], dtype=float)
    prices = np.array([float(lv.price) for lv in levels], dtype=float)
    sizes = np.array([float(lv.contracts) for lv in levels], dtype=float)
    return prices, sizes


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class BidAskSpread(Feature):
    """Spread between best yes ask and best yes bid."""

    name = "bid_ask_spread"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        if context.orderbook is None:
            return None
        ba = _yes_bid_ask(context.orderbook)
        if ba is None:
            return None
        return ba[1] - ba[0]


class MidPrice(Feature):
    """Mid-point of the best yes bid and ask."""

    name = "mid_price"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        if context.orderbook is None:
            return None
        ba = _yes_bid_ask(context.orderbook)
        if ba is None:
            return None
        return (ba[0] + ba[1]) / 2.0


class OrderbookImbalance(Feature):
    """Volume imbalance at the top N orderbook levels.

    Computed as (bid_vol - ask_vol) / (bid_vol + ask_vol).
    Positive values indicate buying pressure.
    """

    name = "orderbook_imbalance"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, n_levels: int = 5) -> None:
        self.n_levels = n_levels

    def compute(self, context: FeatureContext) -> float | None:
        ob = context.orderbook
        if ob is None:
            return None

        _, bid_sizes = _levels_to_arrays(ob.yes_levels)
        _, ask_sizes = _levels_to_arrays(ob.no_levels)

        bid_vol = float(bid_sizes[: self.n_levels].sum()) if len(bid_sizes) > 0 else 0.0
        ask_vol = float(ask_sizes[: self.n_levels].sum()) if len(ask_sizes) > 0 else 0.0

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


class WeightedMidPrice(Feature):
    """Volume-weighted mid price from top-of-book levels.

    Weighted by the opposing side's volume (bid-weighted towards the ask
    and vice versa) which biases the mid towards the heavier side.
    """

    name = "weighted_mid_price"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        ob = context.orderbook
        if ob is None:
            return None
        ba = _yes_bid_ask(ob)
        if ba is None:
            return None
        yes_bid, yes_ask = ba

        bid_size = float(ob.yes_levels[0].contracts) if ob.yes_levels else 0.0
        ask_size = float(ob.no_levels[0].contracts) if ob.no_levels else 0.0

        total = bid_size + ask_size
        if total == 0:
            return (yes_bid + yes_ask) / 2.0

        # Weight bid price by ask size and vice versa
        return (yes_bid * ask_size + yes_ask * bid_size) / total


class DepthRatio(Feature):
    """Ratio of bid depth to ask depth within a configurable price distance.

    Values >1 indicate stronger bid support, <1 stronger ask pressure.
    """

    name = "depth_ratio"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, max_distance: float = 0.10) -> None:
        self.max_distance = max_distance

    def compute(self, context: FeatureContext) -> float | None:
        ob = context.orderbook
        if ob is None:
            return None
        ba = _yes_bid_ask(ob)
        if ba is None:
            return None
        mid = (ba[0] + ba[1]) / 2.0

        bid_prices, bid_sizes = _levels_to_arrays(ob.yes_levels)
        ask_prices, ask_sizes = _levels_to_arrays(ob.no_levels)

        # Filter levels within max_distance of mid
        bid_depth = 0.0
        for p, s in zip(bid_prices, bid_sizes):
            if abs(p - mid) <= self.max_distance:
                bid_depth += s

        ask_depth = 0.0
        for p, s in zip(ask_prices, ask_sizes):
            converted_ask = 1.0 - p
            if abs(converted_ask - mid) <= self.max_distance:
                ask_depth += s

        if ask_depth == 0:
            return None
        return bid_depth / ask_depth


class TradeFlowImbalance(Feature):
    """Net buy vs sell volume over a lookback window.

    Queries the DataStore for recent trades and computes:
    (buy_volume - sell_volume) / total_volume.
    """

    name = "trade_flow_imbalance"
    dependencies = ()

    def __init__(self, lookback_hours: float = 1.0) -> None:
        self.lookback = timedelta(hours=lookback_hours)
        self._lookback_hours = lookback_hours

    def compute(self, context: FeatureContext) -> float | None:
        if context.store is None:
            return None

        cutoff = context.observation_time - self.lookback
        try:
            result = context.store.query(
                "SELECT taker_side, SUM(count) as vol "
                "FROM trades "
                "WHERE ticker = $ticker "
                "  AND trade_time >= $cutoff "
                "  AND trade_time <= $obs_time "
                "GROUP BY taker_side",
                {
                    "ticker": context.ticker,
                    "cutoff": cutoff,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("trade_flow_query_failed", ticker=context.ticker)
            return None

        buy_vol = 0.0
        sell_vol = 0.0
        if result is not None:
            for row in result:
                side = row[0] if isinstance(row, (tuple, list)) else getattr(row, "taker_side", None)
                vol = row[1] if isinstance(row, (tuple, list)) else getattr(row, "vol", 0)
                if side == "yes":
                    buy_vol += float(vol)
                else:
                    sell_vol += float(vol)

        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total


class VolumeProfile(Feature):
    """Skewness of volume distribution across orderbook price levels.

    Positive skew means volume is concentrated at lower prices (more
    bearish), negative skew at higher prices (more bullish).
    """

    name = "volume_profile"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        ob = context.orderbook
        if ob is None:
            return None

        bid_prices, bid_sizes = _levels_to_arrays(ob.yes_levels)
        ask_prices, ask_sizes = _levels_to_arrays(ob.no_levels)

        # Combine all levels into a single volume-at-price distribution
        all_prices = np.concatenate([bid_prices, 1.0 - ask_prices]) if len(ask_prices) > 0 else bid_prices
        all_sizes = np.concatenate([bid_sizes, ask_sizes]) if len(ask_sizes) > 0 else bid_sizes

        if len(all_prices) < 3:
            return 0.0

        # Weighted skewness
        total_vol = all_sizes.sum()
        if total_vol == 0:
            return 0.0

        mean_price = np.average(all_prices, weights=all_sizes)
        variance = np.average((all_prices - mean_price) ** 2, weights=all_sizes)
        if variance == 0:
            return 0.0
        std_price = np.sqrt(variance)
        skew = np.average(((all_prices - mean_price) / std_price) ** 3, weights=all_sizes)
        return float(skew)


class TimeToExpiry(Feature):
    """Hours remaining until the market's close time."""

    name = "time_to_expiry"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        if context.market_state is None:
            return None
        delta = context.market_state.close_time - context.observation_time
        hours = delta.total_seconds() / 3600.0
        return max(hours, 0.0)


class PriceVelocity(Feature):
    """Rate of mid-price change over the lookback period (price/hour).

    Queries the DataStore for historical mid prices.
    """

    name = "price_velocity"
    dependencies = ()

    def __init__(self, lookback_hours: float = 1.0) -> None:
        self.lookback = timedelta(hours=lookback_hours)
        self._lookback_hours = lookback_hours

    def compute(self, context: FeatureContext) -> float | None:
        if context.store is None or context.orderbook is None:
            return None

        ba = _yes_bid_ask(context.orderbook)
        if ba is None:
            return None
        current_mid = (ba[0] + ba[1]) / 2.0

        cutoff = context.observation_time - self.lookback
        try:
            result = context.store.query(
                "SELECT (yes_bid + yes_ask) / 2.0 AS mid "
                "FROM market_states "
                "WHERE ticker = $ticker "
                "  AND ingested_at >= $cutoff "
                "  AND ingested_at <= $obs_time "
                "ORDER BY ingested_at ASC "
                "LIMIT 1",
                {
                    "ticker": context.ticker,
                    "cutoff": cutoff,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("price_velocity_query_failed", ticker=context.ticker)
            return None

        if result is None or len(result) == 0:
            return 0.0

        row = result[0]
        old_mid = float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "mid", current_mid))

        if self._lookback_hours == 0:
            return 0.0
        return (current_mid - old_mid) / self._lookback_hours


class PriceMomentum(Feature):
    """Composite momentum signal across 1h, 4h, and 24h windows.

    Returns the average z-scored price change across windows, where each
    window's change is normalised by a simple estimate of its expected
    magnitude.  Falls back to the available windows if data is sparse.
    """

    name = "price_momentum"
    dependencies = ()
    lookback = timedelta(hours=24)

    _windows_hours: tuple[float, ...] = (1.0, 4.0, 24.0)

    def compute(self, context: FeatureContext) -> float | None:
        if context.store is None or context.orderbook is None:
            return None

        ba = _yes_bid_ask(context.orderbook)
        if ba is None:
            return None
        current_mid = (ba[0] + ba[1]) / 2.0

        scores: list[float] = []
        for window_h in self._windows_hours:
            cutoff = context.observation_time - timedelta(hours=window_h)
            try:
                result = context.store.query(
                    "SELECT (yes_bid + yes_ask) / 2.0 AS mid "
                    "FROM market_states "
                    "WHERE ticker = $ticker "
                    "  AND ingested_at >= $cutoff "
                    "  AND ingested_at <= $obs_time "
                    "ORDER BY ingested_at ASC "
                    "LIMIT 1",
                    {
                        "ticker": context.ticker,
                        "cutoff": cutoff,
                        "obs_time": context.observation_time,
                    },
                )
            except Exception:
                continue

            if result is None or len(result) == 0:
                continue

            row = result[0]
            old_mid = float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "mid", current_mid))

            change = current_mid - old_mid
            # Normalise by sqrt(window) as a rough vol scaling
            norm = np.sqrt(window_h)
            scores.append(change / norm if norm > 0 else 0.0)

        if not scores:
            return 0.0
        return float(np.mean(scores))
