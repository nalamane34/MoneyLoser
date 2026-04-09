"""Time-based features for prediction market models."""

from __future__ import annotations

from datetime import timedelta

from moneygone.features.base import Feature, FeatureContext


class TimeToExpiryHours(Feature):
    """Hours remaining until the market's close time.

    Returns 0.0 if the market has already closed.
    """

    name = "time_to_expiry_hours"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        if context.market_state is None:
            return None
        delta = context.market_state.close_time - context.observation_time
        hours = delta.total_seconds() / 3600.0
        return max(hours, 0.0)


class DayOfWeek(Feature):
    """Day of the week (0=Monday, 6=Sunday).

    Captures weekly seasonality in market activity and liquidity.
    """

    name = "day_of_week"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return float(context.observation_time.weekday())


class HourOfDay(Feature):
    """Hour of the day in UTC (0-23).

    Captures intraday patterns in trading activity.
    """

    name = "hour_of_day"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return float(context.observation_time.hour)


class IsWeekend(Feature):
    """Binary indicator: 1 if weekend, 0 if weekday.

    Weekend markets often have different liquidity and volatility
    characteristics.
    """

    name = "is_weekend"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        return 1.0 if context.observation_time.weekday() >= 5 else 0.0
