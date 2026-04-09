"""Timestamp and datetime utilities.

All helpers produce or consume timezone-aware datetimes in UTC.  Kalshi's
exchange operates 24/7, but the API may undergo scheduled maintenance;
:func:`is_market_hours` checks for known maintenance windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def now_utc() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def to_timestamp_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds since the Unix epoch.

    If *dt* is naive it is assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_timestamp_ms(ts: int) -> datetime:
    """Convert milliseconds since epoch to a UTC-aware datetime."""
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)


def parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 string into a UTC-aware datetime.

    Handles the trailing ``Z`` shorthand that Kalshi uses, as well as
    explicit ``+00:00`` offsets.
    """
    cleaned = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def time_until(target: datetime) -> timedelta:
    """Return the timedelta from *now* to *target*.

    Returns a negative timedelta if *target* is in the past.  If *target*
    is naive it is assumed to be UTC.
    """
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return target - now_utc()


def is_market_hours() -> bool:
    """Return ``True`` if the Kalshi exchange is expected to be open.

    Kalshi trades 24 hours a day, 7 days a week.  However, the exchange
    undergoes a brief scheduled maintenance window every weekday from
    approximately 03:00 to 03:10 UTC (11pm - 11:10pm ET).  This function
    returns ``False`` during that window and ``True`` otherwise.
    """
    now = now_utc()
    # Maintenance window: weekdays 03:00-03:10 UTC
    if now.weekday() < 5:  # Monday=0 ... Friday=4
        if now.hour == 3 and now.minute < 10:
            return False
    return True
