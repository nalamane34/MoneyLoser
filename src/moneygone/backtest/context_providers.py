"""Context providers for injecting external data into backtest FeatureContexts.

Defines the :class:`BacktestContextProvider` protocol and concrete
implementations for crypto, sports, and weather market categories.
Each provider is responsible for loading the relevant snapshot data
for a given ticker at a specific observation time, so the backtest
engine can populate :class:`FeatureContext` fields transparently.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import structlog

from moneygone.data.store import DataStore

logger = structlog.get_logger(__name__)

# Regex to extract the numeric threshold from a crypto ticker.
# Example: KXBTC-26JAN08T0600-B99600 -> 99600
_CRYPTO_THRESHOLD_RE = re.compile(r"-[ABT]([\d.]+)$")

# Kalshi crypto ticker prefix -> CCXT symbol mapping
_CRYPTO_SYMBOL_MAP = {
    "KXBTC": "BTC/USDT",
    "KXBTCD": "BTC/USDT",
    "KXETH": "ETH/USDT",
    "KXETHD": "ETH/USDT",
    "KXDOGE": "DOGE/USDT",
    "KXSHIBA": "SHIB/USDT",
    "KXSHIBAD": "SHIB/USDT",
    "KXSOL": "SOL/USDT",
    "KXSOLD": "SOL/USDT",
    "KXSOLE": "SOL/USDT",
}

# Regex to extract asset prefix from ticker
_CRYPTO_PREFIX_RE = re.compile(r"^(KX[A-Z]+)")


def _ticker_to_ccxt_symbol(ticker: str) -> str | None:
    """Map a Kalshi crypto ticker to its CCXT trading pair symbol."""
    m = _CRYPTO_PREFIX_RE.match(ticker)
    if not m:
        return None
    prefix = m.group(1)
    sym = _CRYPTO_SYMBOL_MAP.get(prefix)
    if sym is None:
        # Try stripping trailing D (daily variant)
        sym = _CRYPTO_SYMBOL_MAP.get(prefix.rstrip("D"))
    return sym


@runtime_checkable
class BacktestContextProvider(Protocol):
    """Protocol for objects that supply external context data during backtests.

    Implementations return a dict whose keys are the optional
    :class:`FeatureContext` field names:

    * ``crypto_snapshot`` -- dict of crypto data (funding rates, OI, etc.)
    * ``sports_snapshot`` -- dict of sports data (player stats, odds, etc.)
    * ``weather_ensemble`` -- :class:`ForecastEnsemble` object
    * ``weather_threshold`` -- numeric threshold from the market ticker
    * ``weather_direction`` -- 1.0 (above) or -1.0 (below)

    Only the keys relevant to the provider's category need be returned.
    """

    def get_context_data(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any]:
        """Return context data for *ticker* at *observation_time*.

        Parameters
        ----------
        ticker:
            The market ticker being evaluated.
        observation_time:
            Point-in-time for the observation.  Implementations must not
            return data from after this timestamp.

        Returns
        -------
        dict[str, Any]
            A dict whose keys correspond to :class:`FeatureContext` fields.
            Missing keys are ignored by the engine.
        """
        ...


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------


class CryptoBacktestContextProvider:
    """Provides crypto snapshot data for crypto-linked Kalshi backtests.

    Queries the ``crypto_context`` table in a :class:`DataStore` for the
    most recent snapshot at or before *observation_time* and parses the
    JSON payload into a ``crypto_snapshot`` dict.

    Also extracts the numeric threshold from the ticker
    (e.g. ``KXBTC-26JAN08T0600-B99600`` -> ``99600``).

    Parameters
    ----------
    store:
        The DataStore that contains the ``crypto_context`` table.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    # -- protocol method --------------------------------------------------

    def get_context_data(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        snapshot = self._query_snapshot(ticker, observation_time)
        if snapshot is not None:
            result["crypto_snapshot"] = snapshot

        threshold = self._extract_threshold(ticker)
        if threshold is not None:
            result["crypto_threshold"] = threshold

        return result

    # -- internals --------------------------------------------------------

    def _query_snapshot(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any] | None:
        """Return the latest crypto_context row at or before *observation_time*.

        The crypto_context table is keyed by CCXT symbol (e.g. ``BTC/USDT``),
        so we first map the Kalshi ticker prefix to its CCXT symbol.
        """
        symbol = _ticker_to_ccxt_symbol(ticker)
        if symbol is None:
            return None

        sql = """
            SELECT snapshot_json
            FROM crypto_context
            WHERE symbol = $symbol
              AND timestamp <= $as_of
            ORDER BY timestamp DESC
            LIMIT 1
        """
        try:
            rows = self._store.query(
                sql,
                {"symbol": symbol, "as_of": observation_time},
            )
            if rows and len(rows) > 0:
                raw = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return raw
        except Exception:
            logger.warning(
                "crypto_context_provider.query_failed",
                ticker=ticker,
                exc_info=True,
            )
        return None

    @staticmethod
    def _extract_threshold(ticker: str) -> float | None:
        """Extract the numeric threshold from a crypto ticker string.

        Examples
        --------
        >>> CryptoBacktestContextProvider._extract_threshold("KXBTC-26JAN08T0600-B99600")
        99600.0
        >>> CryptoBacktestContextProvider._extract_threshold("KXETH-26JAN08-A3200")
        3200.0
        """
        m = _CRYPTO_THRESHOLD_RE.search(ticker)
        if m:
            return float(m.group(1))
        return None


# ---------------------------------------------------------------------------
# Sports
# ---------------------------------------------------------------------------


class SportsBacktestContextProvider:
    """Provides sports snapshot data for sports-linked Kalshi backtests.

    Queries the ``sports_context`` table for the closest snapshot at or
    before *observation_time*.

    Parameters
    ----------
    store:
        The DataStore that contains the ``sports_context`` table.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def get_context_data(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        sql = """
            SELECT snapshot_json
            FROM sports_context
            WHERE ticker = $ticker
              AND timestamp <= $as_of
            ORDER BY timestamp DESC
            LIMIT 1
        """
        try:
            rows = self._store.query(
                sql,
                {"ticker": ticker, "as_of": observation_time},
            )
            if rows and len(rows) > 0:
                raw = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
                if isinstance(raw, str):
                    result["sports_snapshot"] = json.loads(raw)
                elif isinstance(raw, dict):
                    result["sports_snapshot"] = raw
        except Exception:
            logger.warning(
                "sports_context_provider.query_failed",
                ticker=ticker,
                exc_info=True,
            )

        return result


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

# Ticker direction prefix: A = above threshold, B = below threshold
_WEATHER_DIRECTION_MAP = {"A": 1.0, "B": -1.0}
_WEATHER_TICKER_RE = re.compile(r"-([AB])(\d+(?:\.\d+)?)$")


class WeatherBacktestContextProvider:
    """Provides weather ensemble data for weather-linked Kalshi backtests.

    Queries the ``weather_context`` table for the closest forecast
    ensemble at or before *observation_time*.  Also parses the ticker
    to determine the weather threshold and direction.

    Parameters
    ----------
    store:
        The DataStore that contains the ``weather_context`` table.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def get_context_data(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # Parse threshold / direction from ticker
        m = _WEATHER_TICKER_RE.search(ticker)
        if m:
            direction_char, threshold_str = m.group(1), m.group(2)
            result["weather_threshold"] = float(threshold_str)
            result["weather_direction"] = _WEATHER_DIRECTION_MAP.get(
                direction_char, 1.0
            )

        # Query ensemble snapshot
        sql = """
            SELECT ensemble_json
            FROM weather_context
            WHERE ticker = $ticker
              AND timestamp <= $as_of
            ORDER BY timestamp DESC
            LIMIT 1
        """
        try:
            rows = self._store.query(
                sql,
                {"ticker": ticker, "as_of": observation_time},
            )
            if rows and len(rows) > 0:
                raw = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
                if isinstance(raw, str):
                    result["weather_ensemble"] = json.loads(raw)
                elif isinstance(raw, dict):
                    result["weather_ensemble"] = raw
        except Exception:
            logger.warning(
                "weather_context_provider.query_failed",
                ticker=ticker,
                exc_info=True,
            )

        return result


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeBacktestContextProvider:
    """Chains multiple providers, merging their results.

    Later providers overwrite keys from earlier ones if they overlap.

    Parameters
    ----------
    providers:
        Ordered sequence of context providers to call.
    """

    def __init__(self, providers: list[BacktestContextProvider]) -> None:
        self._providers = list(providers)

    def get_context_data(
        self, ticker: str, observation_time: datetime
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for provider in self._providers:
            merged.update(provider.get_context_data(ticker, observation_time))
        return merged
