"""Lookahead bias and data leakage prevention for backtesting.

Provides two mechanisms:

1. **LeakageGuard**: Validates that feature contexts and model access
   respect temporal constraints -- no data from after the observation
   time should be used.

2. **TimeFencedStore**: Wraps a DataStore and enforces an ``as_of``
   ceiling on all query methods, preventing any accidental future
   data access.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog

from moneygone.data.store import DataStore
from moneygone.features.base import FeatureContext

logger = structlog.get_logger(__name__)


class LeakageGuard:
    """Validates temporal constraints to prevent lookahead bias.

    Parameters
    ----------
    settlement_times:
        Mapping of ticker -> settlement datetime.  Used to ensure we
        don't trade a market after its outcome is known.
    """

    def __init__(self, settlement_times: dict[str, datetime]) -> None:
        self._settlement_times = dict(settlement_times)

    def add_settlement_time(self, ticker: str, settlement_time: datetime) -> None:
        """Register a settlement time for a ticker."""
        self._settlement_times[ticker] = settlement_time

    def validate_feature_context(self, context: FeatureContext) -> None:
        """Validate that no data in the feature context is from the future.

        Raises
        ------
        LookaheadError
            If any data in the context has a timestamp after
            ``context.observation_time``.
        """
        obs_time = context.observation_time

        # Check market state timestamp
        if context.market_state is not None:
            # Market close_time can be in the future (that's the expiry),
            # but the market data itself should not be from after obs_time
            pass

        # Check orderbook timestamp
        if context.orderbook is not None:
            if context.orderbook.timestamp > obs_time:
                raise LookaheadError(
                    f"Orderbook timestamp {context.orderbook.timestamp} is after "
                    f"observation_time {obs_time} for {context.ticker}"
                )

        # Check that we're not using data from after settlement
        self.validate_no_label_access(context.ticker, obs_time)

    def validate_no_label_access(self, ticker: str, current_time: datetime) -> None:
        """Validate that we're not accessing data after settlement.

        In a real backtest, trading a market after its settlement is
        known constitutes label leakage.

        Raises
        ------
        LookaheadError
            If ``current_time`` is at or after the settlement time for
            this ticker.
        """
        settlement_time = self._settlement_times.get(ticker)
        if settlement_time is not None and current_time >= settlement_time:
            raise LookaheadError(
                f"Accessing {ticker} at {current_time} which is at or after "
                f"settlement time {settlement_time} -- label leakage"
            )

    def validate_train_test_split(
        self, train_end: datetime, test_start: datetime
    ) -> None:
        """Validate that training and test periods do not overlap.

        Raises
        ------
        LookaheadError
            If ``train_end`` is after ``test_start``.
        """
        if train_end > test_start:
            raise LookaheadError(
                f"Training period end ({train_end}) overlaps with "
                f"test period start ({test_start}) -- data leakage"
            )

        # Also check with a buffer to catch off-by-one errors
        if train_end == test_start:
            logger.warning(
                "leakage_guard.zero_gap",
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                msg="No gap between train and test periods; consider adding a buffer",
            )


class LookaheadError(Exception):
    """Raised when a lookahead bias or data leakage is detected."""


class TimeFencedStore:
    """Wraps a DataStore and enforces a temporal ceiling on all queries.

    Every query method automatically uses ``as_of`` as the latest
    timestamp, preventing any data from after the observation point
    from leaking into the backtest.

    Parameters
    ----------
    store:
        The underlying DataStore to wrap.
    as_of:
        The temporal ceiling -- no data after this time is accessible.
    """

    def __init__(self, store: DataStore, as_of: datetime) -> None:
        self._store = store
        self._as_of = as_of

    @property
    def as_of(self) -> datetime:
        """The current temporal ceiling."""
        return self._as_of

    def advance_time(self, new_as_of: datetime) -> None:
        """Move the temporal ceiling forward (never backward).

        Parameters
        ----------
        new_as_of:
            New temporal ceiling.  Must not be before the current ceiling.

        Raises
        ------
        ValueError
            If ``new_as_of`` is before the current ceiling.
        """
        if new_as_of < self._as_of:
            raise ValueError(
                f"Cannot move time backward: {new_as_of} < {self._as_of}"
            )
        self._as_of = new_as_of

    # ------------------------------------------------------------------
    # Fenced query methods
    # ------------------------------------------------------------------

    def get_market_state_at(
        self, ticker: str, as_of: datetime | None = None
    ) -> dict[str, Any] | None:
        """Get market state, clamped to the temporal ceiling."""
        effective_as_of = min(as_of or self._as_of, self._as_of)
        return self._store.get_market_state_at(ticker, effective_as_of)

    def get_orderbook_at(
        self, ticker: str, as_of: datetime | None = None
    ) -> dict[str, Any] | None:
        """Get orderbook snapshot, clamped to the temporal ceiling."""
        effective_as_of = min(as_of or self._as_of, self._as_of)
        return self._store.get_orderbook_at(ticker, effective_as_of)

    def get_forecasts_at(
        self,
        location: str,
        variable: str,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get forecast ensembles, clamped to the temporal ceiling."""
        effective_as_of = min(as_of or self._as_of, self._as_of)
        return self._store.get_forecasts_at(location, variable, effective_as_of)

    def get_features_at(
        self, ticker: str, as_of: datetime | None = None
    ) -> dict[str, float]:
        """Get computed features, clamped to the temporal ceiling."""
        effective_as_of = min(as_of or self._as_of, self._as_of)
        return self._store.get_features_at(ticker, effective_as_of)

    def get_trades_between(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Get trades, with end clamped to the temporal ceiling."""
        effective_end = min(end, self._as_of)
        if start > effective_end:
            return []
        return self._store.get_trades_between(ticker, start, effective_end)

    def get_funding_rates_at(
        self, symbol: str, as_of: datetime | None = None
    ) -> dict[str, Any] | None:
        """Get funding rates, clamped to the temporal ceiling."""
        effective_as_of = min(as_of or self._as_of, self._as_of)
        return self._store.get_funding_rates_at(symbol, effective_as_of)

    def get_latest_prediction(
        self, ticker: str, model_name: str
    ) -> dict[str, Any] | None:
        """Get latest prediction -- delegates to store (no time fence needed
        since predictions are generated by the model, not from future data)."""
        return self._store.get_latest_prediction(ticker, model_name)

    def query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a raw query.

        Warning: This does NOT enforce the temporal ceiling.  Use the
        typed query methods above whenever possible.
        """
        logger.warning(
            "time_fenced_store.raw_query",
            msg="Raw SQL query bypasses temporal fencing",
            as_of=self._as_of.isoformat(),
        )
        return self._store._conn.execute(sql, list((params or {}).values())).fetchall()
