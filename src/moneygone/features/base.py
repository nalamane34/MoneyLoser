"""Feature base class and context for the feature engineering pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import structlog

from moneygone.exchange.types import Market, OrderbookSnapshot

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Forward-reference protocols for data types not yet implemented
# ---------------------------------------------------------------------------


@runtime_checkable
class ForecastEnsemble(Protocol):
    """Minimal protocol for a weather forecast ensemble object."""

    location_name: str
    variable: str
    init_time: datetime
    valid_time: datetime
    member_values: list[float]
    ensemble_mean: float
    ensemble_std: float


@runtime_checkable
class DataStore(Protocol):
    """Minimal protocol for the DataStore used by features for lookback queries.

    The store must support a ``query`` method that returns query results
    and must fence results to ``observation_time`` so no future data leaks.
    """

    def query(self, sql: str, params: dict[str, Any] | None = None) -> Any: ...


# ---------------------------------------------------------------------------
# FeatureContext
# ---------------------------------------------------------------------------


@dataclass
class FeatureContext:
    """All data available to a feature at a single observation point.

    Attributes:
        ticker: Market ticker being evaluated.
        observation_time: Point-in-time for the observation (no future data).
        market_state: Current market snapshot, if available.
        orderbook: Current orderbook snapshot, if available.
        weather_ensemble: Current weather forecast ensemble, if available.
        crypto_snapshot: Dict of crypto data (funding rates, OI, etc.).
        sports_snapshot: Dict of sports data (player stats, odds, injuries).
        store: DataStore handle for lookback queries, auto-fenced to
            ``observation_time``.
    """

    ticker: str
    observation_time: datetime
    market_state: Market | None = None
    orderbook: OrderbookSnapshot | None = None
    weather_ensemble: Any | None = None  # ForecastEnsemble when available
    weather_threshold: float | None = None  # Market threshold for weather (e.g., 36°F)
    weather_direction: float | None = None  # 1.0 = above threshold, -1.0 = below threshold
    crypto_snapshot: dict[str, Any] | None = None
    sports_snapshot: dict[str, Any] | None = None
    store: Any | None = None  # DataStore when available


# ---------------------------------------------------------------------------
# Feature ABC
# ---------------------------------------------------------------------------


class Feature(ABC):
    """Abstract base class for all features.

    Subclasses must implement :meth:`compute` and set :attr:`name`.
    Optionally override :meth:`compute_batch` for vectorised computation.

    Attributes:
        name: Unique feature name used as the column key.
        dependencies: Names of other features that must be computed first.
        lookback: How far back this feature needs historical data.
    """

    name: str
    dependencies: tuple[str, ...] = ()
    lookback: timedelta = timedelta(0)

    @abstractmethod
    def compute(self, context: FeatureContext) -> float | None:
        """Compute the feature value for a single observation.

        Returns ``None`` when the required input data is missing.
        """

    def compute_batch(self, contexts: list[FeatureContext]) -> list[float | None]:
        """Compute feature values for multiple observations.

        The default implementation iterates :meth:`compute` over each context.
        Override for vectorised performance when possible.
        """
        results: list[float | None] = []
        for ctx in contexts:
            try:
                results.append(self.compute(ctx))
            except Exception:
                log.warning("feature_compute_error", feature=self.name, ticker=ctx.ticker)
                results.append(None)
        return results

    def __repr__(self) -> str:
        return f"<Feature {self.name}>"
