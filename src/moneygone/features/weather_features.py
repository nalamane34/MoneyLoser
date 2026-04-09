"""Weather ensemble forecast features for weather-driven prediction markets."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import structlog

from moneygone.features.base import Feature, FeatureContext

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_members(context: FeatureContext) -> list[float] | None:
    """Extract ensemble member values from the weather ensemble."""
    ens = context.weather_ensemble
    if ens is None:
        return None
    members = getattr(ens, "member_values", None)
    if members is None or len(members) == 0:
        return None
    return [float(v) for v in members]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class EnsembleMean(Feature):
    """Mean forecast value across all ensemble members."""

    name = "ensemble_mean"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        members = _get_members(context)
        if members is None:
            return None
        return float(np.mean(members))


class EnsembleSpread(Feature):
    """Standard deviation of forecasts across ensemble members.

    Higher spread indicates greater forecast uncertainty.
    """

    name = "ensemble_spread"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        members = _get_members(context)
        if members is None:
            return None
        if len(members) < 2:
            return 0.0
        return float(np.std(members, ddof=1))


class EnsembleExceedanceProb(Feature):
    """Fraction of ensemble members exceeding a configurable threshold.

    This directly maps to probability for threshold-based weather markets
    (e.g., "Will temperature exceed 100F?").
    """

    name = "ensemble_exceedance_prob"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def compute(self, context: FeatureContext) -> float | None:
        members = _get_members(context)
        if members is None:
            return None
        arr = np.array(members)
        return float(np.mean(arr > self.threshold))


class ForecastRevisionMagnitude(Feature):
    """Absolute magnitude of the forecast revision from the previous cycle.

    Queries the DataStore for the prior ensemble mean and computes
    |current_mean - previous_mean|.
    """

    name = "forecast_revision_magnitude"
    dependencies = ()
    lookback = timedelta(hours=12)

    def compute(self, context: FeatureContext) -> float | None:
        ens = context.weather_ensemble
        if ens is None:
            return None

        current_mean = getattr(ens, "ensemble_mean", None)
        if current_mean is None:
            members = _get_members(context)
            if members is None:
                return None
            current_mean = float(np.mean(members))
        else:
            current_mean = float(current_mean)

        if context.store is None:
            return None

        try:
            location = getattr(ens, "location_name", "")
            variable = getattr(ens, "variable", "")
            result = context.store.query(
                "SELECT ensemble_mean "
                "FROM forecast_ensembles "
                "WHERE location_name = $location "
                "  AND variable = $variable "
                "  AND ingested_at < $obs_time "
                "ORDER BY ingested_at DESC "
                "LIMIT 1",
                {
                    "location": location,
                    "variable": variable,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("forecast_revision_query_failed")
            return None

        if result is None or len(result) == 0:
            return 0.0

        row = result[0]
        prev_mean = float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "ensemble_mean", current_mean))
        return abs(current_mean - prev_mean)


class ForecastRevisionDirection(Feature):
    """Signed direction of the forecast revision.

    Returns +1 for upward revision (e.g., warmer), -1 for downward
    (e.g., cooler), 0 for no change or missing data.
    """

    name = "forecast_revision_direction"
    dependencies = ()
    lookback = timedelta(hours=12)

    def compute(self, context: FeatureContext) -> float | None:
        ens = context.weather_ensemble
        if ens is None:
            return None

        current_mean = getattr(ens, "ensemble_mean", None)
        if current_mean is None:
            members = _get_members(context)
            if members is None:
                return None
            current_mean = float(np.mean(members))
        else:
            current_mean = float(current_mean)

        if context.store is None:
            return None

        try:
            location = getattr(ens, "location_name", "")
            variable = getattr(ens, "variable", "")
            result = context.store.query(
                "SELECT ensemble_mean "
                "FROM forecast_ensembles "
                "WHERE location_name = $location "
                "  AND variable = $variable "
                "  AND ingested_at < $obs_time "
                "ORDER BY ingested_at DESC "
                "LIMIT 1",
                {
                    "location": location,
                    "variable": variable,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("forecast_revision_dir_query_failed")
            return None

        if result is None or len(result) == 0:
            return 0.0

        row = result[0]
        prev_mean = float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "ensemble_mean", current_mean))

        diff = current_mean - prev_mean
        if abs(diff) < 1e-9:
            return 0.0
        return 1.0 if diff > 0 else -1.0


class ModelDisagreement(Feature):
    """Absolute difference between NOAA and ECMWF ensemble means.

    Expects the weather_ensemble to carry a ``source`` attribute or the
    DataStore to have both NOAA and ECMWF entries.  When only a single
    source is available, queries the store for the other.
    """

    name = "model_disagreement"
    dependencies = ()
    lookback = timedelta(hours=12)

    def compute(self, context: FeatureContext) -> float | None:
        ens = context.weather_ensemble
        if ens is None or context.store is None:
            return None

        current_mean = float(getattr(ens, "ensemble_mean", 0.0))
        location = getattr(ens, "location_name", "")
        variable = getattr(ens, "variable", "")

        # Try to get the alternate model's forecast from the store
        try:
            result = context.store.query(
                "SELECT ensemble_mean "
                "FROM forecast_ensembles "
                "WHERE location_name = $location "
                "  AND variable = $variable "
                "  AND ingested_at <= $obs_time "
                "ORDER BY ingested_at DESC "
                "LIMIT 2",
                {
                    "location": location,
                    "variable": variable,
                    "obs_time": context.observation_time,
                },
            )
        except Exception:
            log.warning("model_disagreement_query_failed")
            return None

        if result is None or len(result) < 2:
            return 0.0

        means = [
            float(row[0]) if isinstance(row, (tuple, list)) else float(getattr(row, "ensemble_mean", 0.0))
            for row in result[:2]
        ]
        return abs(means[0] - means[1])


class ForecastHorizon(Feature):
    """Hours between forecast initialization time and target valid time.

    Shorter horizons generally produce more accurate forecasts.
    """

    name = "forecast_horizon"
    dependencies = ()
    lookback = timedelta(0)

    def compute(self, context: FeatureContext) -> float | None:
        ens = context.weather_ensemble
        if ens is None:
            return None

        init_time = getattr(ens, "init_time", None)
        valid_time = getattr(ens, "valid_time", None)
        if init_time is None or valid_time is None:
            return None

        delta = valid_time - init_time
        return max(delta.total_seconds() / 3600.0, 0.0)


class ClimatologicalAnomaly(Feature):
    """Deviation of the forecast ensemble mean from a 30-year climatological normal.

    Uses a configurable ``normal`` value as a placeholder for the true
    climatological baseline (which would come from an external dataset
    in production).  Positive = warmer/wetter than normal.
    """

    name = "climatological_anomaly"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, normal: float = 0.0) -> None:
        self.normal = normal

    def compute(self, context: FeatureContext) -> float | None:
        ens = context.weather_ensemble
        if ens is None:
            return None

        current_mean = getattr(ens, "ensemble_mean", None)
        if current_mean is None:
            members = _get_members(context)
            if members is None:
                return None
            current_mean = float(np.mean(members))
        else:
            current_mean = float(current_mean)

        return current_mean - self.normal
