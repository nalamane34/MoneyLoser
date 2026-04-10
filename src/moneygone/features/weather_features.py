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


def _target_timestep_idx(context: FeatureContext) -> int | None:
    """Return the index of the forecast timestep closest to market close."""
    ens = context.weather_ensemble
    if ens is None:
        return None
    valid_times = getattr(ens, "valid_times", None)
    member_values = getattr(ens, "member_values", None)
    if not valid_times or not member_values or not member_values[0]:
        return None

    n_timesteps = len(member_values[0])
    target_time = None
    if context.market_state is not None:
        target_time = getattr(context.market_state, "close_time", None)

    if target_time is None or len(valid_times) != n_timesteps:
        return n_timesteps - 1

    best_idx = 0
    best_delta = abs((valid_times[0] - target_time).total_seconds())
    for i, vt in enumerate(valid_times):
        delta = abs((vt - target_time).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_idx = i
    return best_idx


def _get_ensemble_mean_scalar(context: FeatureContext) -> float | None:
    """Get the ensemble mean at the target timestep as a scalar."""
    ens = context.weather_ensemble
    if ens is None:
        return None
    idx = _target_timestep_idx(context)
    if idx is None:
        return None
    em = getattr(ens, "ensemble_mean", None)
    if em is not None and isinstance(em, list) and len(em) > idx:
        return float(em[idx])
    # Fall back to computing from members
    members = _get_members(context)
    if members is not None:
        return float(np.mean(members))
    return None


def _detect_minmax_mode(context: FeatureContext) -> str:
    """Detect whether this market asks about daily min, max, or instantaneous temp.

    Returns 'min' for KXLOWT, 'max' for KXHIGHT, 'instant' for everything else.
    """
    ticker = context.ticker or ""
    if "KXLOWT" in ticker.upper():
        return "min"
    if "KXHIGHT" in ticker.upper():
        return "max"
    return "instant"


def _get_members(context: FeatureContext) -> list[float] | None:
    """Extract ensemble member values for the market's question.

    ``member_values`` is shaped ``[n_members][n_timesteps]``.

    For min-temperature markets (KXLOWT): returns the MINIMUM value
    across all timesteps up to market close for each member.
    For max-temperature markets (KXHIGHT): returns the MAXIMUM value.
    For all other markets: returns the value at the timestep closest
    to market close.
    """
    ens = context.weather_ensemble
    if ens is None:
        return None
    member_values = getattr(ens, "member_values", None)
    if member_values is None or len(member_values) == 0:
        return None

    # Each member is a list of values across timesteps
    n_timesteps = len(member_values[0]) if member_values[0] else 0
    if n_timesteps == 0:
        return None

    valid_times = getattr(ens, "valid_times", None)
    target_time = None
    if context.market_state is not None:
        target_time = getattr(context.market_state, "close_time", None)

    mode = _detect_minmax_mode(context)

    if mode in ("min", "max") and n_timesteps > 1:
        # For daily min/max markets: aggregate across all timesteps up
        # to market close, taking min or max per member.
        # This correctly answers "what is the lowest/highest temp today?"
        end_idx = n_timesteps  # default: use all timesteps
        if valid_times and target_time is not None and len(valid_times) == n_timesteps:
            # Only include timesteps up to (and slightly past) market close
            for i, vt in enumerate(valid_times):
                if vt > target_time:
                    end_idx = max(i, 1)  # at least 1 timestep
                    break

        results = []
        agg_fn = min if mode == "min" else max
        for member in member_values:
            vals = [float(member[i]) for i in range(min(end_idx, len(member)))]
            if vals:
                results.append(agg_fn(vals))
        return results if results else None

    # Default: single timestep closest to market close
    idx = n_timesteps - 1
    if valid_times and target_time is not None and len(valid_times) == n_timesteps:
        best_delta = abs((valid_times[0] - target_time).total_seconds())
        for i, vt in enumerate(valid_times):
            delta = abs((vt - target_time).total_seconds())
            if delta < best_delta:
                best_delta = delta
                idx = i

    return [float(m[idx]) for m in member_values if len(m) > idx]


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
    """Fraction of ensemble members exceeding the market threshold.

    Reads the threshold from ``context.weather_threshold`` (extracted from
    the market title), falling back to a configurable default.
    """

    name = "ensemble_exceedance_prob"
    dependencies = ()
    lookback = timedelta(0)

    def __init__(self, default_threshold: float = 0.0) -> None:
        self._default_threshold = default_threshold

    def compute(self, context: FeatureContext) -> float | None:
        members = _get_members(context)
        if members is None:
            return None

        # Use market-specific threshold if available
        threshold = getattr(context, "weather_threshold", None)
        if threshold is None:
            threshold = self._default_threshold

        arr = np.array(members)
        exceedance = float(np.mean(arr >= threshold))

        # Direction: 1.0 = "above" market (YES if value >= threshold)
        #           -1.0 = "below" market (YES if value < threshold)
        direction = getattr(context, "weather_direction", None)
        if direction is not None and direction < 0:
            # "Below X" market: P(YES) = P(value < threshold) = 1 - P(value >= threshold)
            return 1.0 - exceedance
        return exceedance


class ForecastRevisionMagnitude(Feature):
    """Absolute magnitude of the forecast revision from the previous cycle.

    Queries the DataStore for the prior ensemble mean and computes
    |current_mean - previous_mean|.
    """

    name = "forecast_revision_magnitude"
    dependencies = ()
    lookback = timedelta(hours=12)

    def compute(self, context: FeatureContext) -> float | None:
        current_mean = _get_ensemble_mean_scalar(context)
        if current_mean is None or context.store is None:
            return 0.0  # No revision info available

        ens = context.weather_ensemble
        location = getattr(ens, "location_name", "")
        variable = getattr(ens, "variable", "")

        try:
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
            return 0.0

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
        current_mean = _get_ensemble_mean_scalar(context)
        if current_mean is None or context.store is None:
            return 0.0

        ens = context.weather_ensemble
        location = getattr(ens, "location_name", "")
        variable = getattr(ens, "variable", "")

        try:
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
            return 0.0

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

        current_mean = _get_ensemble_mean_scalar(context)
        if current_mean is None:
            return None
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
        if init_time is None:
            return None

        # Use the target timestep's valid time
        idx = _target_timestep_idx(context)
        valid_times = getattr(ens, "valid_times", None)
        if idx is None or not valid_times or idx >= len(valid_times):
            return None

        delta = valid_times[idx] - init_time
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
        current_mean = _get_ensemble_mean_scalar(context)
        if current_mean is None:
            return None
        return current_mean - self.normal
