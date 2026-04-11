"""Weather ensemble probability model for temperature/precipitation threshold markets.

Uses NOAA/ECMWF ensemble forecast distributions to estimate the probability
of weather exceeding a threshold.  Applies station-specific bias corrections
derived from comparing ensemble grid-point forecasts to NWS CLI observations
(the same data source Kalshi uses for settlement).

v5 changes (from v4):
  - Much tighter probability clip [0.08, 0.92] — never near-certain on weather
  - Steeper spread-dependent shrinkage curve (25% at <0.3°C vs old 8%)
  - Horizon-dependent minimum effective spread floor
  - Minimum effective_std floor of 1.5°C in Gaussian exceedance
  - 0h/hindcast horizon gets heavy skepticism (30% shrinkage)
  - Stronger degenerate ensemble handling
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import structlog
from scipy import stats  # type: ignore[import-untyped]

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Station bias corrections: ensemble_mean - observed_station (°F)
# Computed from 30-day comparison of NOAA GEFS hindcast vs NWS CLI.
# Positive = ensemble warmer than station; negative = ensemble cooler.
#
# These MUST be updated periodically (seasonal drift) via:
#   python scripts/weather_backtest.py bias --days 30
# ---------------------------------------------------------------------------

STATION_BIAS: dict[str, dict[str, float]] = {
    # location_name: {high_bias_f, low_bias_f, high_rmse_f, low_rmse_f}
    # Updated 2026-04-11 from 30-day IEM NWS CLI comparison
    "New York":      {"high_bias_f": -3.4, "low_bias_f":  3.5, "high_rmse_f": 3.8, "low_rmse_f":  4.2},
    "Chicago":       {"high_bias_f": -2.6, "low_bias_f":  1.5, "high_rmse_f": 5.0, "low_rmse_f":  2.8},
    "Los Angeles":   {"high_bias_f": -1.2, "low_bias_f":  3.8, "high_rmse_f": 4.3, "low_rmse_f":  4.2},
    "Miami":         {"high_bias_f": -3.8, "low_bias_f":  3.1, "high_rmse_f": 4.3, "low_rmse_f":  3.5},
    "Dallas":        {"high_bias_f": -1.9, "low_bias_f":  4.9, "high_rmse_f": 2.8, "low_rmse_f":  8.4},
    "Denver":        {"high_bias_f":  0.8, "low_bias_f": 10.2, "high_rmse_f": 4.6, "low_rmse_f": 11.2},
    "Seattle":       {"high_bias_f": -0.2, "low_bias_f":  2.6, "high_rmse_f": 2.3, "low_rmse_f":  5.0},
    "Atlanta":       {"high_bias_f": -3.0, "low_bias_f":  0.4, "high_rmse_f": 4.5, "low_rmse_f":  3.0},
    "Houston":       {"high_bias_f":  0.0, "low_bias_f":  1.7, "high_rmse_f": 2.3, "low_rmse_f":  4.3},
    "Phoenix":       {"high_bias_f": -4.2, "low_bias_f":  4.2, "high_rmse_f": 4.3, "low_rmse_f":  4.3},
    "Minneapolis":   {"high_bias_f": -2.9, "low_bias_f":  1.4, "high_rmse_f": 3.9, "low_rmse_f":  2.9},
    "Oklahoma City": {"high_bias_f": -4.0, "low_bias_f":  8.8, "high_rmse_f": 4.7, "low_rmse_f":  9.9},
    "New Orleans":   {"high_bias_f": -3.5, "low_bias_f":  0.8, "high_rmse_f": 5.6, "low_rmse_f":  2.4},
    "Las Vegas":     {"high_bias_f": -4.5, "low_bias_f":  3.8, "high_rmse_f": 4.6, "low_rmse_f":  9.0},
    "Washington DC": {"high_bias_f": -4.0, "low_bias_f": -0.6, "high_rmse_f": 5.3, "low_rmse_f":  1.3},
    "Austin":        {"high_bias_f": -3.7, "low_bias_f":  6.5, "high_rmse_f": 4.5, "low_rmse_f": 11.1},
}

# Average bias for locations without specific data
_DEFAULT_HIGH_BIAS_F = -2.4
_DEFAULT_LOW_BIAS_F = 3.4
_DEFAULT_HIGH_RMSE_F = 3.9
_DEFAULT_LOW_RMSE_F = 5.3

# NWS rounding uncertainty from F→C→F conversion (~0.5°F = 0.28°C)
_NWS_ROUNDING_UNC_C = 0.28

# Minimum effective std (°C) — ensembles systematically underestimate
# uncertainty (underdispersion).  Even a "perfect" short-range forecast
# has irreducible error from station microclimate, NWS rounding, and
# measurement noise.  1.5°C ≈ 2.7°F floor.
_MIN_EFFECTIVE_STD_C = 1.5


def get_station_bias(location: str, variable: str) -> tuple[float, float]:
    """Get (bias_f, rmse_f) for a location/variable pair.

    Parameters
    ----------
    location: Location name (e.g., "New York")
    variable: "high" or "low"

    Returns
    -------
    (bias_f, rmse_f) in Fahrenheit
    """
    loc_data = STATION_BIAS.get(location, {})
    if variable == "high":
        bias = loc_data.get("high_bias_f", _DEFAULT_HIGH_BIAS_F)
        rmse = loc_data.get("high_rmse_f", _DEFAULT_HIGH_RMSE_F)
    else:
        bias = loc_data.get("low_bias_f", _DEFAULT_LOW_BIAS_F)
        rmse = loc_data.get("low_rmse_f", _DEFAULT_LOW_RMSE_F)
    return bias, rmse


def bias_corrected_exceedance(
    ensemble_mean_c: float,
    ensemble_std_c: float,
    threshold_c: float,
    direction: float,
    location: str,
    variable: str,
) -> float:
    """Compute bias-corrected exceedance probability using Gaussian model.

    Instead of counting raw ensemble members, model the station observation
    as: T_station = ensemble_mean - bias + N(0, σ_effective)

    where σ_effective combines ensemble spread, residual forecast error
    (RMSE after removing systematic bias), and NWS rounding uncertainty.

    Parameters
    ----------
    ensemble_mean_c: Ensemble mean temperature in °C
    ensemble_std_c: Ensemble standard deviation in °C
    threshold_c: Market threshold in °C
    direction: +1.0 for "above", -1.0 for "below"
    location: Location name
    variable: "high" or "low"

    Returns
    -------
    Exceedance probability [0, 1]
    """
    bias_f, rmse_f = get_station_bias(location, variable)
    bias_c = bias_f * 5.0 / 9.0
    rmse_c = rmse_f * 5.0 / 9.0

    # Correct the mean
    corrected_mean = ensemble_mean_c - bias_c

    # Residual uncertainty (RMSE with systematic bias removed)
    residual_std_c = math.sqrt(max(rmse_c ** 2 - bias_c ** 2, 0.01))

    # Effective uncertainty: ensemble + residual + rounding
    raw_effective_std = math.sqrt(
        ensemble_std_c ** 2
        + residual_std_c ** 2
        + _NWS_ROUNDING_UNC_C ** 2
    )
    # Floor: never assume uncertainty below _MIN_EFFECTIVE_STD_C
    effective_std = max(raw_effective_std, _MIN_EFFECTIVE_STD_C)

    z = (threshold_c - corrected_mean) / max(effective_std, 0.01)

    if direction > 0:
        # P(station > threshold)
        return float(1.0 - stats.norm.cdf(z))
    else:
        # P(station < threshold)
        return float(stats.norm.cdf(z))


class WeatherEnsembleModel(ProbabilityModel):
    """Estimate P(weather >= threshold) from ensemble forecast members.

    v4: Uses bias-corrected Gaussian exceedance when station bias data is
    available.  Falls back to raw ensemble exceedance for unknown locations.

    Expected features in the feature dict:
      - ensemble_exceedance_prob: fraction of members exceeding threshold
      - ensemble_mean: mean forecast (°C)
      - ensemble_spread: std dev across members (°C)
      - model_disagreement: NOAA vs ECMWF difference
      - forecast_revision_magnitude: recent revision size
      - forecast_revision_direction: signed direction of revision
      - forecast_horizon: hours from init to valid time
      - climatological_anomaly: deviation from climate normal

    Optional bias-correction features (set by category provider):
      - station_bias_exceedance: Gaussian-corrected P(exceed) from bias model
    """

    name = "weather_ensemble"
    version = "v5"  # v5: conservative calibration, horizon-aware shrinkage

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        # Prefer bias-corrected exceedance if available
        exceedance = features.get("station_bias_exceedance")
        if exceedance is None:
            exceedance = features.get("ensemble_exceedance_prob")

        if exceedance is None:
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        raw_prob = float(exceedance)
        ensemble_spread = features.get("ensemble_spread", 0.0)
        horizon = features.get("forecast_horizon", 0.0)

        # ------- Safety: detect degenerate ensemble (single member) ---------
        if ensemble_spread < 1e-9 and raw_prob in (0.0, 1.0):
            logger.warning(
                "weather_model.degenerate_ensemble",
                exceedance=raw_prob,
                spread=ensemble_spread,
                msg="Single-member ensemble detected, applying heavy shrinkage",
            )
            shrink_pct = 0.40
            probability = raw_prob * (1.0 - shrink_pct) + 0.5 * shrink_pct
            probability = _clip(probability, 0.15, 0.85)
            confidence = 0.25

            return ModelPrediction(
                probability=probability,
                raw_probability=raw_prob,
                confidence=confidence,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # ------- Horizon-aware shrinkage -------
        # Near-zero horizon = hindcast or stale data — unreliable
        # Very long horizon = less skillful — regress toward 0.5
        if horizon < 3.0:
            # Hindcast / stale: heavy skepticism
            horizon_shrink = 0.30
        elif horizon < 12.0:
            horizon_shrink = 0.15
        elif horizon < 24.0:
            horizon_shrink = 0.05
        elif horizon < 72.0:
            horizon_shrink = 0.02
        elif horizon < 120.0:
            horizon_shrink = 0.05
        else:
            # Beyond 5 days: linear regress toward climatology
            horizon_shrink = _clip(0.05 + (horizon - 120) / 200.0, 0.05, 0.35)

        # ------- Spread-dependent shrinkage -------
        # Tighter ensemble → more shrinkage (less member diversity = overconfidence)
        if ensemble_spread < 0.3:
            spread_shrink = 0.20  # Very tight = likely hindcast or underdispersed
        elif ensemble_spread < 0.5:
            spread_shrink = 0.12
        elif ensemble_spread < 1.0:
            spread_shrink = 0.06
        elif ensemble_spread < 2.0:
            spread_shrink = 0.03
        else:
            spread_shrink = 0.01

        # Combine: take the larger of the two shrinkage values
        # (don't double-penalize — the dominant factor drives it)
        total_shrink = max(horizon_shrink, spread_shrink)
        probability = raw_prob * (1.0 - total_shrink) + 0.5 * total_shrink

        # Model disagreement: high disagreement → regress toward 0.5
        disagreement = abs(features.get("model_disagreement", 0.0))
        if disagreement > 2.0:
            shrink = _clip(disagreement * 0.015, 0.0, 0.12)
            probability = probability * (1.0 - shrink) + 0.5 * shrink

        # Forecast revisions — small nudge
        revision_dir = features.get("forecast_revision_direction", 0.0)
        revision_mag = features.get("forecast_revision_magnitude", 0.0)
        probability += _clip(revision_dir * revision_mag * 0.015, -0.02, 0.02)

        # Final clip: never allow near-certainty on weather
        probability = _clip(probability, 0.08, 0.92)

        # Confidence: based on data quality
        base_confidence = 0.65
        confidence = base_confidence
        # Penalize narrow spread (low information)
        confidence -= _clip((1.0 - ensemble_spread) * 0.05, 0.0, 0.15)
        # Penalize model disagreement
        confidence -= _clip(disagreement * 0.02, 0.0, 0.15)
        # Penalize extreme horizons (too close = stale, too far = unskillful)
        if horizon < 6.0:
            confidence -= 0.15  # Stale / hindcast
        elif horizon > 72:
            confidence -= _clip((horizon - 72) / 400.0, 0.0, 0.15)
        confidence = _clip(confidence, 0.25, 0.85)

        return ModelPrediction(
            probability=probability,
            raw_probability=raw_prob,
            confidence=confidence,
            model_name=self.name,
            model_version=self.version,
            features_used=dict(features),
            prediction_time=datetime.now(timezone.utc),
        )

    def predict_proba_batch(self, features: pd.DataFrame) -> list[ModelPrediction]:
        return [
            self.predict_proba(
                {k: float(v) for k, v in row.items() if pd.notna(v)}
            )
            for _, row in features.iterrows()
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weights=None) -> None:
        pass  # No training needed — ensemble is the model

    def save(self, path) -> None:
        pass

    @classmethod
    def load(cls, path) -> "WeatherEnsembleModel":
        return cls()
