"""Weather ensemble probability model for temperature/precipitation threshold markets.

Uses NOAA/ECMWF ensemble forecast distributions to estimate the probability
of weather exceeding a threshold.  The ensemble directly gives us a
probability distribution — no parametric assumptions needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class WeatherEnsembleModel(ProbabilityModel):
    """Estimate P(weather >= threshold) from ensemble forecast members.

    Expected features in the feature dict:
      - ensemble_exceedance_prob: fraction of members exceeding threshold
      - ensemble_mean: mean forecast
      - ensemble_spread: std dev across members (uncertainty)
      - model_disagreement: NOAA vs ECMWF difference
      - forecast_revision_magnitude: recent revision size
      - forecast_revision_direction: signed direction of revision
      - forecast_horizon: hours from init to valid time
      - climatological_anomaly: deviation from climate normal
    """

    name = "weather_ensemble"
    version = "v2"  # v2: direction-aware + calibration shrinkage

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
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

        # ------- Calibration: shrink extreme ensemble fractions -------
        # Raw ensemble fractions (0/30=0.0, 30/30=1.0) overstate certainty
        # because ensembles have finite resolution and common biases.
        # Apply Laplace-style smoothing: treat each extreme like we've
        # seen 1 additional "opposite" member.  For a 30-member ensemble:
        #   0/30 → 1/32 ≈ 0.03,   30/30 → 31/32 ≈ 0.97
        #   1/30 → 2/32 ≈ 0.06,   29/30 → 30/32 ≈ 0.94
        ensemble_spread = features.get("ensemble_spread", 0.0)
        # Estimate member count from spread — if spread=0, all members agree
        # Use a stronger shrinkage when ensemble is unanimous (spread ~0)
        if ensemble_spread < 0.5:
            # Unanimous or near-unanimous ensemble — apply heavier shrinkage
            # Regress toward 0.5 by 5% to account for model limitations
            probability = raw_prob * 0.95 + 0.5 * 0.05
        else:
            probability = raw_prob

        # Model disagreement: high disagreement → regress toward 0.5
        disagreement = abs(features.get("model_disagreement", 0.0))
        if disagreement > 2.0:
            shrink = _clip(disagreement * 0.01, 0.0, 0.10)
            probability = probability * (1.0 - shrink) + 0.5 * shrink

        # Forecast revisions: if forecasts are trending toward threshold, adjust
        revision_dir = features.get("forecast_revision_direction", 0.0)
        revision_mag = features.get("forecast_revision_magnitude", 0.0)
        probability += _clip(revision_dir * revision_mag * 0.02, -0.03, 0.03)

        # Horizon: closer forecasts are more reliable
        horizon = features.get("forecast_horizon", 48.0)
        if horizon > 120:
            # Beyond 5 days, regress toward climatology
            regress = _clip((horizon - 120) / 240.0, 0.0, 0.3)
            probability = probability * (1.0 - regress) + 0.5 * regress

        # Final clip: never allow absolute certainty
        probability = _clip(probability, 0.03, 0.97)

        # Confidence: ensemble-based with quality modifiers
        spread = features.get("ensemble_spread", 5.0)
        base_confidence = 0.70  # Ensembles are well-calibrated
        # High spread → lower confidence
        confidence = base_confidence - _clip(spread * 0.01, 0.0, 0.20)
        # High model disagreement → lower confidence
        confidence -= _clip(disagreement * 0.02, 0.0, 0.15)
        # Distant horizon → lower confidence
        if horizon > 72:
            confidence -= _clip((horizon - 72) / 500.0, 0.0, 0.15)
        confidence = _clip(confidence, 0.30, 0.90)

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
