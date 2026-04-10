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
    version = "v3"  # v3: parametric exceedance + refined calibration

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

        # ------- Safety: detect degenerate ensemble (single member) ---------
        # If ensemble_spread is exactly 0.0, we likely have a single
        # deterministic forecast pretending to be an ensemble.  In that case
        # the exceedance probability is binary (0 or 1) with no uncertainty.
        # Apply aggressive shrinkage toward 0.5 and heavily reduce confidence
        # to prevent overconfident bets on essentially a point forecast.
        ensemble_spread = features.get("ensemble_spread", 0.0)

        if ensemble_spread < 1e-9 and raw_prob in (0.0, 1.0):
            # Degenerate ensemble — shrink 30% toward 0.5
            logger.warning(
                "weather_model.degenerate_ensemble",
                exceedance=raw_prob,
                spread=ensemble_spread,
                msg="Single-member ensemble detected, applying heavy shrinkage",
            )
            shrink_pct = 0.30
            probability = raw_prob * (1.0 - shrink_pct) + 0.5 * shrink_pct
            probability = _clip(probability, 0.15, 0.85)

            # Very low confidence for a point forecast
            confidence = 0.30
            horizon = features.get("forecast_horizon", 48.0)
            if horizon > 72:
                confidence = 0.20

            return ModelPrediction(
                probability=probability,
                raw_probability=raw_prob,
                confidence=confidence,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # ------- Calibration: shrink toward 0.5 for model limitations -------
        # Weather ensembles have systematic biases (e.g., cold bias in NOAA
        # GEFS, convective undersampling). Regress all probabilities toward
        # 0.5 by a small amount proportional to how extreme the prediction is.
        #
        # The exceedance prob now comes from parametric fitting when the
        # ensemble is unanimous, so it's already more nuanced than 0/1.
        # Apply a mild universal shrinkage to account for model limitations.

        # Shrinkage strength: higher when spread is low (less member diversity)
        if ensemble_spread < 0.5:
            # Tight ensemble — apply moderate shrinkage (8% toward 0.5)
            shrink_pct = 0.08
        elif ensemble_spread < 2.0:
            # Normal spread — light shrinkage (3% toward 0.5)
            shrink_pct = 0.03
        else:
            # Wide spread — minimal shrinkage (1%)
            shrink_pct = 0.01
        probability = raw_prob * (1.0 - shrink_pct) + 0.5 * shrink_pct

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
