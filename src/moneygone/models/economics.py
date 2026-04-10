"""Economics probability model for Fed rate, CPI, jobs, and GDP markets.

Uses FRED data, Fed funds futures implied probabilities, and survey
consensus to estimate the probability of economic thresholds being met.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class EconomicsModel(ProbabilityModel):
    """Estimate probabilities for economic indicator threshold markets.

    Expected features in the feature dict:
      - latest_value: most recent release value
      - previous_value: prior release value
      - threshold: market threshold
      - direction: 1.0 = "above", -1.0 = "below"
      - trend_3m: 3-month trend (slope)
      - survey_consensus: consensus estimate if available
      - fed_funds_implied: implied rate from futures
      - hours_to_release: hours until data release
    """

    name = "economics"
    version = "v1"

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        latest = features.get("latest_value")
        threshold = features.get("threshold")
        direction = features.get("direction", 1.0)

        if latest is None or threshold is None:
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # Base probability from current value vs threshold
        previous = features.get("previous_value", latest)
        trend = features.get("trend_3m", 0.0)
        consensus = features.get("survey_consensus")

        # Project forward using trend
        projected = latest + trend

        # If we have survey consensus, weight it heavily
        if consensus is not None:
            projected = 0.4 * projected + 0.6 * consensus

        # Distance from threshold relative to recent volatility
        diff = projected - threshold
        recent_change = abs(latest - previous) if previous != latest else abs(latest * 0.01)
        if recent_change < 0.0001:
            recent_change = abs(threshold * 0.01)

        z_score = diff / recent_change if recent_change > 0 else 0.0

        # Convert z-score to probability (sigmoid)
        import math
        raw_prob = 1.0 / (1.0 + math.exp(-z_score * 0.8))

        # If market is "below" threshold, flip
        if direction < 0:
            raw_prob = 1.0 - raw_prob

        probability = _clip(raw_prob, 0.01, 0.99)

        # Fed funds implied gives us a market-consensus anchor
        fed_implied = features.get("fed_funds_implied")
        if fed_implied is not None and threshold is not None:
            # Blend our estimate with market implied
            probability = 0.6 * probability + 0.4 * fed_implied

        probability = _clip(probability, 0.01, 0.99)

        # Confidence based on data availability
        data_points = sum(
            1 for k in (
                "latest_value", "previous_value", "trend_3m",
                "survey_consensus", "fed_funds_implied",
            )
            if k in features and features[k] is not None
        )
        confidence = _clip(0.35 + 0.10 * data_points, 0.30, 0.80)

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
        pass

    def save(self, path) -> None:
        pass

    @classmethod
    def load(cls, path) -> "EconomicsModel":
        return cls()
