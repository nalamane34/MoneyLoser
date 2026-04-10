"""Crypto volatility-based probability model for price threshold markets.

Uses realized volatility + current price + time to expiry to estimate the
probability of a crypto asset reaching a given price threshold via a
log-normal distribution.  Also incorporates funding rate, OI, and trend
regime as directional adjustments.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class CryptoVolModel(ProbabilityModel):
    """Estimate P(price >= threshold) using vol-based log-normal model.

    Expected features in the feature dict:
      - brti_price (or current_price): current asset price
      - threshold_price: strike/target price from the market
      - hours_to_expiry: time remaining
      - realized_vol_24h: annualized realized vol (primary)
      - implied_vol: Deribit DVOL if available (preferred over realized)
      - funding_rate_signal: perpetual funding rate
      - trend_regime: -1 to +1 directional signal
      - open_interest_change: OI change signal
    """

    name = "crypto_vol"
    version = "v1"

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        current_price = features.get("brti_price") or features.get("current_price")
        threshold = features.get("threshold_price")
        hours = features.get("hours_to_expiry")

        if current_price is None or threshold is None or hours is None:
            logger.debug(
                "crypto_vol.missing_inputs",
                has_price=current_price is not None,
                has_threshold=threshold is not None,
                has_hours=hours is not None,
            )
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        if current_price <= 0 or threshold <= 0 or hours <= 0:
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # Use implied vol if available, fall back to realized
        vol = features.get("implied_vol") or features.get("realized_vol_24h")
        if vol is None or vol <= 0:
            vol = features.get("realized_vol_7d")
        if vol is None or vol <= 0:
            # No vol data — can't price
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # Convert annualized vol to the time horizon
        t_years = hours / 8760.0  # hours in a year
        sigma_t = vol * math.sqrt(t_years)

        # Log-normal: P(S_T >= K) = N(d2) where
        # d2 = (ln(S/K) - 0.5*sigma^2*t) / (sigma*sqrt(t))
        if sigma_t <= 0.0001:
            # Essentially no time/vol — binary based on current vs threshold
            raw_prob = 1.0 if current_price >= threshold else 0.0
        else:
            d2 = (math.log(current_price / threshold) - 0.5 * vol**2 * t_years) / sigma_t
            raw_prob = _normal_cdf(d2)

        # Directional adjustments from market microstructure
        probability = raw_prob
        funding = features.get("funding_rate_signal", 0.0)
        trend = features.get("trend_regime", 0.0)
        oi_change = features.get("open_interest_change", 0.0)

        # Funding rate: positive = bullish bias, negative = bearish
        if threshold > current_price:
            # "Above" market — bullish signals increase probability
            probability += _clip(funding * 50.0, -0.02, 0.02)
            probability += _clip(trend * 0.02, -0.02, 0.02)
        else:
            # "Below" market — bearish signals increase probability
            probability -= _clip(funding * 50.0, -0.02, 0.02)
            probability -= _clip(trend * 0.02, -0.02, 0.02)

        # OI surge can indicate incoming volatility
        probability += _clip(abs(oi_change) * 0.01, 0.0, 0.01)

        probability = _clip(probability, 0.01, 0.99)

        # Confidence based on data quality
        data_points = sum(
            1 for k in (
                "implied_vol", "realized_vol_24h", "funding_rate_signal",
                "trend_regime", "open_interest_change",
            )
            if k in features and features[k] is not None
        )
        confidence = _clip(0.40 + 0.10 * data_points, 0.30, 0.85)

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
        pass  # No training needed — analytical model

    def save(self, path) -> None:
        pass

    @classmethod
    def load(cls, path) -> "CryptoVolModel":
        return cls()
