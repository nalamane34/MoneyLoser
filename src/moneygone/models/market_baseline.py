"""Market-implied baseline probability model.

Uses orderbook microstructure features to produce a probability estimate
that's anchored on the market's own pricing but adjusted for orderbook
imbalances. This enables evaluation of ANY market category — politics,
economics, financials, companies — without needing specialized data feeds.

The model essentially asks: "Does the orderbook structure suggest the
market price is mispriced?" Large imbalances, unusual spreads, and
volume-weighted mid deviations can signal short-term mispricings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MarketBaselineModel(ProbabilityModel):
    """Baseline model using orderbook microstructure signals.

    This model derives probability from the market's own price (mid_price)
    and adjusts based on:
      - orderbook_imbalance: bid/ask volume asymmetry
      - depth_ratio: relative depth of bids vs asks
      - weighted_mid_price: volume-weighted fair value
      - bid_ask_spread: wider spread = more uncertainty

    Expected features (all from market_features.py):
      - mid_price: midpoint of best bid and ask
      - weighted_mid_price: volume-weighted mid
      - orderbook_imbalance: (bid_vol - ask_vol) / total
      - depth_ratio: bid_depth / ask_depth
      - bid_ask_spread: ask - bid
      - time_to_expiry: hours until close
    """

    name = "market_baseline"
    version = "v1"

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        mid = features.get("mid_price")
        if mid is None:
            return ModelPrediction(
                probability=0.5,
                raw_probability=0.5,
                confidence=0.0,
                model_name=self.name,
                model_version=self.version,
                features_used=dict(features),
                prediction_time=datetime.now(timezone.utc),
            )

        # Start with market-implied probability (mid price IS the probability)
        raw_prob = _clip(mid, 0.01, 0.99)

        # Adjust based on orderbook imbalance
        # Positive imbalance (more bids) → price likely to go up → higher prob
        imbalance = features.get("orderbook_imbalance", 0.0)
        imbalance_adj = _clip(imbalance * 0.03, -0.05, 0.05)

        # Volume-weighted mid can reveal where the "smart money" is
        wmid = features.get("weighted_mid_price")
        wmid_adj = 0.0
        if wmid is not None and abs(wmid - mid) > 0.005:
            # Weighted mid deviates from simple mid → adjust toward it
            wmid_adj = _clip((wmid - mid) * 0.5, -0.03, 0.03)

        # Depth ratio: strong bid support suggests price is more likely YES
        depth = features.get("depth_ratio", 1.0)
        if depth > 0:
            import math
            depth_adj = _clip(math.log(depth) * 0.01, -0.02, 0.02)
        else:
            depth_adj = 0.0

        probability = raw_prob + imbalance_adj + wmid_adj + depth_adj
        probability = _clip(probability, 0.03, 0.97)

        # Confidence: based on spread and time to expiry
        spread = features.get("bid_ask_spread", 0.05)
        hours = features.get("time_to_expiry", 24.0)

        # Tight spread + near expiry = higher confidence
        base_confidence = 0.50
        if spread < 0.02:
            base_confidence += 0.10
        elif spread > 0.10:
            base_confidence -= 0.10

        if hours < 6:
            base_confidence += 0.05
        elif hours > 168:  # > 1 week
            base_confidence -= 0.10

        confidence = _clip(base_confidence, 0.30, 0.70)

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
    def load(cls, path) -> "MarketBaselineModel":
        return cls()
