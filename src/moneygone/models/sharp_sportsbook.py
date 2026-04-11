"""Conservative probability model driven by sharp sportsbook pricing.

v2 changes (from v1):
  - New features: spread_implied_win_prob, kalshi_vs_sportsbook_edge
  - Data quality features modulate confidence: line_age_hours,
    is_consensus_fallback, match_quality
  - Tighter probability clip [0.05, 0.95] — never near-certain on sports
  - Confidence based on data quality, not feature count
  - Optional LightGBM residual model overlay when trained model exists
  - Consensus fallback penalty (wider uncertainty)
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SharpSportsbookModel(ProbabilityModel):
    """Use sharp sportsbook prices as the live probability anchor.

    v2: Data-quality-aware confidence, new feature channels, optional
    residual model overlay.
    """

    name = "sharp_sportsbook"
    version = "v2"

    def __init__(self, residual_model_dir: Path | None = None) -> None:
        self._lgb_model: Any = None
        self._lgb_feature_cols: list[str] = []
        if residual_model_dir is not None:
            self._try_load_residual(residual_model_dir)

    def _try_load_residual(self, model_dir: Path) -> None:
        """Load trained LightGBM residual model if available."""
        model_path = model_dir / "sports_residual.lgb"
        meta_path = model_dir / "sports_residual_meta.json"
        if not model_path.exists() or not meta_path.exists():
            return
        try:
            import lightgbm as lgb
            self._lgb_model = lgb.Booster(model_file=str(model_path))
            with open(meta_path) as f:
                meta = json.load(f)
            self._lgb_feature_cols = meta.get("feature_cols", [])
            logger.info(
                "sharp_model.residual_loaded",
                path=str(model_path),
                n_features=len(self._lgb_feature_cols),
                brier_improvement=meta.get("brier_improvement"),
            )
        except Exception:
            logger.warning("sharp_model.residual_load_failed", exc_info=True)
            self._lgb_model = None

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        base_prob = features.get("pinnacle_win_prob")
        using_consensus = False
        if base_prob is None:
            base_prob = features.get("sportsbook_win_prob")
            using_consensus = True
        if base_prob is None:
            # No sportsbook anchor — return uninvestable prediction
            logger.warning(
                "sharp_model.no_sportsbook_data",
                features_available=list(features.keys()),
                msg="No sharp line available — refusing to predict",
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

        probability = float(base_prob)
        raw_prob = probability

        # --- Try residual model first ---
        if self._lgb_model is not None and self._lgb_feature_cols:
            try:
                import numpy as np
                x = np.array([[features.get(c, 0.0) for c in self._lgb_feature_cols]])
                residual = float(self._lgb_model.predict(x)[0])
                # Cap residual adjustment to ±8% — model shouldn't override
                # sharp books by more than that
                residual = _clip(residual, -0.08, 0.08)
                probability += residual
            except Exception:
                logger.debug("sharp_model.residual_predict_failed", exc_info=True)

        # --- Heuristic adjustments (when no residual model) ---
        if self._lgb_model is None:
            movement = features.get("moneyline_movement", 0.0)
            power = features.get("power_rating_edge", 0.0)
            injuries = features.get("team_injury_impact", 0.0)
            home = features.get("home_field_advantage", 0.0)

            # Line movement is the strongest heuristic signal
            probability += _clip(movement * 0.40, -0.03, 0.03)
            # Power rating: small nudge
            probability += _clip(power * 0.02, -0.02, 0.02)
            # Injury impact: bounded
            probability += _clip(injuries * 0.05, -0.03, 0.03)
            # Home field: minimal
            probability += _clip(home * 0.01, -0.01, 0.01)

            # Spread-implied cross-check (#4): if spread data available,
            # nudge toward spread-implied probability
            spread_prob = features.get("spread_implied_win_prob")
            if spread_prob is not None:
                spread_diff = spread_prob - probability
                # Very small nudge — spreads and moneylines should agree
                probability += _clip(spread_diff * 0.08, -0.02, 0.02)

        # Final clip: never allow near-certainty on sports
        probability = _clip(probability, 0.05, 0.95)

        # --- Confidence: based on data quality, not feature count ---
        # Start with base confidence
        confidence = 0.65

        # Pinnacle vs consensus: Pinnacle is sharper
        if using_consensus or features.get("is_consensus_fallback", 0.0) > 0.5:
            confidence -= 0.10  # Consensus is noisier

        # Line staleness (#9): stale lines are less reliable
        line_age = features.get("line_age_hours", 0.0)
        if line_age > 12.0:
            confidence -= 0.10  # Very stale
        elif line_age > 4.0:
            confidence -= 0.05  # Somewhat stale
        elif line_age <= 0.0:
            confidence -= 0.05  # Unknown age

        # Match quality (#9): how well did we map Kalshi → sportsbook event?
        match_quality = features.get("match_quality", 1.0)
        if match_quality < 0.75:
            confidence -= 0.15  # Poor match — might be wrong event
        elif match_quality < 1.0:
            confidence -= 0.05

        # Bonus for having multiple confirming signals
        has_movement = abs(features.get("moneyline_movement", 0.0)) > 0.01
        has_ratings = "power_rating_edge" in features
        has_spread = "spread_implied_win_prob" in features
        confirming = sum([has_movement, has_ratings, has_spread])
        confidence += confirming * 0.03

        # Residual model loaded: extra confidence from ML
        if self._lgb_model is not None:
            confidence += 0.05

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
                {
                    key: float(value)
                    for key, value in row.items()
                    if pd.notna(value)
                }
            )
            for _, row in features.iterrows()
        ]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> None:
        return None

    def save(self, path: Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"name": self.name, "version": self.version}, handle)

    @classmethod
    def load(cls, path: Path) -> ProbabilityModel:
        if path.exists():
            with open(path, "rb") as handle:
                pickle.load(handle)  # noqa: S301
        return cls()
