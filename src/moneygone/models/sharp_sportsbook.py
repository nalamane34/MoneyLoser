"""Conservative probability model driven by sharp sportsbook pricing."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SharpSportsbookModel(ProbabilityModel):
    """Use sharp sportsbook prices as the live probability anchor.

    This is intentionally conservative: it treats Pinnacle/consensus as the
    base probability and applies only small bounded adjustments from injuries,
    power ratings, and line movement.
    """

    name = "sharp_sportsbook"
    version = "v1"

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        base_prob = features.get("pinnacle_win_prob")
        no_sharp_data = False
        if base_prob is None:
            base_prob = features.get("sportsbook_win_prob")
        if base_prob is None:
            logger.warning(
                "sharp_model.no_sportsbook_data",
                features_available=list(features.keys()),
                msg="Falling back to 0.5 — prediction unreliable",
            )
            base_prob = 0.5
            no_sharp_data = True

        movement = features.get("moneyline_movement", 0.0)
        power = features.get("power_rating_edge", 0.0)
        injuries = features.get("team_injury_impact", 0.0)
        home = features.get("home_field_advantage", 0.0)

        probability = float(base_prob)
        probability += _clip(movement * 0.40, -0.03, 0.03)
        probability += _clip(power * 0.02, -0.02, 0.02)
        probability += _clip(injuries * 0.05, -0.03, 0.03)
        probability += _clip(home * 0.01, -0.01, 0.01)
        probability = _clip(probability, 0.01, 0.99)

        available = sum(
            1
            for key in (
                "pinnacle_win_prob",
                "sportsbook_win_prob",
                "moneyline_movement",
                "power_rating_edge",
                "team_injury_impact",
            )
            if key in features
        )
        confidence = _clip(0.50 + 0.08 * available, 0.50, 0.90)
        if no_sharp_data:
            confidence = 0.10  # Very low — this is essentially a coin flip

        return ModelPrediction(
            probability=probability,
            raw_probability=float(base_prob),
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
