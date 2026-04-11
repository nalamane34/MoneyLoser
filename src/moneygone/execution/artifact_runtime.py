"""Runtime adapters for artifact-backed all-market models.

These adapters let the live execution engine reuse the trained market-only
models stored under ``models/trained`` for categories that do not yet have
custom external-data providers.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog

from moneygone.data.market_discovery import MarketCategory
from moneygone.exchange.types import Market
from moneygone.utils.time import now_utc
from moneygone.features.base import FeatureContext
from moneygone.models.base import ModelPrediction, ProbabilityModel

logger = structlog.get_logger(__name__)

_THRESHOLD_RE = re.compile(r"-T([\d.]+)$")
_TITLE_THRESHOLD_RE = re.compile(r"\$?([\d,]+(?:\.\d+)?)")
_ABOVE_TERMS = ("greater", "greater_or_equal", "above", "over", "higher", "at least", "exceed")

_UNIVERSAL_CATEGORY_HINTS: dict[MarketCategory, float] = {
    MarketCategory.FINANCIALS: 1.0,
    MarketCategory.ECONOMICS: 9.0,
    MarketCategory.POLITICS: 11.0,
    MarketCategory.COMPANIES: 11.0,
    MarketCategory.UNKNOWN: 11.0,
}

_FINANCIAL_DAILY_PREFIXES = (
    "KXNATGASD",
    "KXBRENTD",
    "KXCOPPERD",
    "KXXRPD",
    "KXDOGED",
    "KXSILVERD",
    "KXGOLDD",
    "KXSTEELW",
)
_FINANCIAL_RANGE_PREFIXES = (
    "KXEURUSD",
    "KXUSDJPY",
    "KXWTIW",
    "KXXRP",
    "KXDOGE",
)

DEFAULT_UNIVERSAL_MODEL_CANDIDATES = (
    "trained/gbm_universal/model.pkl",
    "trained/logistic_universal/model.pkl",
)


class ArtifactBackedModel(ProbabilityModel):
    """Probability model wrapper around the joblib artifacts in ``models/trained``."""

    def __init__(
        self,
        *,
        artifact_path: Path,
        model_name: str,
        version: str | None = None,
    ) -> None:
        payload = joblib.load(artifact_path)
        self.name = model_name
        self.version = version or str(payload.get("trained_at", "artifact"))
        self.trained_at = _parse_datetime(payload.get("trained_at"))
        self._model = payload["model"]
        self._calibrator = payload.get("calibrator")
        self._scaler = payload.get("scaler")
        self._feature_names = list(payload.get("feature_names", ()))
        self._metrics = dict(payload.get("metrics", {}))

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def predict_proba(self, features: dict[str, float]) -> ModelPrediction:
        frame = pd.DataFrame(
            [[features.get(name, 0.0) for name in self._feature_names]],
            columns=self._feature_names,
        )
        X = frame.values
        if self._scaler is not None:
            X = self._scaler.transform(X)
        raw_prob = float(self._model.predict_proba(X)[0, 1])
        prob = raw_prob
        if self._calibrator is not None:
            calibrated = self._calibrator.predict(np.array([raw_prob]))
            prob = float(calibrated[0])
        confidence = min(1.0, max(0.0, 2.0 * abs(raw_prob - 0.5)))
        return ModelPrediction(
            probability=float(np.clip(prob, 0.0, 1.0)),
            raw_probability=raw_prob,
            confidence=confidence,
            model_name=self.name,
            model_version=self.version,
            features_used=dict(features),
            prediction_time=datetime.now(timezone.utc),
        )

    def predict_proba_batch(self, features: pd.DataFrame) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for _, row in features.iterrows():
            predictions.append(
                self.predict_proba(
                    {
                        key: float(value)
                        for key, value in row.to_dict().items()
                        if value is not None and not pd.isna(value)
                    }
                )
            )
        return predictions

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weights: pd.Series | None = None) -> None:
        raise NotImplementedError("Artifact-backed runtime models are read-only")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Artifact-backed runtime models are read-only")

    @classmethod
    def load(cls, path: Path) -> ProbabilityModel:
        return cls(artifact_path=path, model_name=path.parent.name)


class ArtifactFeaturePipeline:
    """Compute the market-only feature set expected by the saved artifacts."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        category_id: float | None = None,
    ) -> None:
        self._feature_names = list(feature_names)
        self._category_id = category_id

    def compute(self, context: FeatureContext) -> dict[str, float]:
        market = context.market_state
        if market is None:
            return {}

        last_price = float(market.last_price)
        yes_bid = float(market.yes_bid)
        yes_ask = float(market.yes_ask)
        spread = yes_ask - yes_bid
        mid_price = (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else last_price
        threshold = _extract_threshold(market)
        num_legs = float(len(market.mve_selected_legs))

        base = {
            "last_price": last_price,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "spread": spread,
            "log_volume": math.log1p(max(market.volume, 0)),
            "open_interest": float(market.open_interest),
            "hours_to_close": _hours_to_close(market, context.observation_time),
            "hour_of_day": float(market.close_time.hour),
            "day_of_week": float(market.close_time.weekday()),
            "mid_price": mid_price,
            "spread_pct": spread / max(mid_price, 0.01),
            "threshold": threshold,
            "is_above": _is_above_market(market),
            "num_legs": num_legs,
            "implied_single_prob": _implied_single_prob(last_price, num_legs),
            "log_liquidity": math.log1p(max(float(market.liquidity_dollars), 0.0)),
        }
        if self._category_id is not None:
            base["category_id"] = self._category_id
        return {name: float(base.get(name, 0.0)) for name in self._feature_names}


def load_default_artifact_model(model_dir: Path, candidates: list[str], *, model_name: str) -> ArtifactBackedModel | None:
    """Load the first available artifact model from a list of relative paths."""
    for relative in candidates:
        candidate = model_dir / relative
        if not candidate.exists():
            continue
        try:
            return ArtifactBackedModel(
                artifact_path=candidate,
                model_name=model_name,
            )
        except Exception:
            logger.warning("artifact_runtime.model_load_failed", path=str(candidate), exc_info=True)
    return None


def universal_category_id(category: MarketCategory) -> float:
    """Best-effort category hint for the universal artifact model."""
    return _UNIVERSAL_CATEGORY_HINTS.get(category, _UNIVERSAL_CATEGORY_HINTS[MarketCategory.UNKNOWN])


def fallback_categories_for_config(
    *,
    weather_enabled: bool,
    crypto_enabled: bool,
) -> list[MarketCategory]:
    """Return the categories eligible for artifact fallback under the current config.

    DISABLED: The universal artifact model has no real edge on politics,
    entertainment, economics, etc. — it outputs constant probabilities
    and burns cash.  Only specialist models (weather ensemble, sharp
    sportsbook) should trade.
    """
    return []


def build_universal_artifact_fallbacks(
    model_dir: Path,
    categories: list[MarketCategory],
    *,
    model_name: str = "market_universal",
) -> dict[MarketCategory, tuple[ArtifactBackedModel, ArtifactFeaturePipeline]]:
    """Build artifact-backed fallback model/pipeline pairs for market categories."""
    model = load_default_artifact_model(
        model_dir,
        list(DEFAULT_UNIVERSAL_MODEL_CANDIDATES),
        model_name=model_name,
    )
    if model is None:
        return {}

    return {
        category: (
            model,
            ArtifactFeaturePipeline(
                model.feature_names,
                category_id=universal_category_id(category),
            ),
        )
        for category in categories
    }


def is_financial_range_market(market: Market | None) -> bool:
    if market is None:
        return False
    return market.ticker.startswith(_FINANCIAL_RANGE_PREFIXES)


def is_financial_daily_market(market: Market | None) -> bool:
    if market is None:
        return False
    return market.ticker.startswith(_FINANCIAL_DAILY_PREFIXES)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hours_to_close(market: Market, observation_time: datetime) -> float:
    now = now_utc()
    delta = (market.close_time - now).total_seconds() / 3600.0
    return max(delta, 0.0)


def _extract_threshold(market: Market) -> float:
    if market.floor_strike is not None:
        return float(market.floor_strike)
    match = _THRESHOLD_RE.search(market.ticker)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    text = " ".join(filter(None, [market.title, market.subtitle, market.yes_sub_title]))
    for candidate in _TITLE_THRESHOLD_RE.findall(text):
        try:
            return float(candidate.replace(",", ""))
        except ValueError:
            continue
    return 0.0


def _is_above_market(market: Market) -> float:
    strike = market.strike_type.lower()
    if strike:
        return 1.0 if strike in ("greater", "greater_or_equal") else 0.0
    text = " ".join(filter(None, [market.title, market.subtitle, market.yes_sub_title])).lower()
    return 1.0 if any(term in text for term in _ABOVE_TERMS) else 0.0


def _implied_single_prob(last_price: float, num_legs: float) -> float:
    if num_legs <= 0:
        return 0.0
    if 0.0 < last_price < 1.0:
        return last_price ** (1.0 / num_legs)
    return last_price
