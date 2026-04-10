from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import joblib
import numpy as np
import pytest

from moneygone.data.market_discovery import MarketCategory
from moneygone.exchange.types import Market, MarketResult, MarketStatus
from moneygone.execution.artifact_runtime import (
    ArtifactBackedModel,
    ArtifactFeaturePipeline,
    universal_category_id,
)
from moneygone.features.base import FeatureContext


class _FakeModel:
    def predict_proba(self, X):
        return np.array([[0.30, 0.70] for _ in range(len(X))], dtype=float)


class _FakeCalibrator:
    def predict(self, values):
        arr = np.asarray(values, dtype=float)
        return np.clip(arr - 0.05, 0.0, 1.0)


class _FakeScaler:
    def transform(self, X):
        return X


def _market(**overrides) -> Market:
    now = datetime.now(timezone.utc)
    payload = dict(
        ticker="KXTEST-T42.5",
        event_ticker="KXTEST",
        series_ticker="KXTEST",
        title="Will BTC close above 42.5?",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.48"),
        yes_ask=Decimal("0.52"),
        last_price=Decimal("0.51"),
        volume=125,
        open_interest=33,
        close_time=now + timedelta(hours=5),
        result=MarketResult.NOT_SETTLED,
        category="financials",
        created_time=now - timedelta(hours=2),
        liquidity_dollars=Decimal("250.0"),
        strike_type="greater",
        floor_strike=Decimal("42.5"),
        mve_selected_legs=({"ticker": "LEG1"}, {"ticker": "LEG2"}),
    )
    payload.update(overrides)
    return Market(**payload)


def test_artifact_feature_pipeline_matches_runtime_schema() -> None:
    market = _market()
    pipeline = ArtifactFeaturePipeline(
        [
            "last_price",
            "yes_bid",
            "yes_ask",
            "spread",
            "log_volume",
            "open_interest",
            "hours_to_close",
            "hour_of_day",
            "day_of_week",
            "mid_price",
            "spread_pct",
            "threshold",
            "is_above",
            "num_legs",
            "implied_single_prob",
            "log_liquidity",
            "category_id",
        ],
        category_id=universal_category_id(MarketCategory.FINANCIALS),
    )

    features = pipeline.compute(
        FeatureContext(
            ticker=market.ticker,
            observation_time=datetime.now(timezone.utc),
            market_state=market,
        )
    )

    assert features["threshold"] == 42.5
    assert features["is_above"] == 1.0
    assert features["num_legs"] == 2.0
    assert features["category_id"] == universal_category_id(MarketCategory.FINANCIALS)
    assert features["hours_to_close"] >= 6.9
    assert 0.0 < features["implied_single_prob"] < 1.0


def test_artifact_backed_model_loads_joblib_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": _FakeModel(),
            "calibrator": _FakeCalibrator(),
            "scaler": _FakeScaler(),
            "feature_names": ["last_price", "threshold"],
            "trained_at": "2026-04-10T12:00:00+00:00",
            "metrics": {"brier": 0.1},
        },
        artifact_path,
    )

    model = ArtifactBackedModel(
        artifact_path=artifact_path,
        model_name="market_universal",
    )
    prediction = model.predict_proba({"last_price": 0.51, "threshold": 42.5})

    assert prediction.raw_probability == pytest.approx(0.70)
    assert prediction.probability == pytest.approx(0.65)
    assert prediction.model_name == "market_universal"
