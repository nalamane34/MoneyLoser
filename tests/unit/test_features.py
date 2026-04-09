"""Tests for feature computation and the feature registry/pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from moneygone.exchange.types import (
    Market,
    MarketResult,
    MarketStatus,
    OrderbookLevel,
    OrderbookSnapshot,
)
from moneygone.features.base import Feature, FeatureContext
from moneygone.features.market_features import (
    BidAskSpread,
    OrderbookImbalance,
    TimeToExpiry,
)
from moneygone.features.pipeline import FeaturePipeline
from moneygone.features.registry import CyclicDependencyError, FeatureRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 14, 30, tzinfo=timezone.utc)
_CLOSE_TIME = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)


def _make_context(
    *,
    orderbook: OrderbookSnapshot | None = None,
    market: Market | None = None,
) -> FeatureContext:
    """Build a FeatureContext with sensible defaults."""
    if orderbook is None:
        orderbook = OrderbookSnapshot(
            ticker="TEST",
            yes_bids=(
                OrderbookLevel(price=Decimal("0.62"), contracts=Decimal("150")),
                OrderbookLevel(price=Decimal("0.60"), contracts=Decimal("200")),
            ),
            no_bids=(
                OrderbookLevel(price=Decimal("0.40"), contracts=Decimal("120")),
                OrderbookLevel(price=Decimal("0.42"), contracts=Decimal("180")),
            ),
            seq=1,
            timestamp=_NOW,
        )
    if market is None:
        market = Market(
            ticker="TEST",
            event_ticker="EVT-TEST",
            series_ticker="SER-TEST",
            title="Test market",
            status=MarketStatus.OPEN,
            yes_bid=Decimal("0.60"),
            yes_ask=Decimal("0.62"),
            last_price=Decimal("0.61"),
            volume=1000,
            open_interest=500,
            close_time=_CLOSE_TIME,
        )
    return FeatureContext(
        ticker="TEST",
        observation_time=_NOW,
        market_state=market,
        orderbook=orderbook,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBidAskSpread:
    """Test bid-ask spread feature computation."""

    def test_bid_ask_spread_computation(self) -> None:
        """Spread = yes_ask - yes_bid.

        yes_bid = 0.62 (best yes level price),
        yes_ask = 1 - 0.40 (best no level price) = 0.60.
        Wait -- the function _yes_bid_ask says:
          yes_bid = first yes level price = 0.62
          yes_ask = 1 - first no level price = 1 - 0.40 = 0.60
        So spread = 0.60 - 0.62 = -0.02.

        That's a crossed book (bid > ask), which produces a negative spread.
        Let's adjust the orderbook so bid < ask.
        """
        ob = OrderbookSnapshot(
            ticker="TEST",
            yes_bids=(
                OrderbookLevel(price=Decimal("0.58"), contracts=Decimal("150")),
            ),
            no_bids=(
                OrderbookLevel(price=Decimal("0.38"), contracts=Decimal("120")),
            ),
            seq=1,
            timestamp=_NOW,
        )
        ctx = _make_context(orderbook=ob)
        feature = BidAskSpread()
        # yes_bid = 0.58, yes_ask = 1 - 0.38 = 0.62
        # spread = 0.62 - 0.58 = 0.04
        result = feature.compute(ctx)

        assert result is not None
        assert result == pytest.approx(0.04, abs=1e-10)


class TestOrderbookImbalance:
    """Test orderbook imbalance feature computation."""

    def test_orderbook_imbalance_computation(self) -> None:
        """Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        ob = OrderbookSnapshot(
            ticker="TEST",
            yes_bids=(
                OrderbookLevel(price=Decimal("0.60"), contracts=Decimal("200")),
                OrderbookLevel(price=Decimal("0.58"), contracts=Decimal("100")),
            ),
            no_bids=(
                OrderbookLevel(price=Decimal("0.40"), contracts=Decimal("120")),
                OrderbookLevel(price=Decimal("0.42"), contracts=Decimal("80")),
            ),
            seq=1,
            timestamp=_NOW,
        )
        ctx = _make_context(orderbook=ob)
        feature = OrderbookImbalance(n_levels=5)
        result = feature.compute(ctx)

        # bid_vol = 200 + 100 = 300, ask_vol = 120 + 80 = 200
        # imbalance = (300 - 200) / (300 + 200) = 100/500 = 0.2
        assert result is not None
        assert result == pytest.approx(0.2, abs=1e-10)


class TestTimeToExpiry:
    """Test time to expiry feature."""

    def test_time_to_expiry(self) -> None:
        """Hours remaining should equal the delta between observation and close time."""
        ctx = _make_context()
        feature = TimeToExpiry()
        result = feature.compute(ctx)

        expected_hours = (_CLOSE_TIME - _NOW).total_seconds() / 3600.0
        assert result is not None
        assert result == pytest.approx(expected_hours, abs=0.01)
        assert result > 0


class TestFeaturePipeline:
    """Test the FeaturePipeline orchestration."""

    def test_feature_pipeline_returns_all_features(self) -> None:
        """Pipeline should compute all registered features and return them."""
        features = [BidAskSpread(), OrderbookImbalance(), TimeToExpiry()]
        pipeline = FeaturePipeline(features)

        ob = OrderbookSnapshot(
            ticker="TEST",
            yes_bids=(
                OrderbookLevel(price=Decimal("0.58"), contracts=Decimal("150")),
            ),
            no_bids=(
                OrderbookLevel(price=Decimal("0.38"), contracts=Decimal("120")),
            ),
            seq=1,
            timestamp=_NOW,
        )
        ctx = _make_context(orderbook=ob)
        result = pipeline.compute(ctx)

        assert "bid_ask_spread" in result
        assert "orderbook_imbalance" in result
        assert "time_to_expiry" in result
        assert len(result) == 3


class TestFeatureRegistryTopologicalSort:
    """Test the FeatureRegistry topological ordering."""

    def test_feature_registry_topological_sort(self) -> None:
        """Features with dependencies should be ordered correctly."""

        class FeatureA(Feature):
            name = "feature_a"
            dependencies = ()
            def compute(self, context: FeatureContext) -> float | None:
                return 1.0

        class FeatureB(Feature):
            name = "feature_b"
            dependencies = ("feature_a",)
            def compute(self, context: FeatureContext) -> float | None:
                return 2.0

        class FeatureC(Feature):
            name = "feature_c"
            dependencies = ("feature_a", "feature_b")
            def compute(self, context: FeatureContext) -> float | None:
                return 3.0

        registry = FeatureRegistry()
        registry.register(FeatureC())
        registry.register(FeatureA())
        registry.register(FeatureB())

        ordered = registry.resolve_order()
        names = [f.name for f in ordered]

        # A must come before B and C; B must come before C
        assert names.index("feature_a") < names.index("feature_b")
        assert names.index("feature_a") < names.index("feature_c")
        assert names.index("feature_b") < names.index("feature_c")

    def test_registry_detects_cycle(self) -> None:
        """Cyclic dependencies should raise CyclicDependencyError."""

        class FeatureX(Feature):
            name = "feature_x"
            dependencies = ("feature_y",)
            def compute(self, context: FeatureContext) -> float | None:
                return 1.0

        class FeatureY(Feature):
            name = "feature_y"
            dependencies = ("feature_x",)
            def compute(self, context: FeatureContext) -> float | None:
                return 2.0

        registry = FeatureRegistry()
        registry.register(FeatureX())
        registry.register(FeatureY())

        with pytest.raises(CyclicDependencyError):
            registry.resolve_order()
