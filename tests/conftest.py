"""Shared pytest fixtures for the MoneyGone test suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from moneygone.config import RiskConfig
from moneygone.data.store import DataStore
from moneygone.exchange.types import (
    Market,
    MarketResult,
    MarketStatus,
    OrderbookLevel,
    OrderbookSnapshot,
)
from moneygone.signals.fees import KalshiFeeCalculator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_orderbook() -> OrderbookSnapshot:
    """Load the sample orderbook fixture and return an OrderbookSnapshot."""
    raw = json.loads((FIXTURES_DIR / "sample_orderbook.json").read_text())
    yes_bids = tuple(
        OrderbookLevel(price=Decimal(lv["price"]), contracts=Decimal(lv["contracts"]))
        for lv in raw["yes_levels"]
    )
    no_bids = tuple(
        OrderbookLevel(price=Decimal(lv["price"]), contracts=Decimal(lv["contracts"]))
        for lv in raw["no_levels"]
    )
    return OrderbookSnapshot(
        ticker=raw["ticker"],
        yes_bids=yes_bids,
        no_bids=no_bids,
        seq=raw["seq"],
        timestamp=datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00")),
    )


@pytest.fixture()
def sample_markets() -> list[Market]:
    """Load the sample markets fixture and return a list of Market objects."""
    raw_list = json.loads((FIXTURES_DIR / "sample_markets.json").read_text())
    markets: list[Market] = []
    for raw in raw_list:
        markets.append(
            Market(
                ticker=raw["ticker"],
                event_ticker=raw["event_ticker"],
                series_ticker=raw["series_ticker"],
                title=raw["title"],
                status=MarketStatus(raw["status"]),
                yes_bid=Decimal(raw["yes_bid"]),
                yes_ask=Decimal(raw["yes_ask"]),
                last_price=Decimal(raw["last_price"]),
                volume=raw["volume"],
                open_interest=raw["open_interest"],
                close_time=datetime.fromisoformat(
                    raw["close_time"].replace("Z", "+00:00")
                ),
                result=MarketResult(raw["result"]) if raw["result"] else MarketResult.NOT_SETTLED,
                category=raw.get("category", ""),
            )
        )
    return markets


@pytest.fixture()
def sample_weather_ensemble() -> dict:
    """Load the sample weather ensemble fixture as a raw dict."""
    return json.loads((FIXTURES_DIR / "sample_weather_ensemble.json").read_text())


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_store():
    """Create an in-memory DuckDB DataStore, initialise schema, yield, close."""
    store = DataStore(":memory:")
    store.initialize_schema()
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Reusable calculators / configs
# ---------------------------------------------------------------------------


@pytest.fixture()
def fee_calculator() -> KalshiFeeCalculator:
    """Return a KalshiFeeCalculator instance."""
    return KalshiFeeCalculator()


@pytest.fixture()
def risk_config() -> RiskConfig:
    """Return a RiskConfig with test-appropriate defaults."""
    return RiskConfig(
        kelly_fraction=0.25,
        max_position_per_market=50,
        max_position_per_category_pct=0.25,
        max_total_exposure_pct=0.60,
        daily_loss_limit_pct=0.05,
        max_drawdown_pct=0.15,
        min_contract_price=0.05,
        max_contract_price=0.95,
        max_tail_exposure_pct=0.05,
    )
