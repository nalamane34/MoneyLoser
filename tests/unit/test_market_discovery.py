from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from moneygone.data.market_discovery import (
    MarketCategory,
    MarketDiscoveryService,
    _dict_to_market,
    _market_to_dict,
    classify_market,
)
from moneygone.exchange.types import Market, MarketResult, MarketStatus


def _market(
    title: str,
    *,
    ticker: str = "KXTEST-1",
    event_ticker: str = "KXTEST",
    series_ticker: str = "KXTEST",
    category: str = "",
    status: MarketStatus = MarketStatus.OPEN,
) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=title,
        subtitle="",
        yes_sub_title="",
        no_sub_title="",
        status=status,
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.60"),
        last_price=Decimal("0.50"),
        volume=100,
        open_interest=10,
        close_time=datetime.now(timezone.utc),
        result=MarketResult.NOT_SETTLED,
        category=category,
    )


def test_classify_market_prefers_explicit_api_category() -> None:
    market = _market(
        "Will the election winner be decided on Tuesday?",
        category="politics",
    )

    assert classify_market(market) is MarketCategory.POLITICS


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Will CPI inflation fall below 3% by June?", MarketCategory.ECONOMICS),
        ("Will the election winner be decided on Tuesday?", MarketCategory.POLITICS),
        ("Wisconsin Supreme Court winner?", MarketCategory.POLITICS),
        ("Will Donald Trump be at the White House on Jan 20?", MarketCategory.POLITICS),
        ("Will OpenAI revenue be at least $10B in 2027?", MarketCategory.COMPANIES),
        ("Will the Oscar winner be announced at the ceremony?", MarketCategory.ENTERTAINMENT),
        ("Best Picture winner?", MarketCategory.UNKNOWN),
        ("Phoenix vs Los Angeles L NBA Winner?", MarketCategory.SPORTS),
    ],
)
def test_classify_market_avoids_false_sports_matches(
    title: str,
    expected: MarketCategory,
) -> None:
    assert classify_market(_market(title, ticker="KXGENERIC-1")) is expected


def test_classify_market_company_markets_beat_financials() -> None:
    market = _market("Will OpenAI market cap exceed 300B USD by 2027?")

    assert classify_market(market) is MarketCategory.COMPANIES


@pytest.mark.asyncio
async def test_discovery_refresh_fetches_all_open_markets_with_single_bulk_query(
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeRestClient:
        async def get_all_markets(self, **filters):
            calls.append(dict(filters))
            return [
                _market("Bitcoin above 100k?", ticker="KXBTC-1", category="crypto"),
                _market(
                    "Closed market",
                    ticker="KXCLOSED-1",
                    status=MarketStatus.CLOSED,
                    category="politics",
                ),
            ]

    discovery = MarketDiscoveryService(
        rest_client=_FakeRestClient(),  # type: ignore[arg-type]
        cache_path=tmp_path / "markets.json",
    )

    classified = await discovery.refresh()

    assert calls == [{"limit": 1_000, "max_pages": 0, "status": "open", "mve_filter": "exclude"}]
    assert [(market.ticker, category) for market, category in classified] == [
        ("KXBTC-1", MarketCategory.CRYPTO),
    ]


def test_market_discovery_cache_roundtrip_preserves_runtime_fields() -> None:
    market = Market(
        ticker="KXBTC-T42.5",
        event_ticker="KXBTC",
        series_ticker="KXBTC",
        title="BTC above 42.5?",
        subtitle="",
        yes_sub_title="",
        no_sub_title="",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.48"),
        yes_ask=Decimal("0.52"),
        last_price=Decimal("0.51"),
        volume=100,
        open_interest=10,
        close_time=datetime.now(timezone.utc),
        result=MarketResult.NOT_SETTLED,
        category="financials",
        created_time=datetime(2026, 4, 10, 12, tzinfo=timezone.utc),
        open_time=datetime(2026, 4, 10, 13, tzinfo=timezone.utc),
        previous_price=Decimal("0.49"),
        liquidity_dollars=Decimal("250.0"),
        strike_type="greater",
        floor_strike=Decimal("42.5"),
        cap_strike=Decimal("45.0"),
        mve_selected_legs=({"ticker": "LEG1"}, {"ticker": "LEG2"}),
    )
    encoded = _market_to_dict(market, MarketCategory.FINANCIALS)
    decoded, category = _dict_to_market(encoded)

    assert category is MarketCategory.FINANCIALS
    assert decoded.created_time == market.created_time
    assert decoded.open_time == market.open_time
    assert decoded.previous_price == market.previous_price
    assert decoded.liquidity_dollars == market.liquidity_dollars
    assert decoded.floor_strike == market.floor_strike
    assert decoded.cap_strike == market.cap_strike
    assert decoded.mve_selected_legs == market.mve_selected_legs
