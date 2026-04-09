from __future__ import annotations

from decimal import Decimal

from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import (
    Action,
    MarketStatus,
    OrderRequest,
    Side,
    TimeInForce,
)


def test_parse_market_maps_active_status_to_open() -> None:
    market = KalshiRestClient._parse_market(
        {
            "ticker": "KXNBA-TEST",
            "event_ticker": "KXNBA",
            "series_ticker": "KXNBA",
            "title": "Test market",
            "status": "active",
            "yes_bid_dollars": "0.48",
            "yes_ask_dollars": "0.52",
            "last_price_dollars": "0.50",
            "volume": 10,
            "open_interest": 4,
            "close_time": "2026-04-09T18:00:00Z",
        }
    )

    assert market.status is MarketStatus.OPEN


def test_get_balance_parses_cent_denominated_demo_response(monkeypatch) -> None:
    client = KalshiRestClient.__new__(KalshiRestClient)

    async def _fake_request(method: str, path: str):
        assert method == "GET"
        assert path == "/portfolio/balance"
        return {
            "balance": 11007,
            "portfolio_value": 1460,
            "updated_ts": 1775755906,
        }

    client._request = _fake_request  # type: ignore[attr-defined]

    import asyncio

    balance = asyncio.run(KalshiRestClient.get_balance(client))

    assert str(balance.available) == "110.07"
    assert str(balance.total) == "124.67"


def test_get_all_markets_follows_cursor_pagination() -> None:
    client = KalshiRestClient.__new__(KalshiRestClient)
    calls: list[dict[str, object]] = []

    async def _fake_request(method: str, path: str, params=None):
        assert method == "GET"
        assert path == "/markets"
        calls.append(dict(params or {}))
        if not params or not params.get("cursor"):
            return {
                "markets": [
                    {
                        "ticker": "KXTEST-1",
                        "title": "Market 1",
                        "status": "open",
                        "yes_bid_dollars": "0.40",
                        "yes_ask_dollars": "0.60",
                        "last_price_dollars": "0.50",
                        "volume": 10,
                        "open_interest": 5,
                        "close_time": "2026-04-09T18:00:00Z",
                    }
                ],
                "cursor": "next-page",
            }
        return {
            "markets": [
                {
                    "ticker": "KXTEST-2",
                    "title": "Market 2",
                    "status": "open",
                    "yes_bid_dollars": "0.41",
                    "yes_ask_dollars": "0.61",
                    "last_price_dollars": "0.51",
                    "volume": 12,
                    "open_interest": 6,
                    "close_time": "2026-04-09T19:00:00Z",
                }
            ],
            "cursor": None,
        }

    client._request = _fake_request  # type: ignore[attr-defined]

    import asyncio

    markets = asyncio.run(KalshiRestClient.get_all_markets(client, status="open", limit=100))

    assert [market.ticker for market in markets] == ["KXTEST-1", "KXTEST-2"]
    assert calls == [
        {"status": "open", "limit": 100},
        {"status": "open", "limit": 100, "cursor": "next-page"},
    ]


def test_parse_market_keeps_market_subtitles() -> None:
    market = KalshiRestClient._parse_market(
        {
            "ticker": "KXNBAGAME-26APR10PHXLAL-LAL",
            "event_ticker": "KXNBAGAME-26APR10PHXLAL",
            "title": "Phoenix at Los Angeles L Winner?",
            "yes_sub_title": "Los Angeles L",
            "no_sub_title": "Phoenix",
            "status": "active",
            "yes_bid_dollars": "0.48",
            "yes_ask_dollars": "0.52",
            "last_price_dollars": "0.50",
            "volume": 10,
            "open_interest": 4,
            "close_time": "2026-04-09T18:00:00Z",
        }
    )

    assert market.yes_sub_title == "Los Angeles L"
    assert market.no_sub_title == "Phoenix"


def test_create_order_serializes_fixed_point_strings_and_current_tif() -> None:
    client = KalshiRestClient.__new__(KalshiRestClient)
    captured: dict[str, object] = {}

    async def _fake_request(method: str, path: str, json_body=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = dict(json_body or {})
        return {
            "order": {
                "order_id": "ord-1",
                "ticker": "KXTEST-1",
                "side": "yes",
                "action": "buy",
                "status": "pending",
                "count": 5,
                "remaining_count": 5,
                "yes_price_dollars": "0.5700",
                "taker_fees_dollars": "0.0000",
                "maker_fees_dollars": "0.0000",
                "created_time": "2026-04-09T18:00:00Z",
            }
        }

    client._request = _fake_request  # type: ignore[attr-defined]

    import asyncio

    asyncio.run(
        KalshiRestClient.create_order(
            client,
            OrderRequest(
                ticker="KXTEST-1",
                side=Side.YES,
                action=Action.BUY,
                count=5,
                yes_price=Decimal("0.57"),
                time_in_force=TimeInForce.IOC,
            ),
        )
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/portfolio/orders"
    assert captured["body"] == {
        "ticker": "KXTEST-1",
        "side": "yes",
        "action": "buy",
        "count": 5,
        "yes_price_dollars": "0.5700",
        "time_in_force": "immediate_or_cancel",
    }


def test_request_signs_path_without_query_parameters() -> None:
    client = KalshiRestClient.__new__(KalshiRestClient)

    class _FakeLimiter:
        async def acquire(self) -> None:
            return None

    class _FakeAuth:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_headers(self, method: str, path: str) -> dict[str, str]:
            self.calls.append((method, path))
            return {}

    class _FakeResponse:
        status_code = 200
        text = "{}"
        headers = {}

        @staticmethod
        def json() -> dict[str, object]:
            return {"markets": []}

    class _FakeClient:
        is_closed = False

        async def request(self, method: str, path: str, *, params=None, json=None, headers=None):
            return _FakeResponse()

    auth = _FakeAuth()
    client._config = None  # type: ignore[attr-defined]
    client._base_url = "https://demo-api.kalshi.co/trade-api/v2"
    client._auth = auth  # type: ignore[attr-defined]
    client._limiter = _FakeLimiter()  # type: ignore[attr-defined]
    client._client = _FakeClient()  # type: ignore[attr-defined]

    import asyncio

    asyncio.run(
        KalshiRestClient._request(
            client,
            "GET",
            "/markets",
            params={"status": "open", "limit": 100, "cursor": "abc"},
        )
    )

    assert auth.calls == [("GET", "/trade-api/v2/markets")]
