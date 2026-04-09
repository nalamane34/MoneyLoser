"""Async REST client for the Kalshi Trade API v2.

All monetary values use the ``_dollars`` fields (latest API convention; the
legacy cent-denominated fields are deprecated).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
import structlog

from moneygone.config import ExchangeConfig
from moneygone.exchange.auth import KalshiAuth
from moneygone.exchange.errors import (
    AuthError,
    KalshiAPIError,
    OrderError,
    RateLimitError,
)
from moneygone.exchange.rate_limiter import AsyncRateLimiter
from moneygone.exchange.types import (
    Action,
    AmendOrderRequest,
    Balance,
    BatchOrderItem,
    BatchOrderResult,
    Candlestick,
    DailySchedule,
    ExchangeAnnouncement,
    ExchangeSchedule,
    Fill,
    MaintenanceWindow,
    Market,
    MarketResult,
    MarketStatus,
    Order,
    OrderGroup,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderRequest,
    OrderStatus,
    Position,
    QueuePosition,
    Series,
    Settlement,
    SettlementStatus,
    Side,
    Trade,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds
_RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(value: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    # Kalshi returns ISO-8601 with trailing Z or +00:00
    cleaned = value.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def _dec(value: Any) -> Decimal:
    """Safely convert a value to ``Decimal``."""
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def _market_result(value: str | None) -> MarketResult:
    """Convert a raw result string to a ``MarketResult`` enum."""
    if not value:
        return MarketResult.NOT_SETTLED
    try:
        return MarketResult(value)
    except ValueError:
        return MarketResult.NOT_SETTLED


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KalshiRestClient:
    """Async REST client for the Kalshi Trade API.

    Parameters:
        config: Exchange configuration (URLs, credentials, rate limit).
    """

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._auth = KalshiAuth(config.api_key_id, config.private_key_path)
        self._limiter = AsyncRateLimiter(config.rate_limit_rps)
        self._client: httpx.AsyncClient | None = None
        log.info(
            "rest_client.initialized",
            base_url=self._base_url,
            demo_mode=config.demo_mode,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                http2=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> KalshiRestClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Low-level request
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an authenticated, rate-limited request with retries."""
        client = await self._ensure_client()

        # Build the full path for signing (includes query string)
        sign_path = urlparse(self._base_url).path.rstrip("/") + "/" + path.lstrip("/")
        if params:
            # Filter out None values
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                sign_path = sign_path + "?" + urlencode(filtered, doseq=True)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            await self._limiter.acquire()

            headers = self._auth.get_headers(method.upper(), sign_path)

            try:
                response = await client.request(
                    method,
                    path,
                    params={k: v for k, v in (params or {}).items() if v is not None},
                    json=json_body,
                    headers=headers,
                )
            except httpx.TransportError as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2**attempt)
                log.warning(
                    "rest_client.transport_error",
                    attempt=attempt + 1,
                    error=str(exc),
                    retry_in=wait,
                )
                await asyncio.sleep(wait)
                continue

            # Map specific HTTP status codes to typed errors.
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", _BACKOFF_BASE * (2**attempt)))
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=retry_after,
                    status_code=429,
                    response_body=response.text,
                )

            if response.status_code in (401, 403):
                raise AuthError(
                    f"Authentication failed: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_exc = KalshiAPIError(
                    f"Server error {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
                wait = _BACKOFF_BASE * (2**attempt)
                log.warning(
                    "rest_client.server_error",
                    status=response.status_code,
                    attempt=attempt + 1,
                    retry_in=wait,
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code >= 400:
                raise KalshiAPIError(
                    f"API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            # 2xx success
            if response.status_code == 204:
                return {}
            return response.json()  # type: ignore[no-any-return]

        # Exhausted retries
        raise KalshiAPIError(
            f"Request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Deserialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_market(data: dict[str, Any]) -> Market:
        return Market(
            ticker=data["ticker"],
            event_ticker=data.get("event_ticker", ""),
            series_ticker=data.get("series_ticker", ""),
            title=data.get("title", ""),
            status=MarketStatus(data.get("status", "open")),
            yes_bid=_dec(data.get("yes_bid_dollars", data.get("yes_bid"))),
            yes_ask=_dec(data.get("yes_ask_dollars", data.get("yes_ask"))),
            last_price=_dec(data.get("last_price_dollars", data.get("last_price"))),
            volume=int(data.get("volume", 0)),
            open_interest=int(data.get("open_interest", 0)),
            close_time=_ts(data.get("close_time")),
            result=_market_result(data.get("result")),
            category=data.get("category", ""),
        )

    @staticmethod
    def _parse_order(data: dict[str, Any]) -> Order:
        return Order(
            order_id=data["order_id"],
            ticker=data["ticker"],
            side=Side(data["side"]),
            action=Action(data["action"]),
            status=OrderStatus(data.get("status", "pending")),
            count=int(data.get("count", 0)),
            remaining_count=int(data.get("remaining_count", 0)),
            price=_dec(data.get("yes_price_dollars", data.get("yes_price"))),
            taker_fees=_dec(data.get("taker_fees_dollars", data.get("taker_fees", 0))),
            maker_fees=_dec(data.get("maker_fees_dollars", data.get("maker_fees", 0))),
            created_time=_ts(data.get("created_time")),
        )

    @staticmethod
    def _parse_position(data: dict[str, Any]) -> Position:
        return Position(
            ticker=data["ticker"],
            event_ticker=data.get("event_ticker", ""),
            market_result=_market_result(data.get("market_result")),
            yes_count=int(data.get("yes_count", 0)),
            no_count=int(data.get("no_count", 0)),
            realized_pnl=_dec(data.get("realized_pnl_dollars", data.get("realized_pnl", 0))),
            settlement_status=SettlementStatus(data.get("settlement_status", "unsettled")),
        )

    @staticmethod
    def _parse_fill(data: dict[str, Any]) -> Fill:
        return Fill(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            side=Side(data["side"]),
            action=Action(data["action"]),
            count=int(data.get("count", 0)),
            price=_dec(data.get("yes_price_dollars", data.get("yes_price"))),
            is_taker=bool(data.get("is_taker", False)),
            created_time=_ts(data.get("created_time")),
        )

    @staticmethod
    def _parse_trade(data: dict[str, Any]) -> Trade:
        return Trade(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            count=int(data.get("count", 0)),
            yes_price=_dec(data.get("yes_price_dollars", data.get("yes_price"))),
            taker_side=Side(data.get("taker_side", "yes")),
            created_time=_ts(data.get("created_time")),
        )

    @staticmethod
    def _parse_settlement(data: dict[str, Any]) -> Settlement:
        return Settlement(
            ticker=data["ticker"],
            market_result=_market_result(data.get("market_result")),
            revenue=_dec(data.get("revenue_dollars", data.get("revenue", 0))),
            payout=_dec(data.get("payout_dollars", data.get("payout", 0))),
            settled_time=_ts(data.get("settled_time")),
        )

    @staticmethod
    def _parse_orderbook(data: dict[str, Any], ticker: str) -> OrderbookSnapshot:
        yes_levels = tuple(
            OrderbookLevel(price=_dec(lvl[0]), contracts=_dec(lvl[1]))
            for lvl in (data.get("yes") or [])
        )
        no_levels = tuple(
            OrderbookLevel(price=_dec(lvl[0]), contracts=_dec(lvl[1]))
            for lvl in (data.get("no") or [])
        )
        return OrderbookSnapshot(
            ticker=ticker,
            yes_levels=yes_levels,
            no_levels=no_levels,
            seq=int(data.get("seq", 0)),
            timestamp=_ts(data.get("timestamp")),
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_markets(self, **filters: Any) -> list[Market]:
        """Fetch a list of markets with optional filters.

        Supported filters include ``status``, ``event_ticker``,
        ``series_ticker``, ``tickers``, ``cursor``, ``limit``, etc.
        """
        data = await self._request("GET", "/markets", params=filters)
        return [self._parse_market(m) for m in data.get("markets", [])]

    async def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        return self._parse_market(data.get("market", data))

    async def get_orderbook(
        self, ticker: str, depth: int | None = None
    ) -> OrderbookSnapshot:
        """Fetch the orderbook for a single market."""
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        data = await self._request(
            "GET", f"/markets/{ticker}/orderbook", params=params
        )
        return self._parse_orderbook(data.get("orderbook", data), ticker)

    async def get_orderbooks(self, tickers: list[str]) -> list[OrderbookSnapshot]:
        """Fetch orderbooks for multiple markets in a single bulk request."""
        params: dict[str, Any] = {"tickers": ",".join(tickers)}
        data = await self._request("GET", "/markets/orderbooks", params=params)
        results: list[OrderbookSnapshot] = []
        for ticker_key, book_data in (data.get("orderbooks", {})).items():
            results.append(self._parse_orderbook(book_data, ticker_key))
        return results

    async def get_trades(
        self, ticker: str, limit: int | None = None
    ) -> list[Trade]:
        """Fetch public trades for a market."""
        params: dict[str, Any] = {"ticker": ticker}
        if limit is not None:
            params["limit"] = limit
        data = await self._request("GET", "/markets/trades", params=params)
        return [self._parse_trade(t) for t in data.get("trades", [])]

    async def get_events(self, **filters: Any) -> list[dict[str, Any]]:
        """Fetch events with optional filters.

        Returns raw dicts since event structure varies by category.
        """
        data = await self._request("GET", "/events", params=filters)
        return data.get("events", [])

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Fetch the account balance."""
        data = await self._request("GET", "/portfolio/balance")
        return Balance(
            available=_dec(data.get("available_balance_dollars", data.get("available_balance", 0))),
            total=_dec(data.get("total_balance_dollars", data.get("total_balance", 0))),
        )

    async def get_positions(self, **filters: Any) -> list[Position]:
        """Fetch portfolio positions with optional filters.

        Filters: ``ticker``, ``event_ticker``, ``settlement_status``,
        ``cursor``, ``limit``.
        """
        data = await self._request("GET", "/portfolio/positions", params=filters)
        return [
            self._parse_position(p)
            for p in data.get("market_positions", [])
        ]

    async def get_fills(self, **filters: Any) -> list[Fill]:
        """Fetch order fills (trade history).

        Filters: ``ticker``, ``order_id``, ``cursor``, ``limit``.
        """
        data = await self._request("GET", "/portfolio/fills", params=filters)
        return [self._parse_fill(f) for f in data.get("fills", [])]

    async def get_settlements(self, **filters: Any) -> list[Settlement]:
        """Fetch settlement records.

        Filters: ``ticker``, ``cursor``, ``limit``.
        """
        data = await self._request("GET", "/portfolio/settlements", params=filters)
        return [self._parse_settlement(s) for s in data.get("settlements", [])]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def create_order(self, request: OrderRequest) -> Order:
        """Submit a new order to the exchange."""
        body: dict[str, Any] = {
            "ticker": request.ticker,
            "side": request.side.value,
            "action": request.action.value,
            "count": request.count,
            "type": "limit",
            "yes_price_dollars": float(request.yes_price),
            "time_in_force": request.time_in_force.value,
        }
        if request.post_only:
            body["post_only"] = True
        if request.client_order_id is not None:
            body["client_order_id"] = request.client_order_id

        try:
            data = await self._request("POST", "/portfolio/orders", json_body=body)
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to create order: {exc}",
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        return self._parse_order(data.get("order", data))

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a single resting order."""
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to cancel order {order_id}: {exc}",
                order_id=order_id,
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        log.info("rest_client.order_cancelled", order_id=order_id)

    async def batch_cancel_orders(self, order_ids: list[str]) -> None:
        """Cancel multiple orders in a single request."""
        body: dict[str, Any] = {"order_ids": order_ids}
        try:
            await self._request(
                "DELETE", "/portfolio/orders", json_body=body
            )
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to batch cancel orders: {exc}",
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        log.info(
            "rest_client.orders_batch_cancelled", count=len(order_ids)
        )

    async def get_order(self, order_id: str) -> Order:
        """Fetch a single order by ID."""
        data = await self._request("GET", f"/portfolio/orders/{order_id}")
        return self._parse_order(data.get("order", data))

    async def get_orders(self, **filters: Any) -> list[Order]:
        """Fetch orders with optional filters.

        Filters: ``ticker``, ``event_ticker`` (comma-sep, max 10),
        ``min_ts``, ``max_ts``, ``status`` (resting|canceled|executed),
        ``limit`` (1-1000), ``cursor``, ``subaccount``.
        """
        data = await self._request("GET", "/portfolio/orders", params=filters)
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def amend_order(self, request: AmendOrderRequest) -> tuple[Order, Order]:
        """Amend a resting order's price and/or size.

        Returns ``(old_order, new_order)``.
        """
        body: dict[str, Any] = {
            "ticker": request.ticker,
            "side": request.side.value,
            "action": request.action.value,
        }
        if request.yes_price is not None:
            body["yes_price_dollars"] = float(request.yes_price)
        if request.count is not None:
            body["count"] = request.count

        try:
            data = await self._request(
                "POST", f"/portfolio/orders/{request.order_id}/amend",
                json_body=body,
            )
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to amend order {request.order_id}: {exc}",
                order_id=request.order_id,
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        old = self._parse_order(data.get("old_order", data))
        new = self._parse_order(data.get("order", data))
        log.info("rest_client.order_amended", order_id=request.order_id)
        return old, new

    async def decrease_order(
        self,
        order_id: str,
        *,
        reduce_by: int | None = None,
        reduce_to: int | None = None,
    ) -> Order:
        """Reduce a resting order's size without cancelling and re-submitting.

        Exactly one of ``reduce_by`` or ``reduce_to`` must be provided.
        """
        if reduce_by is None and reduce_to is None:
            raise ValueError("Exactly one of reduce_by or reduce_to is required")
        body: dict[str, Any] = {}
        if reduce_by is not None:
            body["reduce_by"] = reduce_by
        else:
            body["reduce_to"] = reduce_to

        try:
            data = await self._request(
                "POST", f"/portfolio/orders/{order_id}/decrease",
                json_body=body,
            )
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to decrease order {order_id}: {exc}",
                order_id=order_id,
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        return self._parse_order(data.get("order", data))

    async def batch_create_orders(
        self, orders: list[BatchOrderItem]
    ) -> list[BatchOrderResult]:
        """Submit multiple orders in a single atomic request.

        Returns one :class:`BatchOrderResult` per input order, each with
        either a filled ``order`` or an ``error`` message.
        """
        items: list[dict[str, Any]] = []
        for req in orders:
            item: dict[str, Any] = {
                "ticker": req.ticker,
                "side": req.side.value,
                "action": req.action.value,
                "count": req.count,
                "yes_price_dollars": float(req.yes_price),
                "time_in_force": req.time_in_force.value,
            }
            if req.post_only:
                item["post_only"] = True
            if req.client_order_id:
                item["client_order_id"] = req.client_order_id
            if req.order_group_id:
                item["order_group_id"] = req.order_group_id
            items.append(item)

        try:
            data = await self._request(
                "POST", "/portfolio/orders/batched",
                json_body={"orders": items},
            )
        except KalshiAPIError as exc:
            raise OrderError(
                f"Failed to batch create orders: {exc}",
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc

        results: list[BatchOrderResult] = []
        for entry in data.get("orders", []):
            order_data = entry.get("order")
            err = entry.get("error")
            results.append(BatchOrderResult(
                client_order_id=entry.get("client_order_id"),
                order=self._parse_order(order_data) if order_data else None,
                error=str(err) if err else None,
            ))
        log.info("rest_client.batch_orders_created", count=len(results))
        return results

    async def get_order_queue_position(self, order_id: str) -> Decimal:
        """Return the number of contracts ahead of this order in the queue."""
        data = await self._request(
            "GET", f"/portfolio/orders/{order_id}/queue_position"
        )
        return _dec(data.get("queue_position_fp", "0"))

    async def get_queue_positions(
        self,
        market_tickers: list[str] | None = None,
        event_ticker: str | None = None,
        subaccount: int = 0,
    ) -> list[QueuePosition]:
        """Fetch queue positions for all resting orders across given markets."""
        params: dict[str, Any] = {"subaccount": subaccount}
        if market_tickers:
            params["market_tickers"] = ",".join(market_tickers)
        if event_ticker:
            params["event_ticker"] = event_ticker

        data = await self._request(
            "GET", "/portfolio/orders/queue_positions", params=params
        )
        return [
            QueuePosition(
                order_id=q["order_id"],
                market_ticker=q["market_ticker"],
                queue_position=_dec(q.get("queue_position_fp", "0")),
            )
            for q in data.get("queue_positions", [])
        ]

    # ------------------------------------------------------------------
    # Order groups
    # ------------------------------------------------------------------

    async def create_order_group(self, contracts_limit: int) -> str:
        """Create an order group with a contract limit.

        Returns the ``order_group_id``.  Attach orders to this group via
        ``order_group_id`` on each :class:`BatchOrderItem` or
        :class:`OrderRequest`.  When the group's cumulative fill reaches
        ``contracts_limit``, remaining orders in the group are cancelled.
        """
        data = await self._request(
            "POST", "/portfolio/order_groups/create",
            json_body={"contracts_limit": contracts_limit},
        )
        gid = data.get("order_group_id", "")
        log.info("rest_client.order_group_created", order_group_id=gid)
        return gid

    async def get_order_groups(self, subaccount: int | None = None) -> list[OrderGroup]:
        """Fetch all order groups for the account."""
        params: dict[str, Any] = {}
        if subaccount is not None:
            params["subaccount"] = subaccount
        data = await self._request("GET", "/portfolio/order_groups", params=params)
        return [
            OrderGroup(
                order_group_id=g["id"],
                contracts_limit=_dec(g.get("contracts_limit_fp", "0")),
                is_auto_cancel_enabled=bool(g.get("is_auto_cancel_enabled", False)),
            )
            for g in data.get("order_groups", [])
        ]

    async def trigger_order_group(self, order_group_id: str) -> None:
        """Manually trigger an order group to release its orders."""
        await self._request(
            "PUT", f"/portfolio/order_groups/{order_group_id}/trigger"
        )
        log.info("rest_client.order_group_triggered", order_group_id=order_group_id)

    async def delete_order_group(self, order_group_id: str) -> None:
        """Delete an order group (cancels all associated resting orders)."""
        await self._request(
            "DELETE", f"/portfolio/order_groups/{order_group_id}"
        )
        log.info("rest_client.order_group_deleted", order_group_id=order_group_id)

    async def reset_order_group(self, order_group_id: str) -> None:
        """Reset an order group's fill counter back to zero."""
        await self._request(
            "PUT", f"/portfolio/order_groups/{order_group_id}/reset"
        )
        log.info("rest_client.order_group_reset", order_group_id=order_group_id)

    async def update_order_group_limit(
        self, order_group_id: str, contracts_limit: int
    ) -> None:
        """Update the contract limit on an existing order group."""
        await self._request(
            "PUT", f"/portfolio/order_groups/{order_group_id}/limit",
            json_body={"contracts_limit": contracts_limit},
        )
        log.info(
            "rest_client.order_group_limit_updated",
            order_group_id=order_group_id,
            contracts_limit=contracts_limit,
        )

    # ------------------------------------------------------------------
    # Portfolio extras
    # ------------------------------------------------------------------

    async def get_total_resting_order_value(self) -> Decimal:
        """Total dollar value of all resting limit orders.

        Useful as a pre-trade risk check to ensure we don't over-commit
        buying power before orders fill.
        """
        data = await self._request(
            "GET", "/portfolio/summary/total_resting_order_value"
        )
        # API returns cents integer
        cents = int(data.get("total_resting_order_value", 0))
        return Decimal(cents) / 100

    # ------------------------------------------------------------------
    # Exchange info
    # ------------------------------------------------------------------

    async def get_exchange_schedule(self) -> ExchangeSchedule:
        """Fetch the exchange's trading schedule and maintenance windows."""
        data = await self._request("GET", "/exchange/schedule")
        sched = data.get("schedule", {})

        std_hours = [
            DailySchedule(
                open_time=h.get("open_time", ""),
                close_time=h.get("close_time", ""),
            )
            for h in sched.get("standard_hours", [])
        ]
        maintenance = [
            MaintenanceWindow(
                start_datetime=_ts(w.get("start_datetime")),
                end_datetime=_ts(w.get("end_datetime")),
            )
            for w in sched.get("maintenance_windows", [])
        ]
        return ExchangeSchedule(
            standard_hours=std_hours,
            maintenance_windows=maintenance,
        )

    async def get_exchange_announcements(self) -> list[ExchangeAnnouncement]:
        """Fetch active exchange announcements (maintenance, outages, etc.)."""
        data = await self._request("GET", "/exchange/announcements")
        return [
            ExchangeAnnouncement(
                type=a.get("type", "info"),
                message=a.get("message", ""),
                delivery_time=_ts(a.get("delivery_time")),
                status=a.get("status", "active"),
            )
            for a in data.get("announcements", [])
        ]

    # ------------------------------------------------------------------
    # Series & events
    # ------------------------------------------------------------------

    async def get_series(self, series_ticker: str) -> Series:
        """Fetch metadata for a single series."""
        data = await self._request("GET", f"/series/{series_ticker}")
        s = data.get("series", data)
        return Series(
            ticker=s.get("ticker", series_ticker),
            title=s.get("title", ""),
            category=s.get("category", ""),
            tags=s.get("tags", []),
            frequency=s.get("frequency", ""),
            settlement_sources=s.get("settlement_sources", []),
        )

    async def get_series_list(self, **filters: Any) -> list[Series]:
        """Fetch all series with optional filters (``category``, ``status``, etc.)."""
        data = await self._request("GET", "/series", params=filters)
        return [
            Series(
                ticker=s.get("ticker", ""),
                title=s.get("title", ""),
                category=s.get("category", ""),
                tags=s.get("tags", []),
                frequency=s.get("frequency", ""),
                settlement_sources=s.get("settlement_sources", []),
            )
            for s in data.get("series", [])
        ]

    async def get_event(
        self, event_ticker: str, with_nested_markets: bool = False
    ) -> dict[str, Any]:
        """Fetch a single event and optionally its nested markets.

        Returns the raw event dict — structure varies by event category.
        """
        data = await self._request(
            "GET", f"/events/{event_ticker}",
            params={"with_nested_markets": with_nested_markets},
        )
        return data.get("event", data)

    async def get_event_forecast_history(
        self,
        series_ticker: str,
        event_ticker: str,
        percentiles: list[int],
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetch forecast percentile history for an event.

        Useful for calibration: compares historical Kalshi forecasts
        against eventual outcomes.

        Parameters
        ----------
        percentiles:
            List of percentile values (0-10000, up to 10 values).
        start_ts / end_ts:
            Unix timestamps for the query window.
        period_interval:
            Candle size in minutes: 0 (raw), 1, 60, or 1440.
        """
        params: dict[str, Any] = {
            "percentiles": percentiles,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        data = await self._request(
            "GET",
            f"/series/{series_ticker}/events/{event_ticker}/forecast_percentile_history",
            params=params,
        )
        return data.get("forecast_history", [])

    # ------------------------------------------------------------------
    # Market candlesticks
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_candlestick(c: dict[str, Any]) -> Candlestick:
        price = c.get("price", {})
        return Candlestick(
            end_period_ts=_ts(str(c.get("end_period_ts", ""))),
            open=_dec(price.get("open", c.get("yes_ask", {}).get("open", 0))),
            high=_dec(price.get("high", c.get("yes_ask", {}).get("high", 0))),
            low=_dec(price.get("low", c.get("yes_ask", {}).get("low", 0))),
            close=_dec(price.get("close", c.get("yes_ask", {}).get("close", 0))),
            volume=int(c.get("volume", 0)),
            open_interest=int(c.get("open_interest", 0)),
        )

    async def get_market_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[Candlestick]:
        """Fetch OHLC candlesticks for a live market.

        Parameters
        ----------
        ticker:
            Market ticker.
        start_ts / end_ts:
            Unix timestamps.
        period_interval:
            Candle size in minutes: 1, 60, or 1440.
        """
        data = await self._request(
            "GET", f"/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return [self._parse_candlestick(c) for c in data.get("candlesticks", [])]

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    async def get_historical_cutoff(self) -> dict[str, str]:
        """Return the cutoff timestamps separating live vs historical data.

        Keys: ``market_settled_ts``, ``trades_created_ts``,
        ``orders_updated_ts``.
        """
        return await self._request("GET", "/historical/cutoff")

    async def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[Candlestick]:
        """Fetch OHLC candlesticks for a settled (historical) market.

        Parameters
        ----------
        ticker:
            Market ticker.
        start_ts / end_ts:
            Unix timestamps (candlestick end times).
        period_interval:
            Candle size in minutes: 1, 60, or 1440.
        """
        data = await self._request(
            "GET", f"/historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return [self._parse_candlestick(c) for c in data.get("candlesticks", [])]

    async def get_historical_trades(self, **filters: Any) -> list[Trade]:
        """Fetch historical (settled) trades.

        Filters: ``ticker``, ``min_ts``, ``max_ts``, ``limit``, ``cursor``.
        """
        data = await self._request("GET", "/historical/trades", params=filters)
        return [self._parse_trade(t) for t in data.get("trades", [])]

    async def get_historical_fills(self, **filters: Any) -> list[Fill]:
        """Fetch historical fills (your personal fill history).

        Filters: ``ticker``, ``max_ts``, ``limit``, ``cursor``.
        """
        data = await self._request("GET", "/historical/fills", params=filters)
        return [self._parse_fill(f) for f in data.get("fills", [])]

    async def get_historical_orders(self, **filters: Any) -> list[Order]:
        """Fetch historical (cancelled/executed) orders.

        Filters: ``ticker``, ``max_ts``, ``limit``, ``cursor``.
        """
        data = await self._request("GET", "/historical/orders", params=filters)
        return [self._parse_order(o) for o in data.get("orders", [])]

    # ------------------------------------------------------------------
    # Search / tags
    # ------------------------------------------------------------------

    async def get_tags_by_category(self) -> dict[str, list[str]]:
        """Return a mapping of series categories to their associated tags.

        Useful for market discovery — find all tickers relevant to a
        category (e.g. "Sports", "Crypto", "Economics").
        """
        data = await self._request("GET", "/search/tags_by_categories")
        return data.get("categories", data)
