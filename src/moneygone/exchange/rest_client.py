"""Async REST client for the Kalshi Trade API v2.

All monetary values use the ``_dollars`` fields (latest API convention; the
legacy cent-denominated fields are deprecated).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

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
    AccountLimits,
    Action,
    AmendOrderRequest,
    Balance,
    BatchOrderItem,
    BatchOrderResult,
    Candlestick,
    CandlestickOHLC,
    DailySchedule,
    ExchangeAnnouncement,
    ExchangeSchedule,
    ExchangeStatus,
    Fill,
    MaintenanceWindow,
    Market,
    MarketCandlesticks,
    MarketResult,
    MarketStatus,
    Milestone,
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
    StructuredTarget,
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


def _money(
    dollars_value: Any | None = None,
    cents_value: Any | None = None,
) -> Decimal:
    """Parse a money field from either dollar or cent denominations."""
    if dollars_value is not None:
        return _dec(dollars_value)
    if cents_value is not None:
        return _dec(cents_value) / Decimal("100")
    return Decimal(0)


def _fixed_point_dollars(value: Any) -> str:
    """Serialize a money value in Kalshi's fixed-point dollar string format."""
    return format(_dec(value).quantize(Decimal("0.0001")), "f")


def _market_result(value: str | None) -> MarketResult:
    """Convert a raw result string to a ``MarketResult`` enum."""
    if not value:
        return MarketResult.NOT_SETTLED
    try:
        return MarketResult(value)
    except ValueError:
        return MarketResult.NOT_SETTLED


def _market_status(value: str | None) -> MarketStatus:
    """Convert raw market status strings into a ``MarketStatus`` enum."""
    normalized = (value or MarketStatus.OPEN.value).strip().lower()
    aliases = {
        "active": MarketStatus.OPEN,
        "finalized": MarketStatus.SETTLED,
        "resolved": MarketStatus.SETTLED,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return MarketStatus(normalized)
    except ValueError:
        log.warning("rest_client.unknown_market_status", raw_status=value)
        return MarketStatus.OPEN


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

        # Kalshi signs only the request path; query parameters are excluded.
        sign_path = urlparse(self._base_url).path.rstrip("/") + "/" + path.lstrip("/")

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
            status=_market_status(data.get("status")),
            yes_bid=_dec(data.get("yes_bid_dollars", data.get("yes_bid"))),
            yes_ask=_dec(data.get("yes_ask_dollars", data.get("yes_ask"))),
            last_price=_dec(data.get("last_price_dollars", data.get("last_price"))),
            volume=int(float(str(data.get("volume_fp", data.get("volume", 0))))),
            open_interest=int(float(str(data.get("open_interest_fp", data.get("open_interest", 0))))),
            close_time=_ts(data.get("close_time")),
            result=_market_result(data.get("result")),
            category=data.get("category", ""),
            subtitle=data.get("subtitle", ""),
            yes_sub_title=data.get("yes_sub_title", ""),
            no_sub_title=data.get("no_sub_title", ""),
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
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
        )

    @staticmethod
    def _parse_trade(data: dict[str, Any]) -> Trade:
        # count_fp is the new fixed-point string field (e.g. "10.00")
        count_raw = data.get("count_fp", data.get("count", 0))
        return Trade(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            count=int(float(str(count_raw))),
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
        # Kalshi API returns orderbook levels under yes_dollars/no_dollars
        # (inside orderbook_fp), falling back to legacy yes/no keys.
        yes_levels = tuple(
            OrderbookLevel(price=_dec(lvl[0]), contracts=_dec(lvl[1]))
            for lvl in (data.get("yes_dollars", data.get("yes")) or [])
        )
        no_levels = tuple(
            OrderbookLevel(price=_dec(lvl[0]), contracts=_dec(lvl[1]))
            for lvl in (data.get("no_dollars", data.get("no")) or [])
        )
        return OrderbookSnapshot(
            ticker=ticker,
            yes_bids=yes_levels,
            no_bids=no_levels,
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

    async def get_markets_page(
        self,
        **filters: Any,
    ) -> tuple[list[Market], str | None]:
        """Fetch a single page of markets plus the next cursor, if any."""
        data = await self._request("GET", "/markets", params=filters)
        return [self._parse_market(m) for m in data.get("markets", [])], data.get("cursor")

    async def get_all_markets(
        self,
        max_pages: int = 0,
        **filters: Any,
    ) -> list[Market]:
        """Fetch pages of markets for the supplied filters.

        Parameters
        ----------
        max_pages:
            Maximum number of pages to fetch.  0 means unlimited (fetch
            all pages).  Each page returns up to ``limit`` markets.
        """
        base_filters = dict(filters)
        cursor = base_filters.pop("cursor", None)
        markets: list[Market] = []
        page_count = 0

        while True:
            page_filters = dict(base_filters)
            if cursor:
                page_filters["cursor"] = cursor
            batch, cursor = await self.get_markets_page(**page_filters)
            markets.extend(batch)
            page_count += 1
            if not cursor:
                break
            if max_pages and page_count >= max_pages:
                break

        return markets

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
        # Kalshi returns orderbook_fp (with _dollars fields) or legacy orderbook
        ob = data.get("orderbook_fp", data.get("orderbook", data))
        return self._parse_orderbook(ob, ticker)

    async def get_orderbooks(self, tickers: list[str]) -> list[OrderbookSnapshot]:
        """Fetch orderbooks for multiple markets in a single bulk request."""
        params: dict[str, Any] = {"tickers": ",".join(tickers)}
        data = await self._request("GET", "/markets/orderbooks", params=params)
        results: list[OrderbookSnapshot] = []
        # Bulk response may use orderbooks_fp or orderbooks key
        books = data.get("orderbooks_fp", data.get("orderbooks", {}))
        for ticker_key, book_data in books.items():
            results.append(self._parse_orderbook(book_data, ticker_key))
        return results

    async def get_trades(
        self,
        ticker: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> tuple[list[Trade], str]:
        """Fetch public trades, optionally filtered by market.

        Returns (trades, cursor).  Pass the cursor to the next call
        for pagination.  An empty cursor means no more pages.
        """
        params: dict[str, Any] = {}
        if ticker is not None:
            params["ticker"] = ticker
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        data = await self._request("GET", "/markets/trades", params=params)
        trades = [self._parse_trade(t) for t in data.get("trades", [])]
        return trades, data.get("cursor", "")

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
        available = _money(
            data.get("available_balance_dollars"),
            data.get("available_balance", data.get("balance")),
        )
        total = _money(
            data.get("total_balance_dollars"),
            data.get("total_balance"),
        )
        if total == Decimal(0) and data.get("portfolio_value") is not None:
            total = available + _money(cents_value=data.get("portfolio_value"))
        elif total == Decimal(0):
            total = available
        return Balance(
            available=available,
            total=total,
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
            "yes_price_dollars": _fixed_point_dollars(request.yes_price),
            "time_in_force": request.time_in_force.api_value,
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
        """Cancel multiple orders in a single request (up to 20)."""
        body: dict[str, Any] = {"order_ids": order_ids}
        try:
            await self._request(
                "DELETE", "/portfolio/orders/batched", json_body=body
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
            body["yes_price_dollars"] = _fixed_point_dollars(request.yes_price)
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
                "yes_price_dollars": _fixed_point_dollars(req.yes_price),
                "time_in_force": req.time_in_force.api_value,
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
    def _parse_ohlc(data: dict[str, Any]) -> CandlestickOHLC:
        """Parse an OHLC sub-object (yes_bid or yes_ask)."""
        return CandlestickOHLC(
            open=_dec(data.get("open_dollars", data.get("open", 0))),
            high=_dec(data.get("high_dollars", data.get("high", 0))),
            low=_dec(data.get("low_dollars", data.get("low", 0))),
            close=_dec(data.get("close_dollars", data.get("close", 0))),
        )

    @staticmethod
    def _parse_candlestick(c: dict[str, Any]) -> Candlestick:
        price = c.get("price", {})

        # Handle dollar-string format (new API) or legacy integer format
        open_val = _dec(price.get("open_dollars", price.get("open", 0)))
        high_val = _dec(price.get("high_dollars", price.get("high", 0)))
        low_val = _dec(price.get("low_dollars", price.get("low", 0)))
        close_val = _dec(price.get("close_dollars", price.get("close", 0)))

        # Fallback to yes_ask if price block is empty
        if open_val == 0 and not price:
            ask = c.get("yes_ask", {})
            open_val = _dec(ask.get("open_dollars", ask.get("open", 0)))
            high_val = _dec(ask.get("high_dollars", ask.get("high", 0)))
            low_val = _dec(ask.get("low_dollars", ask.get("low", 0)))
            close_val = _dec(ask.get("close_dollars", ask.get("close", 0)))

        # Volume and OI: handle fixed-point string or integer
        vol_raw = c.get("volume_fp", c.get("volume", 0))
        oi_raw = c.get("open_interest_fp", c.get("open_interest", 0))

        # Parse end_period_ts: can be a unix timestamp integer or ISO string
        end_ts_raw = c.get("end_period_ts", "")
        if isinstance(end_ts_raw, (int, float)) and end_ts_raw > 0:
            end_ts = datetime.fromtimestamp(int(end_ts_raw), tz=timezone.utc)
        else:
            end_ts = _ts(str(end_ts_raw))

        # Optional extended OHLC for bid/ask
        yes_bid_data = c.get("yes_bid")
        yes_ask_data = c.get("yes_ask")

        return Candlestick(
            end_period_ts=end_ts,
            open=open_val,
            high=high_val,
            low=low_val,
            close=close_val,
            volume=int(float(str(vol_raw))),
            open_interest=int(float(str(oi_raw))),
            mean_price=_dec(price["mean_dollars"]) if price.get("mean_dollars") else None,
            previous_price=_dec(price["previous_dollars"]) if price.get("previous_dollars") else None,
            yes_bid_ohlc=KalshiRestClient._parse_ohlc(yes_bid_data) if yes_bid_data else None,
            yes_ask_ohlc=KalshiRestClient._parse_ohlc(yes_ask_data) if yes_ask_data else None,
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

    async def get_filters_by_sport(self) -> dict[str, Any]:
        """Return search filters organized by sport.

        Useful for discovering sports-related markets and filtering
        by league, team, etc.
        """
        return await self._request("GET", "/search/filters_by_sport")

    # ------------------------------------------------------------------
    # Batch market candlesticks
    # ------------------------------------------------------------------

    async def get_batch_candlesticks(
        self,
        market_tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
        include_latest_before_start: bool = False,
    ) -> list[MarketCandlesticks]:
        """Fetch candlestick data for multiple markets in one request.

        Parameters
        ----------
        market_tickers:
            Up to 100 market tickers.
        start_ts / end_ts:
            Unix timestamps in seconds.
        period_interval:
            Candle size in minutes (1, 60, 1440).
        include_latest_before_start:
            If True, prepends a synthetic candlestick for price continuity.

        Returns up to 10,000 total candlesticks across all markets.
        """
        params: dict[str, Any] = {
            "market_tickers": ",".join(market_tickers),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        if include_latest_before_start:
            params["include_latest_before_start"] = "true"

        data = await self._request("GET", "/markets/candlesticks", params=params)
        results: list[MarketCandlesticks] = []
        for entry in data.get("markets", []):
            ticker = entry.get("market_ticker", "")
            candles = [
                self._parse_candlestick(c)
                for c in entry.get("candlesticks", [])
            ]
            results.append(MarketCandlesticks(
                market_ticker=ticker,
                candlesticks=candles,
            ))
        return results

    # ------------------------------------------------------------------
    # Event candlesticks
    # ------------------------------------------------------------------

    async def get_event_candlesticks(
        self,
        series_ticker: str,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[Candlestick]:
        """Fetch aggregated candlesticks across all markets in an event.

        Parameters
        ----------
        series_ticker:
            Parent series ticker.
        event_ticker:
            Event ticker within the series.
        start_ts / end_ts:
            Unix timestamps.
        period_interval:
            Candle size in minutes: 1, 60, or 1440.
        """
        data = await self._request(
            "GET",
            f"/series/{series_ticker}/events/{event_ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return [self._parse_candlestick(c) for c in data.get("candlesticks", [])]

    # ------------------------------------------------------------------
    # Exchange status & metadata
    # ------------------------------------------------------------------

    async def get_exchange_status(self) -> ExchangeStatus:
        """Check whether the exchange is open for trading."""
        data = await self._request("GET", "/exchange/status")
        return ExchangeStatus(
            trading_active=bool(data.get("trading_active", False)),
            exchange_active=bool(data.get("exchange_active", False)),
        )

    async def get_user_data_timestamp(self) -> str:
        """Get the approximate validation timestamp for user data endpoints.

        Returns an ISO timestamp string indicating how fresh the
        portfolio/orders data is.
        """
        data = await self._request("GET", "/exchange/user_data_timestamp")
        return data.get("timestamp", "")

    async def get_fee_changes(self) -> list[dict[str, Any]]:
        """Fetch any pending or recent fee schedule changes."""
        data = await self._request("GET", "/series/fee_changes")
        return data.get("fee_changes", [])

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account_limits(self) -> AccountLimits:
        """Fetch API tier and rate limit information."""
        data = await self._request("GET", "/account/limits")
        return AccountLimits(
            tier=data.get("tier", ""),
            data=data,
        )

    # ------------------------------------------------------------------
    # Events (extended)
    # ------------------------------------------------------------------

    async def get_event_metadata(self, event_ticker: str) -> dict[str, Any]:
        """Fetch metadata only for an event (lighter than full event)."""
        data = await self._request("GET", f"/events/{event_ticker}/metadata")
        return data.get("metadata", data)

    async def get_multivariate_events(self, **filters: Any) -> list[dict[str, Any]]:
        """Fetch multivariate combo events with optional filtering.

        These are multi-outcome events (e.g. "which team wins the Super Bowl").
        """
        data = await self._request("GET", "/events/multivariate", params=filters)
        return data.get("events", [])

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    async def get_milestone(self, milestone_id: str) -> Milestone:
        """Fetch a single milestone by ID."""
        data = await self._request("GET", f"/milestones/{milestone_id}")
        m = data.get("milestone", data)
        return Milestone(
            milestone_id=str(m.get("id", milestone_id)),
            title=m.get("title", ""),
            category=m.get("category", ""),
            data=m,
        )

    async def get_milestones(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **filters: Any,
    ) -> list[Milestone]:
        """Fetch milestones with optional date filtering.

        Parameters
        ----------
        start_date / end_date:
            ISO date strings for filtering.
        """
        params: dict[str, Any] = dict(filters)
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = await self._request("GET", "/milestones", params=params)
        return [
            Milestone(
                milestone_id=str(m.get("id", "")),
                title=m.get("title", ""),
                category=m.get("category", ""),
                data=m,
            )
            for m in data.get("milestones", [])
        ]

    # ------------------------------------------------------------------
    # Live data
    # ------------------------------------------------------------------

    async def get_live_data(self, milestone_id: str) -> dict[str, Any]:
        """Fetch live data for a specific milestone.

        Returns real-time data relevant to the milestone's resolution
        (e.g. current temperature, game score).
        """
        data = await self._request("GET", f"/live_data/milestone/{milestone_id}")
        return data

    async def get_live_game_stats(self, milestone_id: str) -> dict[str, Any]:
        """Fetch play-by-play game statistics for a milestone.

        Only applicable for sports-related milestones.
        """
        data = await self._request(
            "GET", f"/live_data/milestone/{milestone_id}/game_stats"
        )
        return data

    async def get_live_data_batch(
        self, milestone_ids: list[str]
    ) -> dict[str, Any]:
        """Fetch live data for multiple milestones in one request."""
        params: dict[str, Any] = {
            "milestone_ids": ",".join(milestone_ids),
        }
        data = await self._request("GET", "/live_data/batch", params=params)
        return data

    # ------------------------------------------------------------------
    # Structured targets
    # ------------------------------------------------------------------

    async def get_structured_target(
        self, structured_target_id: str
    ) -> StructuredTarget:
        """Fetch a single structured target by ID."""
        data = await self._request(
            "GET", f"/structured_targets/{structured_target_id}"
        )
        st = data.get("structured_target", data)
        return StructuredTarget(
            structured_target_id=str(st.get("id", structured_target_id)),
            data=st,
        )

    async def get_structured_targets(self, **filters: Any) -> list[StructuredTarget]:
        """Fetch all structured targets (paginated, max 2000 per page).

        Filters: ``cursor``, ``limit``, etc.
        """
        data = await self._request(
            "GET", "/structured_targets", params=filters
        )
        return [
            StructuredTarget(
                structured_target_id=str(st.get("id", "")),
                data=st,
            )
            for st in data.get("structured_targets", [])
        ]

    # ------------------------------------------------------------------
    # Historical data (extended)
    # ------------------------------------------------------------------

    async def get_historical_market(self, ticker: str) -> Market:
        """Fetch a single historical (settled/archived) market by ticker."""
        data = await self._request("GET", f"/historical/markets/{ticker}")
        return self._parse_market(data.get("market", data))

    async def get_historical_markets(self, **filters: Any) -> list[Market]:
        """Fetch archived markets from the historical database.

        Filters: ``ticker``, ``event_ticker``, ``series_ticker``,
        ``status``, ``cursor``, ``limit``.
        """
        data = await self._request("GET", "/historical/markets", params=filters)
        return [self._parse_market(m) for m in data.get("markets", [])]

    # ------------------------------------------------------------------
    # Order group (single)
    # ------------------------------------------------------------------

    async def get_order_group(self, order_group_id: str) -> OrderGroup:
        """Fetch a single order group by ID."""
        data = await self._request(
            "GET", f"/portfolio/order_groups/{order_group_id}"
        )
        g = data.get("order_group", data)
        return OrderGroup(
            order_group_id=g.get("id", order_group_id),
            contracts_limit=_dec(g.get("contracts_limit_fp", "0")),
            is_auto_cancel_enabled=bool(g.get("is_auto_cancel_enabled", False)),
        )

    # ------------------------------------------------------------------
    # Multivariate event collections
    # ------------------------------------------------------------------

    async def get_mve_collection(
        self, collection_ticker: str
    ) -> dict[str, Any]:
        """Fetch a multivariate event collection."""
        data = await self._request(
            "GET",
            f"/multivariate_event_collections/{collection_ticker}",
        )
        return data.get("collection", data)

    async def get_mve_collections(self, **filters: Any) -> list[dict[str, Any]]:
        """Fetch all multivariate event collections."""
        data = await self._request(
            "GET", "/multivariate_event_collections", params=filters
        )
        return data.get("collections", [])

    async def lookup_mve_market(
        self, collection_ticker: str, **params: Any
    ) -> dict[str, Any]:
        """Look up a specific market in a multivariate event collection."""
        data = await self._request(
            "PUT",
            f"/multivariate_event_collections/{collection_ticker}/lookup",
            json_body=params,
        )
        return data

    # ------------------------------------------------------------------
    # Incentive programs
    # ------------------------------------------------------------------

    async def get_incentive_programs(self, **filters: Any) -> list[dict[str, Any]]:
        """List available incentive programs."""
        data = await self._request(
            "GET", "/incentive_programs", params=filters
        )
        return data.get("incentive_programs", [])
