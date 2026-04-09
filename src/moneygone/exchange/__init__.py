"""Kalshi exchange layer: authentication, REST client, and WebSocket client."""

from moneygone.exchange.auth import KalshiAuth
from moneygone.exchange.errors import (
    AuthError,
    KalshiAPIError,
    OrderError,
    RateLimitError,
    WebSocketError,
)
from moneygone.exchange.rate_limiter import AsyncRateLimiter
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import (
    Action,
    Balance,
    Fill,
    Market,
    MarketResult,
    MarketStatus,
    Order,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderRequest,
    OrderStatus,
    Position,
    Settlement,
    SettlementStatus,
    Side,
    TimeInForce,
    Trade,
    WSEvent,
)
from moneygone.exchange.ws_client import KalshiWebSocket

__all__ = [
    # Auth
    "KalshiAuth",
    # Clients
    "KalshiRestClient",
    "KalshiWebSocket",
    # Rate limiting
    "AsyncRateLimiter",
    # Errors
    "KalshiAPIError",
    "AuthError",
    "RateLimitError",
    "OrderError",
    "WebSocketError",
    # Enums
    "Action",
    "MarketResult",
    "MarketStatus",
    "OrderStatus",
    "SettlementStatus",
    "Side",
    "TimeInForce",
    # Data types
    "Balance",
    "Fill",
    "Market",
    "Order",
    "OrderbookLevel",
    "OrderbookSnapshot",
    "OrderRequest",
    "Position",
    "Settlement",
    "Trade",
    "WSEvent",
]
