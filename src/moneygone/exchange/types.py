"""Frozen dataclasses for all Kalshi exchange data types.

All monetary values use ``Decimal`` for precision. All timestamps use
``datetime`` (timezone-aware UTC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    IOC = "ioc"  # Immediate-or-cancel
    GTC = "gtc"  # Good-til-cancelled
    GTD = "gtd"  # Good-til-date
    FOK = "fok"  # Fill-or-kill


class OrderStatus(str, Enum):
    RESTING = "resting"
    PENDING = "pending"
    CANCELED = "canceled"
    EXECUTED = "executed"
    PARTIAL = "partial"


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


class MarketResult(str, Enum):
    YES = "yes"
    NO = "no"
    ALL_NO = "all_no"
    ALL_YES = "all_yes"
    VOIDED = "voided"
    NOT_SETTLED = ""


class SettlementStatus(str, Enum):
    SETTLED = "settled"
    UNSETTLED = "unsettled"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Market:
    """Snapshot of a Kalshi market."""

    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    status: MarketStatus
    yes_bid: Decimal
    yes_ask: Decimal
    last_price: Decimal
    volume: int
    open_interest: int
    close_time: datetime
    result: MarketResult = MarketResult.NOT_SETTLED
    category: str = ""


@dataclass(frozen=True, slots=True)
class OrderbookLevel:
    """Single price/size level in the orderbook.

    Represents a bid at a given price with a given contract count.
    Prices and counts are fixed-point strings from the API, stored as Decimal.
    """

    price: Decimal
    contracts: Decimal


@dataclass(frozen=True, slots=True)
class OrderbookSnapshot:
    """Point-in-time snapshot of a market's orderbook.

    Kalshi's orderbook returns **bids only**:
    - ``yes_bids``: bids for YES contracts, sorted ascending (best = last)
    - ``no_bids``: bids for NO contracts, sorted ascending (best = last)

    Asks are derived from the reciprocal relationship:
    - Best YES ask = $1.00 - best NO bid price
    - Best NO ask = $1.00 - best YES bid price
    """

    ticker: str
    yes_bids: tuple[OrderbookLevel, ...]
    no_bids: tuple[OrderbookLevel, ...]
    seq: int
    timestamp: datetime

    # Legacy aliases for backward compatibility
    @property
    def yes_levels(self) -> tuple[OrderbookLevel, ...]:
        return self.yes_bids

    @property
    def no_levels(self) -> tuple[OrderbookLevel, ...]:
        return self.no_bids

    @property
    def best_yes_bid(self) -> Decimal | None:
        """Highest YES bid price, or None if no bids."""
        return self.yes_bids[-1].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Decimal | None:
        """Highest NO bid price, or None if no bids."""
        return self.no_bids[-1].price if self.no_bids else None

    @property
    def best_yes_ask(self) -> Decimal | None:
        """Implied YES ask = $1.00 - best NO bid, or None if no NO bids."""
        if not self.no_bids:
            return None
        return Decimal("1.00") - self.no_bids[-1].price

    @property
    def best_no_ask(self) -> Decimal | None:
        """Implied NO ask = $1.00 - best YES bid, or None if no YES bids."""
        if not self.yes_bids:
            return None
        return Decimal("1.00") - self.yes_bids[-1].price

    @property
    def yes_spread(self) -> Decimal | None:
        """Spread on YES side = best_yes_ask - best_yes_bid."""
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def mid_price(self) -> Decimal | None:
        """Midpoint between best YES bid and best YES ask."""
        bid = self.best_yes_bid
        ask = self.best_yes_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def yes_depth(self, within_dollars: Decimal = Decimal("0.05")) -> Decimal:
        """Total YES bid volume within N dollars of best bid."""
        if not self.yes_bids:
            return Decimal("0")
        best = self.yes_bids[-1].price
        total = Decimal("0")
        for level in reversed(self.yes_bids):
            if best - level.price <= within_dollars:
                total += level.contracts
            else:
                break
        return total

    def no_depth(self, within_dollars: Decimal = Decimal("0.05")) -> Decimal:
        """Total NO bid volume within N dollars of best bid."""
        if not self.no_bids:
            return Decimal("0")
        best = self.no_bids[-1].price
        total = Decimal("0")
        for level in reversed(self.no_bids):
            if best - level.price <= within_dollars:
                total += level.contracts
            else:
                break
        return total

    @classmethod
    def from_api_response(cls, ticker: str, data: dict, seq: int = 0,
                          timestamp: datetime | None = None) -> "OrderbookSnapshot":
        """Parse an orderbook from Kalshi's API response format.

        Expects ``data`` to have ``orderbook_fp.yes_dollars`` and
        ``orderbook_fp.no_dollars`` arrays of [price_str, count_str] pairs.
        """
        from moneygone.utils.time import now_utc

        ob = data.get("orderbook_fp", data)
        yes_raw = ob.get("yes_dollars", [])
        no_raw = ob.get("no_dollars", [])

        yes_bids = tuple(
            OrderbookLevel(price=Decimal(p), contracts=Decimal(c))
            for p, c in yes_raw
        )
        no_bids = tuple(
            OrderbookLevel(price=Decimal(p), contracts=Decimal(c))
            for p, c in no_raw
        )
        return cls(
            ticker=ticker,
            yes_bids=yes_bids,
            no_bids=no_bids,
            seq=seq,
            timestamp=timestamp or now_utc(),
        )


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Parameters for submitting a new order."""

    ticker: str
    side: Side
    action: Action
    count: int
    yes_price: Decimal
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    client_order_id: str | None = None


@dataclass(frozen=True, slots=True)
class Order:
    """An order as returned by the exchange."""

    order_id: str
    ticker: str
    side: Side
    action: Action
    status: OrderStatus
    count: int
    remaining_count: int
    price: Decimal
    taker_fees: Decimal
    maker_fees: Decimal
    created_time: datetime


@dataclass(frozen=True, slots=True)
class Position:
    """Current position in a single market."""

    ticker: str
    event_ticker: str
    market_result: MarketResult
    yes_count: int
    no_count: int
    realized_pnl: Decimal
    settlement_status: SettlementStatus


@dataclass(frozen=True, slots=True)
class Fill:
    """A single fill (partial or full) on an order."""

    trade_id: str
    ticker: str
    side: Side
    action: Action
    count: int
    price: Decimal
    is_taker: bool
    created_time: datetime


@dataclass(frozen=True, slots=True)
class Trade:
    """A public trade on the tape."""

    trade_id: str
    ticker: str
    count: int
    yes_price: Decimal
    taker_side: Side
    created_time: datetime


@dataclass(frozen=True, slots=True)
class Balance:
    """Account balance snapshot."""

    available: Decimal
    total: Decimal


@dataclass(frozen=True, slots=True)
class Settlement:
    """Settlement record for a resolved market."""

    ticker: str
    market_result: MarketResult
    revenue: Decimal
    payout: Decimal
    settled_time: datetime


@dataclass(frozen=True, slots=True)
class Candlestick:
    """OHLC candlestick for a market."""

    end_period_ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    open_interest: int


@dataclass(frozen=True, slots=True)
class QueuePosition:
    """Order queue position for a resting limit order."""

    order_id: str
    market_ticker: str
    queue_position: Decimal  # Fixed-point, number of preceding shares


@dataclass(frozen=True, slots=True)
class OrderGroup:
    """An order group that can hold/trigger multiple orders together."""

    order_group_id: str
    contracts_limit: Decimal  # Fixed-point max contracts
    is_auto_cancel_enabled: bool


@dataclass(frozen=True, slots=True)
class ExchangeAnnouncement:
    """An exchange-wide announcement (info/warning/error)."""

    type: str           # "info" | "warning" | "error"
    message: str
    delivery_time: datetime
    status: str         # "active" | "inactive"


@dataclass(frozen=True, slots=True)
class DailySchedule:
    """Open/close times for a single trading day (Eastern Time)."""

    open_time: str   # "HH:MM"
    close_time: str  # "HH:MM"


@dataclass(frozen=True, slots=True)
class MaintenanceWindow:
    """Scheduled maintenance window."""

    start_datetime: datetime
    end_datetime: datetime


@dataclass(frozen=True, slots=True)
class ExchangeSchedule:
    """Exchange trading schedule."""

    standard_hours: list[DailySchedule]
    maintenance_windows: list[MaintenanceWindow]


@dataclass(frozen=True, slots=True)
class Series:
    """A Kalshi series (collection of related events)."""

    ticker: str
    title: str
    category: str
    tags: list[str]
    frequency: str  # e.g. "daily", "weekly"
    settlement_sources: list[str]


@dataclass(frozen=True, slots=True)
class AmendOrderRequest:
    """Parameters for amending an existing resting order."""

    order_id: str
    ticker: str
    side: Side
    action: Action
    yes_price: Decimal | None = None   # new limit price (dollars)
    count: int | None = None            # new total size


@dataclass(frozen=True, slots=True)
class BatchOrderItem:
    """One order within a batch create request."""

    ticker: str
    side: Side
    action: Action
    count: int
    yes_price: Decimal
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    client_order_id: str | None = None
    order_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class BatchOrderResult:
    """Result for a single order in a batch create response."""

    client_order_id: str | None
    order: Order | None
    error: str | None


@dataclass(frozen=True, slots=True)
class WSEvent:
    """A single event received over the WebSocket connection."""

    channel: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    timestamp: datetime | None = None
