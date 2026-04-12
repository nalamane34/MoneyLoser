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

    @property
    def api_value(self) -> str:
        """Return the current Kalshi REST enum value for this TIF."""
        mapping = {
            TimeInForce.IOC: "immediate_or_cancel",
            TimeInForce.GTC: "good_till_canceled",
            TimeInForce.FOK: "fill_or_kill",
        }
        if self not in mapping:
            raise ValueError(
                f"time_in_force={self.value!r} is not supported by the current Kalshi REST API"
            )
        return mapping[self]


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
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    created_time: datetime | None = None
    open_time: datetime | None = None
    previous_price: Decimal = Decimal("0")
    liquidity_dollars: Decimal = Decimal("0")
    no_bid: Decimal = Decimal("0")  # no_bid_dollars
    no_ask: Decimal = Decimal("0")  # no_ask_dollars
    volume_24h: int = 0  # volume_24h_fp
    market_type: str = ""  # e.g. "binary"
    strike_type: str = ""
    floor_strike: Decimal | None = None
    cap_strike: Decimal | None = None
    mve_selected_legs: tuple[Any, ...] = ()


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
    price: Decimal  # yes_price_dollars
    taker_fees: Decimal
    maker_fees: Decimal
    created_time: datetime
    no_price: Decimal = Decimal("0")
    fill_count: int = 0
    order_type: str = "limit"  # "limit" | "market"
    taker_fill_cost: Decimal = Decimal("0")
    maker_fill_cost: Decimal = Decimal("0")
    expiration_time: datetime | None = None
    last_update_time: datetime | None = None
    client_order_id: str = ""
    order_group_id: str = ""


@dataclass(frozen=True, slots=True)
class Position:
    """Current position in a single market.

    The API returns ``position_fp``: positive = YES contracts, negative = NO.
    Legacy ``yes_count``/``no_count`` are derived properties for convenience.
    """

    ticker: str
    position: int  # from position_fp; positive=YES, negative=NO
    market_exposure: Decimal  # market_exposure_dollars
    realized_pnl: Decimal  # realized_pnl_dollars
    total_traded: Decimal  # total_traded_dollars
    fees_paid: Decimal  # fees_paid_dollars
    last_updated_ts: datetime | None = None
    event_ticker: str = ""
    market_result: MarketResult = MarketResult.NOT_SETTLED
    resting_orders_count: int = 0

    @property
    def yes_count(self) -> int:
        return max(self.position, 0)

    @property
    def no_count(self) -> int:
        return abs(min(self.position, 0))

    @property
    def side(self) -> Side | None:
        if self.position > 0:
            return Side.YES
        elif self.position < 0:
            return Side.NO
        return None


@dataclass(frozen=True, slots=True)
class Fill:
    """A single fill (partial or full) on an order."""

    fill_id: str  # primary key (API: fill_id)
    ticker: str
    side: Side
    action: Action
    count: int  # from count_fp
    price: Decimal  # yes_price_dollars
    no_price: Decimal  # no_price_dollars
    fee_cost: Decimal  # fee_cost (fixed-point dollars)
    is_taker: bool
    created_time: datetime
    order_id: str | None = None
    client_order_id: str | None = None
    trade_id: str = ""  # legacy alias for fill_id

    @property
    def contract_price(self) -> Decimal:
        """Economic contract price in the traded side's native frame.

        Kalshi fills expose ``yes_price_dollars`` and may also include
        ``no_price_dollars``.  Local accounting should debit/credit the
        price of the actual contract that traded:

        - YES fill -> ``yes_price_dollars``
        - NO fill -> ``no_price_dollars`` when present, otherwise ``1 - yes_price``
        """
        if self.side == Side.YES:
            return self.price
        if self.no_price > Decimal("0"):
            return self.no_price
        return Decimal("1") - self.price


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

    available: Decimal  # available_balance (cents) or available_balance_dollars
    total: Decimal  # balance + portfolio_value
    updated_ts: int = 0  # unix timestamp of last update


@dataclass(frozen=True, slots=True)
class Settlement:
    """Settlement record for a resolved market."""

    ticker: str
    market_result: MarketResult
    revenue: Decimal  # normalized to dollars
    settled_time: datetime
    event_ticker: str = ""
    yes_count: int = 0  # from yes_count_fp
    no_count: int = 0  # from no_count_fp
    yes_total_cost: Decimal = Decimal("0")  # yes_total_cost_dollars
    no_total_cost: Decimal = Decimal("0")  # no_total_cost_dollars
    fee_cost: Decimal = Decimal("0")  # fee_cost (fixed-point dollars)


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
    # Extended fields from batch/new API
    mean_price: Decimal | None = None
    previous_price: Decimal | None = None
    yes_bid_ohlc: CandlestickOHLC | None = None
    yes_ask_ohlc: CandlestickOHLC | None = None


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
class CandlestickOHLC:
    """OHLC prices for a specific series (yes_bid, yes_ask) in a candlestick."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True, slots=True)
class MarketCandlesticks:
    """Candlesticks grouped by market from the batch endpoint."""

    market_ticker: str
    candlesticks: list[Candlestick]


@dataclass(frozen=True, slots=True)
class ExchangeStatus:
    """Current exchange operational status."""

    trading_active: bool
    exchange_active: bool


@dataclass(frozen=True, slots=True)
class Milestone:
    """A Kalshi milestone (resolution criteria / structured target)."""

    milestone_id: str
    title: str
    category: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructuredTarget:
    """A structured target definition for market resolution."""

    structured_target_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccountLimits:
    """API tier and rate limit information."""

    tier: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WSEvent:
    """A single event received over the WebSocket connection."""

    channel: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    timestamp: datetime | None = None
