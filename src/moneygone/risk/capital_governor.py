"""Shared capital reservations and global trading pause state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable

from moneygone.exchange.types import Action, Order, OrderStatus, Side

_ZERO = Decimal("0")
_ONE = Decimal("1")
_ACTIVE_ORDER_STATUSES = {
    OrderStatus.RESTING,
    OrderStatus.PARTIAL,
    OrderStatus.PENDING,
}


@dataclass(frozen=True)
class CapitalReservation:
    """Reserved capital for an intent or an open order."""

    key: str
    owner: str
    ticker: str
    category: str
    contracts: int
    price: Decimal
    dollars: Decimal
    created_at: datetime


@dataclass(frozen=True)
class CapitalGovernorSnapshot:
    """Read-only summary of current reserved capital and pause state."""

    total_reserved: Decimal
    reserved_by_category: dict[str, Decimal]
    reserved_contracts_by_ticker: dict[str, int]
    tail_reserved: Decimal
    paused: bool
    pause_reasons: dict[str, str]


class CapitalGovernor:
    """Tracks globally reserved capital and shared trading pauses."""

    def __init__(self) -> None:
        self._reservations: dict[str, CapitalReservation] = {}
        self._pause_reasons: dict[str, str] = {}

    def reserve_intent(
        self,
        key: str,
        *,
        owner: str,
        ticker: str,
        category: str,
        contracts: int,
        price: Decimal,
        available_cash: Decimal,
    ) -> bool:
        """Reserve capital for an in-flight order intent."""
        if contracts <= 0:
            return False
        dollars = Decimal(contracts) * price
        if dollars <= _ZERO:
            return False
        if dollars > available_cash:
            return False
        self._reservations[key] = CapitalReservation(
            key=key,
            owner=owner,
            ticker=ticker,
            category=category,
            contracts=contracts,
            price=price,
            dollars=dollars,
            created_at=datetime.now(timezone.utc),
        )
        return True

    def release(self, key: str) -> None:
        """Release an intent or order reservation."""
        self._reservations.pop(key, None)

    def sync_open_orders(
        self,
        orders: list[Order],
        *,
        category_lookup: dict[str, str] | Callable[[str], str | None] | None = None,
    ) -> None:
        """Replace all open-order reservations with the current tracked orders."""
        intent_reservations = {
            key: reservation
            for key, reservation in self._reservations.items()
            if not key.startswith("order:")
        }
        order_reservations: dict[str, CapitalReservation] = {}

        for order in orders:
            if order.status not in _ACTIVE_ORDER_STATUSES:
                continue
            if order.action != Action.BUY:
                continue
            contracts = int(order.remaining_count)
            if contracts <= 0:
                continue
            price = self._economic_order_price(order)
            if price <= _ZERO:
                continue
            category = self._resolve_category(order.ticker, category_lookup)
            key = f"order:{order.order_id}"
            order_reservations[key] = CapitalReservation(
                key=key,
                owner="open_order",
                ticker=order.ticker,
                category=category,
                contracts=contracts,
                price=price,
                dollars=Decimal(contracts) * price,
                created_at=order.created_time,
            )

        self._reservations = {**intent_reservations, **order_reservations}

    def pause_trading(self, source: str, reason: str) -> None:
        """Pause all new trading for a named source."""
        self._pause_reasons[source] = reason

    def resume_trading(self, source: str) -> None:
        """Clear a named trading pause source."""
        self._pause_reasons.pop(source, None)

    def snapshot(self) -> CapitalGovernorSnapshot:
        """Return a point-in-time summary."""
        total_reserved = _ZERO
        reserved_by_category: dict[str, Decimal] = {}
        reserved_contracts_by_ticker: dict[str, int] = {}
        tail_reserved = _ZERO
        for reservation in self._reservations.values():
            total_reserved += reservation.dollars
            reserved_by_category[reservation.category] = (
                reserved_by_category.get(reservation.category, _ZERO)
                + reservation.dollars
            )
            reserved_contracts_by_ticker[reservation.ticker] = (
                reserved_contracts_by_ticker.get(reservation.ticker, 0)
                + reservation.contracts
            )
            if reservation.price < Decimal("0.15") or reservation.price > Decimal("0.85"):
                tail_reserved += reservation.dollars
        return CapitalGovernorSnapshot(
            total_reserved=total_reserved,
            reserved_by_category=reserved_by_category,
            reserved_contracts_by_ticker=reserved_contracts_by_ticker,
            tail_reserved=tail_reserved,
            paused=bool(self._pause_reasons),
            pause_reasons=dict(self._pause_reasons),
        )

    @staticmethod
    def _resolve_category(
        ticker: str,
        category_lookup: dict[str, str] | Callable[[str], str | None] | None,
    ) -> str:
        if category_lookup is None:
            return "unknown"
        if callable(category_lookup):
            return category_lookup(ticker) or "unknown"
        return category_lookup.get(ticker, "unknown")

    @staticmethod
    def _economic_order_price(order: Order) -> Decimal:
        if order.side == Side.YES:
            return order.price
        if order.no_price > _ZERO:
            return order.no_price
        return _ONE - order.price
