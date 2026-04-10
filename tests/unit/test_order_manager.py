"""Tests for deterministic order reconciliation in OrderManager."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.exchange.types import Action, Fill, Order, OrderStatus, Side
from moneygone.execution.order_manager import OrderManager


_NOW = datetime(2026, 4, 9, 14, 30, tzinfo=timezone.utc)


class _NullClient:
    """No-op client for on_fill tests."""


def _make_order(
    *,
    order_id: str,
    remaining_count: int = 10,
    status: OrderStatus = OrderStatus.RESTING,
) -> Order:
    return Order(
        order_id=order_id,
        ticker="TEST",
        side=Side.YES,
        action=Action.BUY,
        status=status,
        count=10,
        remaining_count=remaining_count,
        price=Decimal("0.50"),
        taker_fees=Decimal("0.00"),
        maker_fees=Decimal("0.00"),
        created_time=_NOW,
    )


def _make_fill(
    *,
    trade_id: str = "trade-1",
    count: int = 3,
    order_id: str | None = None,
    client_order_id: str | None = None,
) -> Fill:
    return Fill(
        fill_id=trade_id,
        ticker="TEST",
        side=Side.YES,
        action=Action.BUY,
        count=count,
        price=Decimal("0.50"),
        no_price=Decimal("0.50"),
        fee_cost=Decimal("0"),
        is_taker=True,
        created_time=_NOW,
        order_id=order_id,
        client_order_id=client_order_id,
    )


def _make_manager(*orders: Order) -> OrderManager:
    manager = OrderManager(_NullClient())
    manager._open_orders = {order.order_id: order for order in orders}  # type: ignore[attr-defined]
    return manager


class TestOrderManagerFillReconciliation:
    """Deterministic fill matching should prefer identifiers over guesses."""

    def test_on_fill_prefers_order_id_when_multiple_orders_match(self) -> None:
        """An explicit order_id should resolve the correct open order."""
        manager = _make_manager(
            _make_order(order_id="order-1"),
            _make_order(order_id="order-2"),
        )

        manager.on_fill(_make_fill(order_id="order-2"))

        assert manager._open_orders["order-1"].remaining_count == 10  # type: ignore[attr-defined]
        assert manager._open_orders["order-2"].remaining_count == 7  # type: ignore[attr-defined]
        assert manager._open_orders["order-2"].status == OrderStatus.PARTIAL  # type: ignore[attr-defined]

    def test_on_fill_prefers_client_order_id_when_available(self) -> None:
        """A client_order_id should map back to the originating order."""
        manager = _make_manager(
            _make_order(order_id="order-1"),
            _make_order(order_id="order-2"),
        )
        manager._client_order_ids["client-2"] = "order-2"  # type: ignore[attr-defined]
        manager._order_client_order_ids["order-2"] = "client-2"  # type: ignore[attr-defined]

        manager.on_fill(_make_fill(client_order_id="client-2"))

        assert manager._open_orders["order-1"].remaining_count == 10  # type: ignore[attr-defined]
        assert manager._open_orders["order-2"].remaining_count == 7  # type: ignore[attr-defined]
        assert manager._open_orders["order-2"].status == OrderStatus.PARTIAL  # type: ignore[attr-defined]

    def test_on_fill_updates_unique_candidate_without_identifiers(self) -> None:
        """When only one candidate exists, fallback matching is safe."""
        manager = _make_manager(_make_order(order_id="order-1"))

        manager.on_fill(_make_fill())

        assert manager.open_order_count == 1
        assert manager._open_orders["order-1"].remaining_count == 7  # type: ignore[attr-defined]
        assert manager._open_orders["order-1"].status == OrderStatus.PARTIAL  # type: ignore[attr-defined]

    def test_on_fill_leaves_state_unchanged_when_ambiguous(self) -> None:
        """Multiple candidates without identifiers should not be guessed."""
        manager = _make_manager(
            _make_order(order_id="order-1"),
            _make_order(order_id="order-2"),
        )
        before = dict(manager._open_orders)  # type: ignore[attr-defined]

        manager.on_fill(_make_fill())

        assert manager._open_orders == before  # type: ignore[attr-defined]
        assert manager.open_order_count == 2
