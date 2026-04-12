"""Tests for deterministic order reconciliation in OrderManager."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moneygone.exchange.types import Action, Fill, Order, OrderStatus, Side
from moneygone.execution.order_manager import OrderManager


_NOW = datetime(2026, 4, 9, 14, 30, tzinfo=timezone.utc)


class _NullClient:
    """No-op client for on_fill tests."""

    async def get_orders(self, **_filters):
        return []


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


class _ReconcileClient:
    def __init__(self, orders: list[Order]) -> None:
        self.orders = orders
        self.calls: list[dict[str, object]] = []

    async def get_orders(self, **filters):
        self.calls.append(dict(filters))
        return list(self.orders)


class _CancelClient:
    def __init__(self, refreshed_order: Order | None = None) -> None:
        self.cancelled: list[str] = []
        self.refreshed_order = refreshed_order

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    async def get_order(self, order_id: str) -> Order:
        if self.refreshed_order is None:
            raise RuntimeError("refresh unavailable")
        return self.refreshed_order


class TestOrderManagerReconcile:
    def test_reconcile_fetches_paginated_active_orders_not_just_resting(self) -> None:
        client = _ReconcileClient(
            [
                _make_order(order_id="resting-order", status=OrderStatus.RESTING),
                _make_order(order_id="partial-order", status=OrderStatus.PARTIAL),
                _make_order(order_id="pending-order", status=OrderStatus.PENDING),
                _make_order(order_id="done-order", status=OrderStatus.EXECUTED),
            ]
        )
        manager = OrderManager(client)  # type: ignore[arg-type]

        import asyncio

        asyncio.run(manager.reconcile())

        assert set(manager._open_orders) == {  # type: ignore[attr-defined]
            "resting-order",
            "partial-order",
            "pending-order",
        }
        assert client.calls == [{"limit": 1000, "paginate": True}]

    def test_reconcile_clears_pending_cancel_for_orders_still_active_on_exchange(self) -> None:
        client = _ReconcileClient(
            [_make_order(order_id="order-1", status=OrderStatus.RESTING)]
        )
        manager = _make_manager(_make_order(order_id="order-1"))
        manager._client = client  # type: ignore[attr-defined]
        manager._pending_cancel_order_ids.add("order-1")  # type: ignore[attr-defined]

        import asyncio

        asyncio.run(manager.reconcile())

        assert "order-1" not in manager._pending_cancel_order_ids  # type: ignore[attr-defined]


class TestOrderManagerCancelLifecycle:
    def test_cancel_order_keeps_order_tracked_when_confirmation_is_ambiguous(self) -> None:
        order = _make_order(order_id="order-1")
        client = _CancelClient(
            refreshed_order=_make_order(order_id="order-1", status=OrderStatus.PENDING)
        )
        manager = _make_manager(order)
        manager._client = client  # type: ignore[attr-defined]

        import asyncio

        confirmed = asyncio.run(manager.cancel_order("order-1"))

        assert confirmed is False
        assert client.cancelled == ["order-1"]
        assert "order-1" in manager._open_orders  # type: ignore[attr-defined]
        assert "order-1" in manager._pending_cancel_order_ids  # type: ignore[attr-defined]

    def test_cancel_order_forgets_order_after_confirmed_exchange_closure(self) -> None:
        order = _make_order(order_id="order-1")
        client = _CancelClient(
            refreshed_order=_make_order(order_id="order-1", status=OrderStatus.CANCELED)
        )
        manager = _make_manager(order)
        manager._client = client  # type: ignore[attr-defined]

        import asyncio

        confirmed = asyncio.run(manager.cancel_order("order-1"))

        assert confirmed is True
        assert client.cancelled == ["order-1"]
        assert "order-1" not in manager._open_orders  # type: ignore[attr-defined]
        assert "order-1" not in manager._pending_cancel_order_ids  # type: ignore[attr-defined]
