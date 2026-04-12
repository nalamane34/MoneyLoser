"""Order lifecycle management for the Kalshi execution layer.

Tracks all open orders, handles submissions and cancellations, processes
fill events, and periodically reconciles local state against the exchange.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from moneygone.exchange.errors import OrderError
from moneygone.exchange.types import Fill, Order, OrderRequest, OrderStatus, Side

if TYPE_CHECKING:
    from moneygone.exchange.rest_client import KalshiRestClient

logger = structlog.get_logger(__name__)

_ACTIVE_ORDER_STATUSES = {
    OrderStatus.RESTING,
    OrderStatus.PARTIAL,
    OrderStatus.PENDING,
}


class OrderManager:
    """Manages the full order lifecycle.

    Maintains a local map of open orders and synchronises with the
    exchange via REST calls.

    Parameters
    ----------
    rest_client:
        Authenticated Kalshi REST client for order operations.
    """

    def __init__(self, rest_client: KalshiRestClient) -> None:
        self._client = rest_client
        self._open_orders: dict[str, Order] = {}
        self._client_order_ids: dict[str, str] = {}  # client_order_id -> order_id
        self._order_client_order_ids: dict[str, str] = {}  # order_id -> client_order_id
        self._pending_cancel_order_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit a new order to the exchange.

        Generates a unique ``client_order_id`` for idempotency if one is
        not already set on the request.

        Parameters
        ----------
        request:
            Order parameters (ticker, side, action, count, price, etc.).

        Returns
        -------
        Order
            The acknowledged order from the exchange.
        """
        # Attach a client_order_id for idempotency
        if request.client_order_id is None:
            coid = str(uuid.uuid4())
            # OrderRequest is frozen, so create a new instance
            request = OrderRequest(
                ticker=request.ticker,
                side=request.side,
                action=request.action,
                count=request.count,
                yes_price=request.yes_price,
                time_in_force=request.time_in_force,
                post_only=request.post_only,
                client_order_id=coid,
            )
        else:
            coid = request.client_order_id

        logger.info(
            "order_manager.submitting",
            ticker=request.ticker,
            side=request.side.value,
            action=request.action.value,
            count=request.count,
            price=str(request.yes_price),
            economic_price=str(
                request.yes_price if request.side == Side.YES else (1 - request.yes_price)
            ),
            client_order_id=coid,
        )

        order = await self._client.create_order(request)

        # Only keep exchange-active orders in the local open-order map.
        if order.status in _ACTIVE_ORDER_STATUSES:
            self._open_orders[order.order_id] = order
            self._client_order_ids[coid] = order.order_id
            self._order_client_order_ids[order.order_id] = coid

        logger.info(
            "order_manager.submitted",
            order_id=order.order_id,
            status=order.status.value,
            ticker=order.ticker,
            tracked=order.status in _ACTIVE_ORDER_STATUSES,
        )
        return order

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by exchange order ID.

        Returns ``True`` when cancellation is authoritatively closed from the
        local order manager's perspective. If exchange confirmation is still
        ambiguous, the order remains tracked in a pending-cancel state so the
        engine does not immediately re-enter the same thesis.
        """
        logger.info("order_manager.cancelling", order_id=order_id)
        tracked_locally = order_id in self._open_orders
        if tracked_locally:
            self._pending_cancel_order_ids.add(order_id)
        await self._client.cancel_order(order_id)
        if not tracked_locally:
            logger.info("order_manager.cancelled", order_id=order_id, confirmed_closed=True)
            return True

        confirmed_closed = False
        try:
            refreshed = await self._client.get_order(order_id)
        except OrderError as exc:
            if exc.status_code == 404:
                self._forget_order(order_id)
                confirmed_closed = True
            else:
                logger.debug(
                    "order_manager.cancel_confirmation_failed",
                    order_id=order_id,
                    exc_info=True,
                )
        except Exception:
            logger.debug(
                "order_manager.cancel_confirmation_failed",
                order_id=order_id,
                exc_info=True,
            )
        else:
            if refreshed.status in (OrderStatus.CANCELED, OrderStatus.EXECUTED):
                self._forget_order(order_id)
                confirmed_closed = True
            else:
                self._open_orders[order_id] = refreshed
                if refreshed.client_order_id:
                    self._client_order_ids[refreshed.client_order_id] = order_id
                    self._order_client_order_ids[order_id] = refreshed.client_order_id

        logger.info(
            "order_manager.cancelled",
            order_id=order_id,
            confirmed_closed=confirmed_closed,
            pending_cancel=order_id in self._pending_cancel_order_ids,
        )
        return confirmed_closed

    async def cancel_all(self, ticker: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by ticker.

        Parameters
        ----------
        ticker:
            If provided, only cancel orders for this ticker.

        Returns
        -------
        int
            Number of orders cancelled.
        """
        to_cancel = [
            oid
            for oid, order in self._open_orders.items()
            if ticker is None or order.ticker == ticker
        ]

        if not to_cancel:
            return 0

        logger.info(
            "order_manager.cancelling_all",
            count=len(to_cancel),
            ticker=ticker,
        )

        batch_size = getattr(self._client, "max_batch_cancel_size", 20)
        batch_size = max(1, min(20, int(batch_size)))
        batch_size = max(1, min(batch_size, 5))

        for start in range(0, len(to_cancel), batch_size):
            batch = to_cancel[start:start + batch_size]
            try:
                if len(batch) > 1:
                    await self._client.batch_cancel_orders(batch)
                else:
                    await self._client.cancel_order(batch[0])
            except OrderError as exc:
                if exc.status_code != 429 or len(batch) == 1:
                    raise
                logger.warning(
                    "order_manager.batch_cancel_rate_limited",
                    batch_size=len(batch),
                    retry_after=getattr(exc, "retry_after", None),
                )
                await asyncio.sleep(getattr(exc, "retry_after", None) or 1.5)
                for order_id in batch:
                    await self._client.cancel_order(order_id)
                    await asyncio.sleep(0.15)

            if start + batch_size < len(to_cancel):
                # Kalshi counts each order in a batch against the per-second
                # order-op limit, so pace large shutdown cancellations.
                await asyncio.sleep(1.05)

        for oid in to_cancel:
            self._forget_order(oid)

        logger.info(
            "order_manager.cancelled_all",
            count=len(to_cancel),
            ticker=ticker,
        )
        return len(to_cancel)

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def _order_matches_fill(self, order: Order, fill: Fill) -> bool:
        return (
            order.ticker == fill.ticker
            and order.side == fill.side
            and order.action == fill.action
            and order.status in (
                OrderStatus.RESTING,
                OrderStatus.PARTIAL,
                OrderStatus.PENDING,
            )
        )

    def _find_matching_order(self, fill: Fill) -> Order | None:
        """Resolve the local order for a fill without guessing."""
        if fill.order_id is not None:
            order = self._open_orders.get(fill.order_id)
            if order is None:
                logger.warning(
                    "order_manager.fill_unknown_order_id",
                    fill_id=fill.fill_id,
                    order_id=fill.order_id,
                    ticker=fill.ticker,
                )
                return None
            if not self._order_matches_fill(order, fill):
                logger.error(
                    "order_manager.fill_identifier_mismatch",
                    fill_id=fill.fill_id,
                    order_id=fill.order_id,
                    ticker=fill.ticker,
                    order_ticker=order.ticker,
                )
                return None
            return order

        if fill.client_order_id is not None:
            order_id = self._client_order_ids.get(fill.client_order_id)
            if order_id is None:
                logger.warning(
                    "order_manager.fill_unknown_client_order_id",
                    fill_id=fill.fill_id,
                    client_order_id=fill.client_order_id,
                    ticker=fill.ticker,
                )
                return None
            order = self._open_orders.get(order_id)
            if order is None:
                logger.warning(
                    "order_manager.fill_stale_client_order_id",
                    fill_id=fill.fill_id,
                    client_order_id=fill.client_order_id,
                    order_id=order_id,
                    ticker=fill.ticker,
                )
                return None
            if not self._order_matches_fill(order, fill):
                logger.error(
                    "order_manager.fill_identifier_mismatch",
                    fill_id=fill.fill_id,
                    client_order_id=fill.client_order_id,
                    order_id=order_id,
                    ticker=fill.ticker,
                    order_ticker=order.ticker,
                )
                return None
            return order

        candidates = [
            order
            for order in self._open_orders.values()
            if self._order_matches_fill(order, fill)
        ]
        if len(candidates) == 1:
            return candidates[0]

        logger.warning(
            "order_manager.fill_ambiguous",
            fill_id=fill.fill_id,
            ticker=fill.ticker,
            candidate_count=len(candidates),
            candidate_order_ids=[order.order_id for order in candidates],
        )
        return None

    def _forget_order(self, order_id: str) -> None:
        """Remove an order from local tracking and clear identifier indexes."""
        self._open_orders.pop(order_id, None)
        self._pending_cancel_order_ids.discard(order_id)
        client_order_id = self._order_client_order_ids.pop(order_id, None)
        if client_order_id is not None:
            self._client_order_ids.pop(client_order_id, None)

    def on_fill(self, fill: Fill) -> None:
        """Process a fill event and update order tracking.

        When all contracts on an order have been filled, the order is
        removed from the open orders map.
        """
        matching_order = self._find_matching_order(fill)

        if matching_order is None:
            logger.warning(
                "order_manager.fill_no_match",
                fill_id=fill.fill_id,
                ticker=fill.ticker,
            )
            return

        # Update remaining count
        new_remaining = max(0, matching_order.remaining_count - fill.count)
        if new_remaining == 0:
            new_status = OrderStatus.EXECUTED
        else:
            new_status = OrderStatus.PARTIAL

        # Replace with updated order (frozen dataclass)
        updated = Order(
            order_id=matching_order.order_id,
            ticker=matching_order.ticker,
            side=matching_order.side,
            action=matching_order.action,
            status=new_status,
            count=matching_order.count,
            remaining_count=new_remaining,
            price=matching_order.price,
            taker_fees=matching_order.taker_fees,
            maker_fees=matching_order.maker_fees,
            created_time=matching_order.created_time,
        )

        if new_status == OrderStatus.EXECUTED:
            self._forget_order(matching_order.order_id)
            logger.info(
                "order_manager.order_filled",
                order_id=matching_order.order_id,
                ticker=fill.ticker,
            )
        else:
            self._open_orders[matching_order.order_id] = updated
            logger.info(
                "order_manager.order_partial_fill",
                order_id=matching_order.order_id,
                ticker=fill.ticker,
                remaining=new_remaining,
            )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self) -> None:
        """Synchronize local order state with the exchange.

        Fetches the full order set with cursor pagination, filters to
        exchange-active orders, and replaces local tracking.  This avoids
        dropping partially filled or pending orders simply because they
        are not returned by the resting-only view.
        """
        logger.info("order_manager.reconciling")

        exchange_orders = await self._client.get_orders(limit=1_000, paginate=True)
        active_statuses = {
            OrderStatus.RESTING,
            OrderStatus.PARTIAL,
            OrderStatus.PENDING,
        }
        exchange_orders = [
            order for order in exchange_orders if order.status in active_statuses
        ]
        exchange_map = {o.order_id: o for o in exchange_orders}
        pending_cleared = 0

        # Reconciliation is the authoritative exchange view. If an order is
        # still active here, any earlier local pending-cancel marker should be
        # cleared so the engine does not treat a valid live order as limbo.
        for order_id in list(self._pending_cancel_order_ids):
            if order_id in exchange_map:
                self._pending_cancel_order_ids.discard(order_id)
                pending_cleared += 1

        # Find orders we track locally but the exchange doesn't know about
        stale = set(self._open_orders.keys()) - set(exchange_map.keys())
        for oid in stale:
            logger.warning(
                "order_manager.stale_order_removed",
                order_id=oid,
                ticker=self._open_orders[oid].ticker,
            )
            self._forget_order(oid)

        # Find orders on the exchange we don't track locally
        new = set(exchange_map.keys()) - set(self._open_orders.keys())
        for oid in new:
            logger.warning(
                "order_manager.unknown_order_found",
                order_id=oid,
                ticker=exchange_map[oid].ticker,
            )

        # Update local state in-place to preserve client_order_id mappings
        # for orders that existed in both local and exchange state.
        for oid, order in exchange_map.items():
            self._open_orders[oid] = order

        logger.info(
            "order_manager.reconciled",
            open_orders=len(self._open_orders),
            stale_removed=len(stale),
            new_found=len(new),
            pending_cleared=pending_cleared,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_open_orders(self, ticker: str | None = None) -> list[Order]:
        """Return currently tracked open orders.

        Parameters
        ----------
        ticker:
            If provided, only return orders for this ticker.
        """
        orders = list(self._open_orders.values())
        if ticker is not None:
            orders = [o for o in orders if o.ticker == ticker]
        return orders

    @property
    def open_order_count(self) -> int:
        """Number of currently tracked open orders."""
        return len(self._open_orders)
