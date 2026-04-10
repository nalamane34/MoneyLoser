"""Execution strategies controlling how orders are placed and managed.

Three strategies are provided:

- **PassiveStrategy**: Posts a limit order at or better than best bid/ask
  to earn maker rebates.  Cancels after a configurable timeout.
- **AggressiveStrategy**: Places a limit order at the best opposite price
  to fill immediately (taker).
- **AdaptiveStrategy**: Starts passive, switches to aggressive if not
  filled within a timeout window.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from moneygone.exchange.errors import OrderError
from moneygone.exchange.types import (
    Action,
    Order,
    OrderRequest,
    OrderStatus,
    OrderbookSnapshot,
    Side,
    TimeInForce,
)
from moneygone.signals.edge import EdgeResult
from moneygone.sizing.kelly import SizeResult

if TYPE_CHECKING:
    from moneygone.exchange.rest_client import KalshiRestClient
    from moneygone.execution.order_manager import OrderManager

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ExecutionStrategy(ABC):
    """Abstract base for order execution strategies.

    Subclasses implement :meth:`execute` which translates an edge and
    size decision into a submitted order (or ``None`` if execution is
    not attempted).
    """

    @abstractmethod
    async def execute(
        self,
        edge_result: EdgeResult,
        size_result: SizeResult,
        order_manager: OrderManager,
        orderbook: OrderbookSnapshot,
    ) -> Order | None:
        """Execute the strategy: build and submit an order.

        Parameters
        ----------
        edge_result:
            The computed edge with side, price, and liquidity info.
        size_result:
            The Kelly-computed position size.
        order_manager:
            Order manager for submitting/cancelling orders.
        orderbook:
            Current orderbook snapshot for price discovery.

        Returns
        -------
        Order | None
            The submitted order, or None if execution was skipped.
        """


# ---------------------------------------------------------------------------
# Passive (maker) strategy
# ---------------------------------------------------------------------------


class PassiveStrategy(ExecutionStrategy):
    """Place a post-only limit order at or better than best bid/ask.

    Designed to earn maker rebates (zero fees on Kalshi). The order rests
    in the book and waits for a counterparty.  If not filled within
    ``timeout_seconds``, the order is cancelled.

    Parameters
    ----------
    timeout_seconds:
        Maximum time to wait for a fill before cancelling.
    price_improve_cents:
        Number of cents to improve over best bid/ask (0 = join).
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        price_improve_cents: int = 0,
    ) -> None:
        self._timeout = timeout_seconds
        self._improve = Decimal(str(price_improve_cents)) / Decimal("100")

    async def execute(
        self,
        edge_result: EdgeResult,
        size_result: SizeResult,
        order_manager: OrderManager,
        orderbook: OrderbookSnapshot,
    ) -> Order | None:
        if size_result.contracts <= 0:
            return None

        price = self._compute_passive_price(edge_result, orderbook)
        if price is None:
            logger.warning(
                "passive.no_price",
                ticker=orderbook.ticker,
                side=edge_result.side,
            )
            return None

        side = Side(edge_result.side)
        request = OrderRequest(
            ticker=orderbook.ticker,
            side=side,
            action=Action(edge_result.action),
            count=size_result.contracts,
            yes_price=price,
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )

        try:
            order = await order_manager.submit_order(request)
        except OrderError as exc:
            if "post only cross" in str(exc).lower():
                logger.warning(
                    "passive.post_only_cross",
                    ticker=orderbook.ticker,
                    side=edge_result.side,
                    price=str(price),
                )
                return None
            raise

        logger.info(
            "passive.order_placed",
            order_id=order.order_id,
            ticker=order.ticker,
            price=str(price),
            contracts=size_result.contracts,
        )

        # Wait for fill or timeout
        filled = await self._wait_for_fill(order, order_manager)
        if not filled:
            try:
                await order_manager.cancel_order(order.order_id)
                logger.info(
                    "passive.timeout_cancel",
                    order_id=order.order_id,
                    timeout=self._timeout,
                )
            except Exception:
                logger.warning(
                    "passive.cancel_failed",
                    order_id=order.order_id,
                    exc_info=True,
                )
            return None

        return order

    def _compute_passive_price(
        self,
        edge: EdgeResult,
        orderbook: OrderbookSnapshot,
    ) -> Decimal | None:
        """Determine the passive limit price (always returned as yes_price).

        The Kalshi API always uses ``yes_price_dollars`` regardless of side.
        For YES buys: yes_price = one tick below YES ask (= target_price).
        For NO buys: compute NO bid one tick below NO ask, then convert
        to yes_price via ``1 - no_bid``.
        """
        if edge.side == "yes":
            if not orderbook.yes_levels:
                return None
            # target_price = YES ask = 1 - best NO bid
            # Post one tick below the ask
            price = edge.target_price - Decimal("0.01") + self._improve
            price = max(Decimal("0.01"), min(Decimal("0.99"), price))
        else:
            if not orderbook.no_levels:
                return None
            # target_price = NO ask = 1 - best YES bid
            # Desired NO bid = NO ask - 1 tick
            no_bid = edge.target_price - Decimal("0.01") + self._improve
            no_bid = max(Decimal("0.01"), min(Decimal("0.99"), no_bid))
            # Convert to yes_price for the API
            price = _ONE - no_bid

        return price

    async def _wait_for_fill(
        self, order: Order, order_manager: OrderManager
    ) -> bool:
        """Poll order status until filled or timeout."""
        deadline = asyncio.get_event_loop().time() + self._timeout
        poll_interval = 0.5

        while asyncio.get_event_loop().time() < deadline:
            # Check if the order is still in open orders
            open_orders = order_manager.get_open_orders()
            still_open = any(o.order_id == order.order_id for o in open_orders)
            if not still_open:
                # Order was removed from open orders => filled or cancelled
                return True
            await asyncio.sleep(poll_interval)

        return False


# ---------------------------------------------------------------------------
# Aggressive (taker) strategy
# ---------------------------------------------------------------------------


class AggressiveStrategy(ExecutionStrategy):
    """Place a limit order at the best ask price to fill immediately.

    Crosses the spread and pays taker fees, but gets immediate execution.
    Uses IOC (Immediate-or-Cancel) to avoid resting unfilled.

    Parameters
    ----------
    prefer_maker:
        If True, use GTC instead of IOC to potentially save fees if the
        order doesn't immediately match (unlikely at best ask).
    """

    def __init__(self, prefer_maker: bool = False) -> None:
        self._prefer_maker = prefer_maker

    async def execute(
        self,
        edge_result: EdgeResult,
        size_result: SizeResult,
        order_manager: OrderManager,
        orderbook: OrderbookSnapshot,
    ) -> Order | None:
        if size_result.contracts <= 0:
            return None

        # target_price is in the side's frame (YES ask or NO ask).
        # The API always uses yes_price_dollars, so convert NO prices.
        if edge_result.side == "yes":
            price = edge_result.target_price
        else:
            # target_price = NO ask; to fill at ask, yes_price = 1 - NO_ask
            price = _ONE - edge_result.target_price

        if price <= _ZERO or price >= _ONE:
            logger.warning(
                "aggressive.invalid_price",
                ticker=orderbook.ticker,
                price=str(price),
            )
            return None

        side = Side(edge_result.side)
        tif = TimeInForce.GTC if self._prefer_maker else TimeInForce.IOC

        request = OrderRequest(
            ticker=orderbook.ticker,
            side=side,
            action=Action(edge_result.action),
            count=size_result.contracts,
            yes_price=price,
            time_in_force=tif,
            post_only=False,
        )

        order = await order_manager.submit_order(request)

        logger.info(
            "aggressive.order_placed",
            order_id=order.order_id,
            ticker=order.ticker,
            price=str(price),
            contracts=size_result.contracts,
            time_in_force=tif.value,
        )

        return order


# ---------------------------------------------------------------------------
# Adaptive strategy
# ---------------------------------------------------------------------------


class AdaptiveStrategy(ExecutionStrategy):
    """Start passive, switch to aggressive if not filled within timeout.

    Combines the cost savings of passive execution with the certainty
    of aggressive execution as a fallback.

    Parameters
    ----------
    passive_timeout_seconds:
        How long to wait for a passive fill before switching.
    price_improve_cents:
        Cents to improve over best bid in passive mode.
    prefer_maker:
        Whether to prefer maker-style execution even in aggressive mode.
    """

    def __init__(
        self,
        passive_timeout_seconds: float = 15.0,
        price_improve_cents: int = 0,
        prefer_maker: bool = True,
    ) -> None:
        self._passive = PassiveStrategy(
            timeout_seconds=passive_timeout_seconds,
            price_improve_cents=price_improve_cents,
        )
        self._aggressive = AggressiveStrategy(prefer_maker=prefer_maker)
        self._passive_timeout = passive_timeout_seconds

    async def execute(
        self,
        edge_result: EdgeResult,
        size_result: SizeResult,
        order_manager: OrderManager,
        orderbook: OrderbookSnapshot,
    ) -> Order | None:
        if size_result.contracts <= 0:
            return None

        logger.info(
            "adaptive.trying_passive",
            ticker=orderbook.ticker,
            timeout=self._passive_timeout,
        )

        # Try passive first
        order = await self._passive.execute(
            edge_result, size_result, order_manager, orderbook
        )

        if order is not None:
            logger.info(
                "adaptive.passive_success",
                order_id=order.order_id,
                ticker=order.ticker,
            )
            return order

        # Passive failed or timed out -- switch to aggressive
        logger.info(
            "adaptive.switching_to_aggressive",
            ticker=orderbook.ticker,
        )

        order = await self._aggressive.execute(
            edge_result, size_result, order_manager, orderbook
        )

        if order is not None:
            logger.info(
                "adaptive.aggressive_success",
                order_id=order.order_id,
                ticker=order.ticker,
            )

        return order
