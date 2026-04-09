"""Fill simulation for backtesting order execution.

Provides three fill models of increasing realism:

- **instant**: Always fills at the order price (unrealistic baseline).
- **queue**: Estimates queue position and fills proportionally to volume.
- **realistic**: Combines queue model with partial fill probability and
  price slippage.

Used by the backtest engine to simulate how orders would have executed
against historical orderbook snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

import structlog

from moneygone.exchange.types import OrderbookSnapshot, OrderRequest, Side
from moneygone.signals.fees import KalshiFeeCalculator

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


class FillModel(str, Enum):
    INSTANT = "instant"
    QUEUE = "queue"
    REALISTIC = "realistic"


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """Result of a simulated order fill."""

    filled: bool
    """Whether the order was filled (fully or partially)."""

    fill_price: Decimal
    """Average fill price (may differ from order price due to slippage)."""

    filled_contracts: int
    """Number of contracts that filled."""

    fees: Decimal
    """Estimated fees for the fill."""

    slippage: Decimal
    """Price difference vs. order price (positive = worse for us)."""

    fill_model: str
    """Name of the fill model used."""


class FillSimulator:
    """Simulates order fills against historical orderbook snapshots.

    Parameters
    ----------
    model:
        Fill model to use: "instant", "queue", or "realistic".
    fee_calculator:
        Fee calculator for cost estimation.
    slippage_bps:
        Additional basis points of slippage to add (for conservative
        estimates beyond what the model captures).
    """

    def __init__(
        self,
        model: str = "realistic",
        fee_calculator: KalshiFeeCalculator | None = None,
        slippage_bps: float = 0.0,
    ) -> None:
        self._model = FillModel(model)
        self._fees = fee_calculator or KalshiFeeCalculator()
        self._slippage_bps = Decimal(str(slippage_bps)) / Decimal("10000")

    def simulate_fill(
        self,
        order: OrderRequest,
        orderbook: OrderbookSnapshot,
        time_in_queue: timedelta = timedelta(0),
    ) -> SimulatedFill:
        """Simulate an order fill against a historical orderbook snapshot.

        Parameters
        ----------
        order:
            The order to simulate.
        orderbook:
            The orderbook state at the time of the order.
        time_in_queue:
            How long the order has been resting in the book (affects
            queue position in queue/realistic models).

        Returns
        -------
        SimulatedFill
            The simulated fill result.
        """
        if self._model == FillModel.INSTANT:
            return self._instant_fill(order)
        elif self._model == FillModel.QUEUE:
            return self._queue_fill(order, orderbook, time_in_queue)
        else:
            return self._realistic_fill(order, orderbook, time_in_queue)

    # ------------------------------------------------------------------
    # Instant fill model
    # ------------------------------------------------------------------

    def _instant_fill(self, order: OrderRequest) -> SimulatedFill:
        """Always fills at the order price.  Unrealistic baseline."""
        is_maker = order.post_only
        if is_maker:
            fees = self._fees.maker_fee(order.count, order.yes_price)
        else:
            fees = self._fees.taker_fee(order.count, order.yes_price)

        return SimulatedFill(
            filled=True,
            fill_price=order.yes_price,
            filled_contracts=order.count,
            fees=fees,
            slippage=_ZERO,
            fill_model="instant",
        )

    # ------------------------------------------------------------------
    # Queue fill model
    # ------------------------------------------------------------------

    def _queue_fill(
        self,
        order: OrderRequest,
        orderbook: OrderbookSnapshot,
        time_in_queue: timedelta,
    ) -> SimulatedFill:
        """Estimate queue position and fill proportionally to volume.

        Assumes orders at the same price level fill in FIFO order.
        The longer an order rests, the more likely it is to fill.
        """
        levels = self._get_relevant_levels(order, orderbook)
        if not levels:
            return self._no_fill(order, "queue")

        # Find the level at or better than our order price
        order_price = order.yes_price
        matching_volume = 0
        for level in levels:
            if self._price_matches(order, level.price):
                matching_volume += int(level.contracts)

        if matching_volume == 0:
            # Our price doesn't match any level => crosses the spread (taker fill)
            return self._taker_fill(order, orderbook)

        # Estimate fill probability based on queue position
        # Assume we're at the back of the queue at our price level
        # Time in queue improves our position
        queue_seconds = time_in_queue.total_seconds()
        # Rough model: 1% of volume fills per second of queue time
        fill_fraction = min(1.0, queue_seconds * 0.01 / max(matching_volume, 1))

        filled_contracts = int(order.count * fill_fraction)
        if filled_contracts <= 0:
            return self._no_fill(order, "queue")

        is_maker = order.post_only
        if is_maker:
            fees = self._fees.maker_fee(filled_contracts, order_price)
        else:
            fees = self._fees.taker_fee(filled_contracts, order_price)

        return SimulatedFill(
            filled=True,
            fill_price=order_price,
            filled_contracts=filled_contracts,
            fees=fees,
            slippage=_ZERO,
            fill_model="queue",
        )

    # ------------------------------------------------------------------
    # Realistic fill model
    # ------------------------------------------------------------------

    def _realistic_fill(
        self,
        order: OrderRequest,
        orderbook: OrderbookSnapshot,
        time_in_queue: timedelta,
    ) -> SimulatedFill:
        """Combines queue model with partial fill probability and slippage.

        Adds:
        - Partial fill probability based on available liquidity
        - Price slippage when walking through multiple book levels
        - Additional configurable slippage (slippage_bps)
        """
        levels = self._get_relevant_levels(order, orderbook)
        if not levels:
            return self._no_fill(order, "realistic")

        order_price = order.yes_price

        # Check if we cross the spread (aggressive order)
        best_ask = self._get_best_ask(order, orderbook)
        is_crossing = best_ask is not None and self._price_crosses(order, best_ask)

        if is_crossing and best_ask is not None:
            # Walk the book, simulating fills at each level
            return self._walk_book_fill(order, orderbook)

        # Passive order: use queue model with additional realism
        queue_seconds = time_in_queue.total_seconds()

        # Find total volume at our price level
        matching_volume = 0
        for level in levels:
            if self._price_matches(order, level.price):
                matching_volume += int(level.contracts)

        if matching_volume == 0:
            return self._no_fill(order, "realistic")

        # Queue-based fill probability with diminishing returns
        # Higher volume at our level means more competition
        base_fill_prob = min(
            1.0,
            (queue_seconds * 0.005) / max(1, matching_volume / max(order.count, 1)),
        )

        # Partial fill: scale by liquidity ratio
        liquidity_ratio = min(1.0, matching_volume / max(order.count, 1))
        fill_prob = base_fill_prob * (0.5 + 0.5 * liquidity_ratio)

        filled_contracts = max(0, int(order.count * fill_prob))
        if filled_contracts <= 0:
            return self._no_fill(order, "realistic")

        # Apply configurable slippage
        slippage = order_price * self._slippage_bps
        fill_price = order_price + slippage if order.side == Side.YES else order_price + slippage

        is_maker = order.post_only
        if is_maker:
            fees = self._fees.maker_fee(filled_contracts, fill_price)
        else:
            fees = self._fees.taker_fee(filled_contracts, fill_price)

        return SimulatedFill(
            filled=True,
            fill_price=fill_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            filled_contracts=filled_contracts,
            fees=fees,
            slippage=slippage.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            fill_model="realistic",
        )

    # ------------------------------------------------------------------
    # Book walking (for crossing orders)
    # ------------------------------------------------------------------

    def _walk_book_fill(
        self,
        order: OrderRequest,
        orderbook: OrderbookSnapshot,
    ) -> SimulatedFill:
        """Walk through orderbook levels to simulate aggressive fills."""
        levels = self._get_relevant_levels(order, orderbook)

        # Sort levels by price: ascending for buys (best ask first)
        sorted_levels = sorted(levels, key=lambda lvl: lvl.price)

        remaining = order.count
        total_cost = _ZERO
        total_filled = 0

        for level in sorted_levels:
            if remaining <= 0:
                break

            # Only fill at prices at or better than our limit
            if not self._price_acceptable(order, level.price):
                continue

            fill_at_level = min(remaining, int(level.contracts))
            total_cost += Decimal(fill_at_level) * level.price
            total_filled += fill_at_level
            remaining -= fill_at_level

        if total_filled == 0:
            return self._no_fill(order, "realistic")

        avg_price = (total_cost / Decimal(total_filled)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Add configurable slippage
        slippage = avg_price * self._slippage_bps + (avg_price - order.yes_price)

        fees = self._fees.taker_fee(total_filled, avg_price)

        return SimulatedFill(
            filled=True,
            fill_price=avg_price,
            filled_contracts=total_filled,
            fees=fees,
            slippage=slippage.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            fill_model="realistic",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_relevant_levels(self, order: OrderRequest, ob: OrderbookSnapshot):
        """Get the orderbook levels relevant for this order's side."""
        if order.side == Side.YES:
            return ob.yes_levels
        return ob.no_levels

    def _get_best_ask(self, order: OrderRequest, ob: OrderbookSnapshot) -> Decimal | None:
        """Get the best (lowest) ask price for the order's side."""
        levels = self._get_relevant_levels(order, ob)
        if not levels:
            return None
        return min(lvl.price for lvl in levels)

    def _price_matches(self, order: OrderRequest, level_price: Decimal) -> bool:
        """Check if a level is at our order price."""
        return level_price == order.yes_price

    def _price_crosses(self, order: OrderRequest, ask_price: Decimal) -> bool:
        """Check if our order price crosses (meets or exceeds) the ask."""
        return order.yes_price >= ask_price

    def _price_acceptable(self, order: OrderRequest, level_price: Decimal) -> bool:
        """Check if a level's price is acceptable for our order."""
        # For a buy, we accept prices at or below our limit
        return level_price <= order.yes_price

    def _taker_fill(self, order: OrderRequest, ob: OrderbookSnapshot) -> SimulatedFill:
        """Simulate an immediate taker fill at the best available price."""
        return self._walk_book_fill(order, ob)

    def _no_fill(self, order: OrderRequest, model: str) -> SimulatedFill:
        """Return a no-fill result."""
        return SimulatedFill(
            filled=False,
            fill_price=order.yes_price,
            filled_contracts=0,
            fees=_ZERO,
            slippage=_ZERO,
            fill_model=model,
        )
