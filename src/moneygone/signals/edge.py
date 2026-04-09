"""Edge calculation: compare model probabilities against market prices.

The edge calculator determines whether a model's probability estimate
differs enough from the market-implied probability to justify a trade,
after accounting for fees and liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import structlog

from moneygone.exchange.types import OrderbookSnapshot
from moneygone.signals.fees import KalshiFeeCalculator

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True)
class EdgeResult:
    """Result of an edge computation for a single market opportunity."""

    raw_edge: float
    """model_prob - implied_prob (positive means model sees value)."""

    fee_adjusted_edge: float
    """Raw edge minus fee cost, expressed in probability units."""

    implied_probability: float
    """Market-implied probability derived from the target price."""

    model_probability: float
    """Model's estimated true probability."""

    available_liquidity: int
    """Number of contracts resting at the target price level."""

    estimated_fill_rate: float
    """Fraction 0-1 of the order expected to fill (1.0 for IOC at top)."""

    is_actionable: bool
    """True only when edge exceeds threshold AND liquidity is sufficient."""

    side: str
    """'yes' or 'no' -- which side of the market to trade."""

    action: str
    """'buy' or 'sell' -- direction of the trade."""

    target_price: Decimal
    """Price at which the trade would execute."""

    expected_value: Decimal
    """Expected profit per contract after fees (payout is $1)."""


class EdgeCalculator:
    """Computes actionable edge between model predictions and market prices.

    Parameters
    ----------
    fee_calculator:
        Used to deduct fees from the raw edge.
    min_edge_threshold:
        Minimum fee-adjusted edge (probability units) to consider actionable.
    max_edge_sanity:
        If raw edge exceeds this, the signal is likely a model error and is
        rejected (trading principle: if it looks too good, it's wrong).
    min_liquidity:
        Minimum contracts at the target level to consider actionable.
    """

    def __init__(
        self,
        fee_calculator: KalshiFeeCalculator,
        min_edge_threshold: float = 0.02,
        max_edge_sanity: float = 0.30,
        min_liquidity: int = 1,
    ) -> None:
        self._fees = fee_calculator
        self._min_edge = min_edge_threshold
        self._max_edge = max_edge_sanity
        self._min_liq = min_liquidity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_edge(
        self,
        model_prob: float,
        orderbook: OrderbookSnapshot,
        *,
        is_maker: bool = True,
    ) -> EdgeResult:
        """Evaluate edge for both YES-buy and NO-buy, return the better one.

        For YES buy:
            edge = model_prob - yes_ask_price
        For NO buy:
            edge = (1 - model_prob) - no_ask_price

        The side with the higher fee-adjusted edge is returned.
        """
        yes_result = self._compute_side(
            model_prob, orderbook, side="yes", is_maker=is_maker
        )
        no_result = self._compute_side(
            model_prob, orderbook, side="no", is_maker=is_maker
        )

        # Pick the side with higher fee-adjusted edge
        if yes_result.fee_adjusted_edge >= no_result.fee_adjusted_edge:
            return yes_result
        return no_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_side(
        self,
        model_prob: float,
        orderbook: OrderbookSnapshot,
        *,
        side: str,
        is_maker: bool,
    ) -> EdgeResult:
        """Compute edge for a single side (yes or no)."""
        best_level = self._best_executable_level(orderbook, side)
        if best_level is None:
            return self._empty_result(model_prob, side)

        target_price, available_liquidity = best_level
        relevant_prob = model_prob if side == "yes" else 1.0 - model_prob
        action = "buy"

        implied_prob = float(target_price)
        raw_edge = relevant_prob - implied_prob

        # Fee deduction
        fee = self._fees.fee_per_contract(target_price, is_maker=is_maker)
        fee_adjusted_edge = raw_edge - float(fee)

        # Sanity check: edge too large is likely model error
        edge_too_large = abs(raw_edge) > self._max_edge
        liquidity_ok = available_liquidity >= self._min_liq
        edge_sufficient = fee_adjusted_edge >= self._min_edge

        is_actionable = edge_sufficient and liquidity_ok and not edge_too_large

        if edge_too_large:
            logger.warning(
                "edge_sanity_check_failed",
                ticker=orderbook.ticker,
                side=side,
                raw_edge=round(raw_edge, 4),
                max_allowed=self._max_edge,
            )

        # Estimated fill rate: 1.0 for top-of-book IOC, scale down for maker
        estimated_fill_rate = 1.0 if not is_maker else min(1.0, available_liquidity / max(available_liquidity, 1))

        ev = Decimal(str(fee_adjusted_edge)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        return EdgeResult(
            raw_edge=round(raw_edge, 6),
            fee_adjusted_edge=round(fee_adjusted_edge, 6),
            implied_probability=round(implied_prob, 6),
            model_probability=round(model_prob, 6),
            available_liquidity=available_liquidity,
            estimated_fill_rate=round(estimated_fill_rate, 4),
            is_actionable=is_actionable,
            side=side,
            action=action,
            target_price=target_price,
            expected_value=ev,
        )

    def _best_executable_level(
        self,
        orderbook: OrderbookSnapshot,
        side: str,
    ) -> tuple[Decimal, int] | None:
        """Return the best executable ask and its liquidity for ``side``.

        Kalshi orderbooks are bid-only ladders sorted ascending, with the
        best price at the end. The executable ask is derived from the
        opposite side's best bid.
        """
        if side == "yes":
            if not orderbook.no_bids:
                return None
            best_bid = orderbook.no_bids[-1]
        else:
            if not orderbook.yes_bids:
                return None
            best_bid = orderbook.yes_bids[-1]

        target_price = _ONE - best_bid.price
        return target_price, int(best_bid.contracts)

    def _empty_result(self, model_prob: float, side: str) -> EdgeResult:
        """Return a non-actionable result when no levels exist."""
        return EdgeResult(
            raw_edge=0.0,
            fee_adjusted_edge=0.0,
            implied_probability=0.0,
            model_probability=round(model_prob, 6),
            available_liquidity=0,
            estimated_fill_rate=0.0,
            is_actionable=False,
            side=side,
            action="buy",
            target_price=_ZERO,
            expected_value=_ZERO,
        )
