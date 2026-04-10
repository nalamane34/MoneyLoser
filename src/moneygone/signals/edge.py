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

    actionable_reason: str | None = None
    """Primary reason the opportunity was not actionable, if any."""

    edge_sufficient: bool = False
    """Whether fee-adjusted edge met the minimum threshold."""

    liquidity_ok: bool = False
    """Whether available liquidity met the minimum requirement."""

    sanity_ok: bool = True
    """Whether the raw edge passed the sanity cap."""


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
        actionable_reason: str | None = None
        if edge_too_large:
            actionable_reason = "edge_sanity_check"
        elif not liquidity_ok:
            actionable_reason = "insufficient_liquidity"
        elif not edge_sufficient:
            actionable_reason = "edge_below_threshold"

        if edge_too_large:
            logger.warning(
                "edge_sanity_check_failed",
                ticker=orderbook.ticker,
                side=side,
                raw_edge=round(raw_edge, 4),
                max_allowed=self._max_edge,
            )

        estimated_fill_rate = self._estimate_fill_rate(
            orderbook=orderbook,
            side=side,
            target_price=target_price,
            available_liquidity=available_liquidity,
            is_maker=is_maker,
        )

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
            actionable_reason=actionable_reason,
            edge_sufficient=edge_sufficient,
            liquidity_ok=liquidity_ok,
            sanity_ok=not edge_too_large,
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

    def _estimate_fill_rate(
        self,
        *,
        orderbook: OrderbookSnapshot,
        side: str,
        target_price: Decimal,
        available_liquidity: int,
        is_maker: bool,
    ) -> float:
        """Estimate fill probability at the intended execution style.

        IOC orders at the best ask are assumed to fill immediately.
        Passive orders are discounted by spread width and top-of-book depth.
        """
        if not is_maker:
            return 1.0

        same_side_bids = orderbook.yes_bids if side == "yes" else orderbook.no_bids
        same_side_best_bid = same_side_bids[-1].price if same_side_bids else None

        if same_side_best_bid is None:
            spread_ticks = 5
        else:
            spread_ticks = max(
                1,
                int(
                    (
                        (target_price - same_side_best_bid)
                        * Decimal("100")
                    ).to_integral_value(rounding=ROUND_HALF_UP)
                ),
            )

        if spread_ticks <= 1:
            spread_factor = 0.9
        elif spread_ticks == 2:
            spread_factor = 0.7
        elif spread_ticks <= 4:
            spread_factor = 0.5
        else:
            spread_factor = 0.3

        liquidity_factor = min(1.0, 0.5 + (available_liquidity / 100.0))
        return max(0.1, min(0.95, spread_factor * liquidity_factor))

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
            actionable_reason="no_executable_level",
            edge_sufficient=False,
            liquidity_ok=False,
            sanity_ok=True,
        )
