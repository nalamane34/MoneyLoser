"""Kalshi fee calculator implementing the exchange's exact fee schedule.

Kalshi charges fees only to takers (market orders and crossing limit orders).
Resting limit orders that provide liquidity (makers) pay zero fees.

Taker fee formula
-----------------
    fee_per_contract = min(0.07 * price * (1 - price), 0.02)

where ``price`` is expressed in dollars (0.0 to 1.0 range, e.g. 0.65 for 65 cents).

The quadratic term ``price * (1 - price)`` peaks at price = 0.50 and is
symmetric around it, so contracts near 50/50 carry the highest fees while
extreme-probability contracts cost less.

The cap at $0.02 per contract means that for any price between ~0.18 and ~0.82,
the effective fee is exactly $0.02/contract.  Outside that band the fee is lower
due to the quadratic formula.
"""

from __future__ import annotations

from decimal import ROUND_UP, Decimal

import structlog

logger = structlog.get_logger(__name__)

# Kalshi fee constants (expressed as Decimal for precision)
_TAKER_FEE_RATE = Decimal("0.07")
_FEE_CAP_PER_CONTRACT = Decimal("0.02")
_ZERO = Decimal("0")
_ONE = Decimal("1")


class KalshiFeeCalculator:
    """Calculates Kalshi exchange fees for taker and maker orders.

    All monetary values use ``Decimal`` for precision.  Prices are expected
    in the 0-to-1 dollar range (e.g. ``Decimal("0.65")`` for 65 cents).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def taker_fee(self, contracts: int, price: Decimal) -> Decimal:
        """Total taker fee for a trade of *contracts* at *price*.

        Parameters
        ----------
        contracts:
            Number of contracts in the trade (must be >= 0).
        price:
            Per-contract price in dollars, 0 < price < 1.

        Returns
        -------
        Decimal
            Total fee in dollars, always >= 0.
        """
        if contracts <= 0:
            return _ZERO
        return Decimal(contracts) * self.fee_per_contract(price, is_maker=False)

    def maker_fee(self, contracts: int, price: Decimal) -> Decimal:  # noqa: ARG002
        """Total maker fee for a trade.  Always zero on Kalshi.

        Resting limit orders that provide liquidity are fee-exempt.
        """
        return _ZERO

    def fee_per_contract(self, price: Decimal, *, is_maker: bool = False) -> Decimal:
        """Fee charged per contract at the given price.

        Parameters
        ----------
        price:
            Per-contract price in dollars, 0 < price < 1.
        is_maker:
            ``True`` for resting limit orders (always returns 0).

        Returns
        -------
        Decimal
            Fee per contract in dollars.
        """
        if is_maker:
            return _ZERO
        raw = _TAKER_FEE_RATE * price * (_ONE - price)
        # Cap at $0.02 per contract
        fee = min(raw, _FEE_CAP_PER_CONTRACT)
        # Round to nearest cent (Kalshi works in cents)
        return fee.quantize(Decimal("0.01"), rounding=ROUND_UP)

    def breakeven_edge(self, price: Decimal, *, is_maker: bool = False) -> Decimal:
        """Minimum edge (in probability units) needed to break even after fees.

        For a binary contract priced at *price*, the edge is simply the fee
        per contract expressed as a fraction of the $1 payout.

        Parameters
        ----------
        price:
            Per-contract price in dollars, 0 < price < 1.
        is_maker:
            ``True`` for maker orders (breakeven edge = 0).

        Returns
        -------
        Decimal
            Minimum probability edge required (0-1 scale).
        """
        return self.fee_per_contract(price, is_maker=is_maker)
