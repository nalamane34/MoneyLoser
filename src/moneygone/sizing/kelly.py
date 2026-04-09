"""Fractional Kelly criterion position sizer for binary prediction markets.

The Kelly criterion gives the optimal fraction of bankroll to wager when
the edge and odds are known.  For a binary contract:

    Full Kelly: f* = (p * b - q) / b

where:
    p = model probability of the outcome
    q = 1 - p
    b = payout odds (net profit per dollar risked)

For a YES contract at price c:
    b = (1 - c) / c    (risk c to win 1 - c)

For a NO contract at price c:
    b = c / (1 - c)    (risk 1 - c to win c ... but NO price = 1 - yes_price,
                         so effective b for NO at no_price c is (1 - c) / c)

We apply fractional Kelly (default 25%) because:
  - Full Kelly maximizes geometric growth but with extreme variance
  - Half Kelly achieves ~75% of growth with ~50% of variance
  - Quarter Kelly is conservative and appropriate for model uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

import structlog

from moneygone.signals.edge import EdgeResult

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True)
class SizeResult:
    """Output of the Kelly sizer."""

    kelly_fraction: float
    """Raw full-Kelly fraction (can be negative -- meaning don't trade)."""

    adjusted_fraction: float
    """After fractional Kelly, confidence scaling, and regime adjustment."""

    contracts: int
    """Integer number of contracts to trade."""

    dollar_risk: Decimal
    """Maximum dollar loss on this trade (contracts * price)."""

    dollar_ev: Decimal
    """Expected dollar profit (contracts * fee_adjusted_edge)."""

    capped_by: str | None
    """Which limit capped the size, if any.  None means uncapped."""


class KellySizer:
    """Size positions using fractional Kelly criterion.

    Parameters
    ----------
    kelly_fraction:
        Fraction of full Kelly to use (default 0.25 = quarter Kelly).
    max_position_pct:
        Maximum fraction of bankroll allocated to a single trade.
    min_contracts:
        Minimum contracts to bother trading (below this, skip).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.20,
        min_contracts: int = 1,
    ) -> None:
        self._fraction = kelly_fraction
        self._max_pct = max_position_pct
        self._min_contracts = min_contracts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        edge_result: EdgeResult,
        bankroll: Decimal,
        model_confidence: float,
        existing_exposure: Decimal,
        regime_adjustment: float = 1.0,
    ) -> SizeResult:
        """Compute position size for a given edge result.

        Parameters
        ----------
        edge_result:
            The computed edge (must be actionable for nonzero output).
        bankroll:
            Total available bankroll (cash + unrealized).
        model_confidence:
            Model's self-reported confidence, 0-1.  Scales the bet.
        existing_exposure:
            Dollar value of existing positions (used for budget cap).
        regime_adjustment:
            Multiplier from regime detector, 0-1.  0 = don't trade.

        Returns
        -------
        SizeResult
            The recommended position size with all adjustments applied.
        """
        # Guard: zero or negative bankroll
        if bankroll <= _ZERO:
            logger.warning("kelly_zero_bankroll")
            return self._zero_result(reason="zero_bankroll")

        # Guard: regime says don't trade
        if regime_adjustment <= 0.0:
            return self._zero_result(reason="regime_halt")

        # Guard: edge not actionable
        if not edge_result.is_actionable:
            return self._zero_result(reason="not_actionable")

        price = edge_result.target_price
        if price <= _ZERO or price >= _ONE:
            return self._zero_result(reason="invalid_price")

        # ----- Compute full Kelly fraction -----
        p = edge_result.model_probability
        q = 1.0 - p

        if edge_result.side == "yes":
            # Buying YES at price c: odds b = (1-c)/c
            c = float(price)
            b = (1.0 - c) / c
        else:
            # Buying NO at price c: the no_price is c, odds b = (1-c)/c
            c = float(price)
            # For NO side, the relevant prob is (1-p) and payout odds = (1-c)/c
            p = 1.0 - edge_result.model_probability
            q = 1.0 - p
            b = (1.0 - c) / c

        # Full Kelly: f* = (p*b - q) / b
        if b <= 0:
            return self._zero_result(reason="zero_odds")

        full_kelly = (p * b - q) / b

        # Negative Kelly = don't trade
        if full_kelly <= 0:
            return SizeResult(
                kelly_fraction=round(full_kelly, 6),
                adjusted_fraction=0.0,
                contracts=0,
                dollar_risk=_ZERO,
                dollar_ev=_ZERO,
                capped_by="negative_kelly",
            )

        # ----- Apply adjustments -----
        adjusted = full_kelly * self._fraction  # fractional Kelly
        adjusted *= model_confidence             # scale by confidence
        adjusted *= regime_adjustment            # scale by regime

        capped_by: str | None = None

        # Cap at max position percentage of bankroll
        if adjusted > self._max_pct:
            adjusted = self._max_pct
            capped_by = "max_position_pct"

        # ----- Convert fraction to contracts -----
        available_bankroll = bankroll - existing_exposure
        if available_bankroll <= _ZERO:
            return self._zero_result(reason="no_available_bankroll")

        # Dollar amount to risk
        dollar_amount = Decimal(str(adjusted)) * available_bankroll
        # Contracts = dollar_amount / price_per_contract
        contracts_dec = dollar_amount / price
        contracts = int(contracts_dec.to_integral_value(rounding=ROUND_DOWN))

        # Enforce minimum
        if contracts < self._min_contracts:
            contracts = 0

        # Cap by available liquidity
        if contracts > edge_result.available_liquidity:
            contracts = edge_result.available_liquidity
            if capped_by is None:
                capped_by = "available_liquidity"

        dollar_risk = Decimal(contracts) * price
        dollar_risk = dollar_risk.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        dollar_ev = Decimal(contracts) * Decimal(str(edge_result.fee_adjusted_edge))
        dollar_ev = dollar_ev.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.debug(
            "kelly_sized",
            full_kelly=round(full_kelly, 4),
            adjusted=round(adjusted, 4),
            contracts=contracts,
            dollar_risk=str(dollar_risk),
            capped_by=capped_by,
        )

        return SizeResult(
            kelly_fraction=round(full_kelly, 6),
            adjusted_fraction=round(adjusted, 6),
            contracts=contracts,
            dollar_risk=dollar_risk,
            dollar_ev=dollar_ev,
            capped_by=capped_by,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_result(self, reason: str) -> SizeResult:
        return SizeResult(
            kelly_fraction=0.0,
            adjusted_fraction=0.0,
            contracts=0,
            dollar_risk=_ZERO,
            dollar_ev=_ZERO,
            capped_by=reason,
        )
