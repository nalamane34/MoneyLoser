"""Pre-trade risk limit checks.

Enforces hard limits on position size, category concentration, total
exposure, daily loss, and drawdown before any order is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from moneygone.config import RiskConfig

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedTrade:
    """A trade that needs risk approval before submission."""

    ticker: str
    category: str
    side: str
    action: str
    contracts: int
    price: Decimal


@dataclass(frozen=True)
class RiskCheckResult:
    """Outcome of a pre-trade risk check."""

    approved: bool
    """True if the trade is fully or partially approved."""

    adjusted_size: int | None
    """Reduced contract count if partially approved, None if fully approved or rejected."""

    rejection_reason: str | None
    """Human-readable rejection reason, None if approved."""

    limit_triggered: str | None
    """Name of the limit that caused rejection or adjustment, None if clean pass."""


@dataclass
class PortfolioState:
    """Snapshot of portfolio state needed for risk checks."""

    positions: dict[str, int] = field(default_factory=dict)
    """ticker -> net contract count (positive = long, negative = short)."""

    position_costs: dict[str, Decimal] = field(default_factory=dict)
    """ticker -> total cost basis."""

    category_exposure: dict[str, Decimal] = field(default_factory=dict)
    """category -> total dollar exposure."""

    total_exposure: Decimal = _ZERO
    """Sum of all position costs."""

    bankroll: Decimal = _ZERO
    """Total equity (cash + unrealized PnL)."""

    daily_pnl: Decimal = _ZERO
    """Realized + unrealized PnL for the current day."""

    peak_equity: Decimal = _ZERO
    """Highest equity value observed."""

    current_equity: Decimal = _ZERO
    """Current equity value."""


# ---------------------------------------------------------------------------
# Risk limits engine
# ---------------------------------------------------------------------------


class RiskLimits:
    """Pre-trade risk check engine enforcing hard limits from RiskConfig.

    All checks are synchronous and designed to be called on every trade
    before order submission.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._daily_pnl = _ZERO
        self._peak_equity = _ZERO

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Run all limit checks on a proposed trade.

        Returns the first failing check.  If a limit would be breached
        but can be partially satisfied with fewer contracts, returns
        the most restrictive adjusted size across all checks.
        """
        checks = [
            self._check_contract_price,
            self._check_daily_loss,
            self._check_drawdown,
            self._check_position_limit,
            self._check_category_concentration,
            self._check_total_exposure,
            self._check_tail_exposure,
        ]

        # Track the most restrictive partial approval
        best_adjusted: int | None = None
        best_limit: str | None = None

        for check_fn in checks:
            result = check_fn(proposed, portfolio)
            if not result.approved:
                logger.info(
                    "risk_limit_triggered",
                    ticker=proposed.ticker,
                    limit=result.limit_triggered,
                    reason=result.rejection_reason,
                )
                return result
            if result.adjusted_size is not None:
                if best_adjusted is None or result.adjusted_size < best_adjusted:
                    best_adjusted = result.adjusted_size
                    best_limit = result.limit_triggered

        return RiskCheckResult(
            approved=True,
            adjusted_size=best_adjusted,
            rejection_reason=None,
            limit_triggered=best_limit,
        )

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update the running daily PnL tracker."""
        self._daily_pnl += pnl

    def reset_daily(self) -> None:
        """Reset daily PnL at the start of a new trading day."""
        self._daily_pnl = _ZERO
        logger.info("daily_pnl_reset")

    def check_drawdown(
        self, current_equity: Decimal, peak_equity: Decimal
    ) -> bool:
        """Return True if drawdown exceeds the configured maximum.

        Updates the internal peak equity tracker as a side effect.
        """
        if peak_equity > self._peak_equity:
            self._peak_equity = peak_equity

        if self._peak_equity <= _ZERO:
            return False

        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return float(drawdown) >= self._config.max_drawdown_pct

    # ------------------------------------------------------------------
    # Individual limit checks
    # ------------------------------------------------------------------

    def _check_contract_price(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,  # noqa: ARG002
    ) -> RiskCheckResult:
        """Reject contracts priced outside the allowed range."""
        price_f = float(proposed.price)
        if price_f < self._config.min_contract_price:
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Contract price {price_f:.2f} below minimum "
                    f"{self._config.min_contract_price:.2f}"
                ),
                limit_triggered="min_contract_price",
            )
        if price_f > self._config.max_contract_price:
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Contract price {price_f:.2f} above maximum "
                    f"{self._config.max_contract_price:.2f}"
                ),
                limit_triggered="max_contract_price",
            )
        return _APPROVED

    def _check_daily_loss(
        self,
        proposed: ProposedTrade,  # noqa: ARG002
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Halt trading if daily loss limit is breached."""
        if portfolio.bankroll <= _ZERO:
            return _APPROVED

        daily_loss_pct = float(-portfolio.daily_pnl / portfolio.bankroll)
        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit "
                    f"{self._config.daily_loss_limit_pct:.2%}"
                ),
                limit_triggered="daily_loss_limit",
            )
        return _APPROVED

    def _check_drawdown(
        self,
        proposed: ProposedTrade,  # noqa: ARG002
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Halt trading if max drawdown is breached."""
        if self.check_drawdown(portfolio.current_equity, portfolio.peak_equity):
            peak = self._peak_equity
            dd = float((peak - portfolio.current_equity) / peak) if peak > _ZERO else 0.0
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Drawdown {dd:.2%} exceeds maximum {self._config.max_drawdown_pct:.2%}"
                ),
                limit_triggered="max_drawdown",
            )
        return _APPROVED

    def _check_position_limit(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Limit contracts in any single market."""
        current = portfolio.positions.get(proposed.ticker, 0)
        new_total = abs(current) + proposed.contracts

        if new_total > self._config.max_position_per_market:
            allowed = self._config.max_position_per_market - abs(current)
            if allowed > 0:
                return RiskCheckResult(
                    approved=True,
                    adjusted_size=allowed,
                    rejection_reason=None,
                    limit_triggered="max_position_per_market",
                )
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Position in {proposed.ticker} would be {new_total} contracts, "
                    f"max is {self._config.max_position_per_market}"
                ),
                limit_triggered="max_position_per_market",
            )
        return _APPROVED

    def _check_category_concentration(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Limit exposure concentration in a single category."""
        if portfolio.bankroll <= _ZERO:
            return _APPROVED

        trade_cost = Decimal(proposed.contracts) * proposed.price
        current_cat = portfolio.category_exposure.get(proposed.category, _ZERO)
        new_cat = current_cat + trade_cost
        cat_pct = float(new_cat / portfolio.bankroll)

        if cat_pct > self._config.max_position_per_category_pct:
            # Calculate how many contracts fit within limit
            max_cat_dollar = Decimal(
                str(self._config.max_position_per_category_pct)
            ) * portfolio.bankroll
            remaining = max_cat_dollar - current_cat
            if remaining > _ZERO and proposed.price > _ZERO:
                allowed = int(remaining / proposed.price)
                if allowed > 0:
                    return RiskCheckResult(
                        approved=True,
                        adjusted_size=allowed,
                        rejection_reason=None,
                        limit_triggered="max_position_per_category_pct",
                    )
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Category '{proposed.category}' exposure would be {cat_pct:.2%}, "
                    f"max is {self._config.max_position_per_category_pct:.2%}"
                ),
                limit_triggered="max_position_per_category_pct",
            )
        return _APPROVED

    def _check_total_exposure(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Limit total portfolio exposure as fraction of bankroll."""
        if portfolio.bankroll <= _ZERO:
            return _APPROVED

        trade_cost = Decimal(proposed.contracts) * proposed.price
        new_total = portfolio.total_exposure + trade_cost
        total_pct = float(new_total / portfolio.bankroll)

        if total_pct > self._config.max_total_exposure_pct:
            max_dollar = Decimal(
                str(self._config.max_total_exposure_pct)
            ) * portfolio.bankroll
            remaining = max_dollar - portfolio.total_exposure
            if remaining > _ZERO and proposed.price > _ZERO:
                allowed = int(remaining / proposed.price)
                if allowed > 0:
                    return RiskCheckResult(
                        approved=True,
                        adjusted_size=allowed,
                        rejection_reason=None,
                        limit_triggered="max_total_exposure_pct",
                    )
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Total exposure would be {total_pct:.2%}, "
                    f"max is {self._config.max_total_exposure_pct:.2%}"
                ),
                limit_triggered="max_total_exposure_pct",
            )
        return _APPROVED

    def _check_tail_exposure(
        self,
        proposed: ProposedTrade,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Cap exposure to tail contracts (priced < 0.15 or > 0.85)."""
        if portfolio.bankroll <= _ZERO:
            return _APPROVED

        price_f = float(proposed.price)
        is_tail = price_f < 0.15 or price_f > 0.85

        if not is_tail:
            return _APPROVED

        # Compute current tail exposure (approximation from portfolio state)
        # The full calculation uses ExposureCalculator; here we do a quick check
        trade_cost = Decimal(proposed.contracts) * proposed.price
        # Assume portfolio.total_exposure includes any existing tail positions
        # For a precise check, the RiskManager orchestrator uses ExposureCalculator
        tail_pct = float(trade_cost / portfolio.bankroll)

        if tail_pct > self._config.max_tail_exposure_pct:
            max_dollar = Decimal(
                str(self._config.max_tail_exposure_pct)
            ) * portfolio.bankroll
            if proposed.price > _ZERO:
                allowed = int(max_dollar / proposed.price)
                if allowed > 0:
                    return RiskCheckResult(
                        approved=True,
                        adjusted_size=allowed,
                        rejection_reason=None,
                        limit_triggered="max_tail_exposure_pct",
                    )
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=(
                    f"Tail contract exposure {tail_pct:.2%} exceeds limit "
                    f"{self._config.max_tail_exposure_pct:.2%}"
                ),
                limit_triggered="max_tail_exposure_pct",
            )
        return _APPROVED


# Sentinel for passing checks
_APPROVED = RiskCheckResult(
    approved=True,
    adjusted_size=None,
    rejection_reason=None,
    limit_triggered=None,
)
