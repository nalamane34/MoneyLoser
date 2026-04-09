"""Risk manager -- orchestrates all pre- and post-trade risk checks.

Central entry point for the execution layer to validate trades and
maintain risk state.  Composes RiskLimits, PortfolioTracker,
DrawdownMonitor, and ExposureCalculator into a single interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import structlog

from moneygone.config import RiskConfig
from moneygone.exchange.types import Fill
from moneygone.risk.drawdown import DrawdownMonitor
from moneygone.risk.exposure import ExposureCalculator
from moneygone.risk.portfolio import PortfolioTracker
from moneygone.sizing.risk_limits import (
    PortfolioState,
    ProposedTrade,
    RiskCheckResult,
    RiskLimits,
)

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")


@dataclass(frozen=True)
class RiskSummary:
    """Snapshot of all current risk metrics."""

    total_equity: Decimal
    """Cash + unrealized PnL."""

    cash: Decimal
    """Available cash balance."""

    total_exposure: Decimal
    """Sum of all position costs."""

    exposure_pct: float
    """Total exposure as percentage of equity."""

    category_exposure: dict[str, Decimal]
    """Exposure broken down by category."""

    tail_exposure: Decimal
    """Exposure to tail contracts (priced < 0.15 or > 0.85)."""

    tail_exposure_pct: float
    """Tail exposure as percentage of equity."""

    current_drawdown: float
    """Current drawdown from peak as percentage."""

    max_drawdown_seen: float
    """Worst drawdown observed this session."""

    peak_equity: Decimal
    """Highest equity observed."""

    daily_pnl: Decimal
    """Realized + unrealized PnL for the current day."""

    n_positions: int
    """Number of active positions."""

    circuit_breaker_active: bool
    """True if drawdown circuit breaker is triggered."""


class RiskManager:
    """Orchestrates all risk checks and portfolio updates.

    Parameters
    ----------
    risk_config:
        Risk limit configuration.
    risk_limits:
        Pre-trade risk limit checker.
    portfolio:
        Portfolio state tracker.
    drawdown_monitor:
        Drawdown tracker for circuit breaker logic.
    exposure_calculator:
        Exposure breakdown calculator.
    categories:
        Mapping of ticker -> category for concentration checks.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        risk_limits: RiskLimits,
        portfolio: PortfolioTracker,
        drawdown_monitor: DrawdownMonitor,
        exposure_calculator: ExposureCalculator,
        categories: dict[str, str] | None = None,
    ) -> None:
        self._config = risk_config
        self._limits = risk_limits
        self._portfolio = portfolio
        self._drawdown = drawdown_monitor
        self._exposure = exposure_calculator
        self._categories = categories or {}
        self._daily_pnl = _ZERO

    # ------------------------------------------------------------------
    # Pre-trade
    # ------------------------------------------------------------------

    def pre_trade_check(self, proposed: ProposedTrade) -> RiskCheckResult:
        """Run all risk checks on a proposed trade.

        Checks are run in order; the first failure is returned.
        If all pass, returns approved.

        Parameters
        ----------
        proposed:
            The trade to validate.

        Returns
        -------
        RiskCheckResult
            Approved (possibly with adjusted size) or rejected.
        """
        # 1. Circuit breaker check
        if self.check_circuit_breakers():
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason="Circuit breaker active -- all trading halted",
                limit_triggered="circuit_breaker",
            )

        # 2. Build portfolio state snapshot for limit checks
        portfolio_state = self._build_portfolio_state()

        # 3. Run limit checks
        result = self._limits.check(proposed, portfolio_state)
        if not result.approved and result.adjusted_size is None:
            return result

        # 4. Check correlation penalty if we have category info
        if proposed.category and proposed.category in self._get_category_tickers():
            category_tickers = self._get_category_tickers()[proposed.category]
            penalty = self._exposure.compute_correlation_penalty(
                category_tickers, proposed.category
            )
            if penalty < 1.0 and result.adjusted_size is not None:
                adjusted = int(result.adjusted_size * penalty)
                if adjusted <= 0:
                    return RiskCheckResult(
                        approved=False,
                        adjusted_size=None,
                        rejection_reason=(
                            f"Correlation penalty ({penalty:.2f}) reduced size to zero"
                        ),
                        limit_triggered="correlation_penalty",
                    )
                return RiskCheckResult(
                    approved=True,
                    adjusted_size=adjusted,
                    rejection_reason=None,
                    limit_triggered="correlation_penalty",
                )

        return result

    # ------------------------------------------------------------------
    # Post-trade
    # ------------------------------------------------------------------

    def post_trade_update(self, fill: Fill) -> None:
        """Update all risk state after a fill is received.

        Parameters
        ----------
        fill:
            The fill to process.
        """
        # Update portfolio
        self._portfolio.on_fill(fill)

        # Update drawdown tracking
        equity = self._portfolio.get_equity()
        self._drawdown.track(equity)

        # Update daily PnL tracking in risk limits
        self._limits.update_daily_pnl(
            fill.price * Decimal(fill.count)
            if fill.action.value == "sell"
            else _ZERO
        )

        logger.debug(
            "post_trade_updated",
            ticker=fill.ticker,
            equity=str(equity),
            drawdown=round(self._drawdown.current_drawdown(), 4),
        )

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def check_circuit_breakers(self) -> bool:
        """Check if any circuit breaker condition is met.

        Returns
        -------
        bool
            True if trading should be halted.
        """
        return self._drawdown.is_circuit_breaker_triggered(
            self._config.max_drawdown_pct
        )

    # ------------------------------------------------------------------
    # Risk summary
    # ------------------------------------------------------------------

    def get_risk_summary(
        self,
        market_prices: dict[str, Decimal] | None = None,
    ) -> RiskSummary:
        """Compute a full snapshot of current risk metrics.

        Parameters
        ----------
        market_prices:
            Current YES mid prices for mark-to-market.  If not provided,
            uses cost-basis equity.

        Returns
        -------
        RiskSummary
            Complete risk state.
        """
        positions = self._portfolio.positions

        if market_prices:
            equity = self._portfolio.get_marked_equity(market_prices)
            market_exp = self._exposure.compute_market_exposure(
                positions, market_prices
            )
            total_exposure = sum(market_exp.values(), _ZERO)
            cat_exposure = self._exposure.compute_category_exposure(
                positions, self._categories, market_prices
            )
            tail_exposure = self._exposure.compute_tail_exposure(
                positions, market_prices
            )
        else:
            equity = self._portfolio.get_equity()
            total_exposure = self._portfolio.get_total_exposure()
            cat_exposure = self._portfolio.get_exposure_by_category(
                self._categories
            )
            tail_exposure = _ZERO

        exposure_pct = (
            float(total_exposure / equity) if equity > _ZERO else 0.0
        )
        tail_pct = (
            float(tail_exposure / equity) if equity > _ZERO else 0.0
        )

        return RiskSummary(
            total_equity=equity.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            cash=self._portfolio.cash.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            total_exposure=total_exposure.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            exposure_pct=round(exposure_pct, 4),
            category_exposure=cat_exposure,
            tail_exposure=tail_exposure.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            tail_exposure_pct=round(tail_pct, 4),
            current_drawdown=round(self._drawdown.current_drawdown(), 4),
            max_drawdown_seen=round(self._drawdown.max_drawdown_seen, 4),
            peak_equity=self._drawdown.peak_equity,
            daily_pnl=self._daily_pnl.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            n_positions=sum(
                1 for p in positions.values() if not p.is_flat
            ),
            circuit_breaker_active=self.check_circuit_breakers(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_portfolio_state(self) -> PortfolioState:
        """Build a PortfolioState snapshot from current tracker state."""
        positions: dict[str, int] = {}
        position_costs: dict[str, Decimal] = {}
        for ticker, pos in self._portfolio.positions.items():
            positions[ticker] = pos.net_count
            position_costs[ticker] = pos.cost_basis

        equity = self._portfolio.get_equity()

        return PortfolioState(
            positions=positions,
            position_costs=position_costs,
            category_exposure=self._portfolio.get_exposure_by_category(
                self._categories
            ),
            total_exposure=self._portfolio.get_total_exposure(),
            bankroll=equity,
            daily_pnl=self._daily_pnl,
            peak_equity=self._drawdown.peak_equity,
            current_equity=equity,
        )

    def _get_category_tickers(self) -> dict[str, list[str]]:
        """Invert the categories dict to get category -> list of tickers."""
        result: dict[str, list[str]] = {}
        for ticker, cat in self._categories.items():
            result.setdefault(cat, []).append(ticker)
        return result
