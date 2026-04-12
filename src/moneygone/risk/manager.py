"""Risk manager -- orchestrates all pre- and post-trade risk checks.

Central entry point for the execution layer to validate trades and
maintain risk state.  Composes RiskLimits, PortfolioTracker,
DrawdownMonitor, and ExposureCalculator into a single interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

import structlog

from moneygone.config import RiskConfig
from moneygone.exchange.types import Fill, Settlement
from moneygone.risk.capital_governor import CapitalGovernor
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
_TAIL_LOW = Decimal("0.15")
_TAIL_HIGH = Decimal("0.85")


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
    """Realized PnL delta tracked for the current session."""

    n_positions: int
    """Number of active positions."""

    circuit_breaker_active: bool
    """True if drawdown circuit breaker is triggered."""


@dataclass(frozen=True)
class CapitalView:
    """Live capital inputs shared across engine and closer strategies."""

    bankroll: Decimal
    current_equity: Decimal
    available_cash: Decimal
    total_exposure: Decimal
    reserved_exposure: Decimal
    equity_source: str
    paused: bool
    pause_reasons: dict[str, str]


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
        capital_governor: CapitalGovernor | None = None,
    ) -> None:
        self._config = risk_config
        self._limits = risk_limits
        self._portfolio = portfolio
        self._drawdown = drawdown_monitor
        self._exposure = exposure_calculator
        self._categories = categories or {}
        self._capital = capital_governor
        self._daily_pnl = _ZERO
        self._last_realized_pnl = self._portfolio.realized_pnl
        self._session_date: date = datetime.now(timezone.utc).date()

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
        if self.is_trading_paused():
            pause_reasons = self.pause_reasons
            pause_text = "; ".join(
                f"{source}: {reason}" for source, reason in sorted(pause_reasons.items())
            )
            return RiskCheckResult(
                approved=False,
                adjusted_size=None,
                rejection_reason=f"Trading paused -- {pause_text}",
                limit_triggered="global_trading_pause",
            )

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
        equity = self.get_capital_view().current_equity
        self._drawdown.track(equity)

        self._sync_daily_pnl_from_portfolio()

        logger.debug(
            "post_trade_updated",
            ticker=fill.ticker,
            equity=str(equity),
            drawdown=round(self._drawdown.current_drawdown(), 4),
        )

    def post_settlement_update(self, settlement: Settlement) -> None:
        """Update all risk state after a market settlement is received."""
        self._portfolio.on_settlement(settlement)

        equity = self.get_capital_view().current_equity
        self._drawdown.track(equity)
        self._sync_daily_pnl_from_portfolio()

        logger.debug(
            "post_settlement_updated",
            ticker=settlement.ticker,
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
            available_cash = self.get_capital_view().available_cash
            reserved_snapshot = (
                self._capital.snapshot() if self._capital is not None else None
            )
            if reserved_snapshot is not None:
                total_exposure += reserved_snapshot.total_reserved
                tail_exposure += reserved_snapshot.tail_reserved
                for category, dollars in reserved_snapshot.reserved_by_category.items():
                    cat_exposure[category] = cat_exposure.get(category, _ZERO) + dollars
        else:
            portfolio_state = self._build_portfolio_state()
            equity = portfolio_state.current_equity
            total_exposure = portfolio_state.total_exposure
            cat_exposure = portfolio_state.category_exposure
            tail_exposure = portfolio_state.tail_exposure
            available_cash = portfolio_state.available_cash

        exposure_pct = (
            float(total_exposure / equity) if equity > _ZERO else 0.0
        )
        tail_pct = (
            float(tail_exposure / equity) if equity > _ZERO else 0.0
        )

        return RiskSummary(
            total_equity=equity.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            cash=available_cash.quantize(
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

    def get_capital_view(self) -> CapitalView:
        """Return the current shared capital view for sizing and execution."""
        portfolio_state = self._build_portfolio_state()
        snapshot = (
            self._capital.snapshot() if self._capital is not None else None
        )
        equity_snapshot = self._portfolio.get_effective_equity_snapshot()
        return CapitalView(
            bankroll=portfolio_state.bankroll,
            current_equity=portfolio_state.current_equity,
            available_cash=portfolio_state.available_cash,
            total_exposure=portfolio_state.total_exposure,
            reserved_exposure=portfolio_state.reserved_exposure,
            equity_source=equity_snapshot.source,
            paused=bool(snapshot and snapshot.paused),
            pause_reasons=dict(snapshot.pause_reasons) if snapshot else {},
        )

    def reserve_trade_intent(
        self,
        key: str,
        *,
        owner: str,
        ticker: str,
        category: str,
        contracts: int,
        price: Decimal,
    ) -> bool:
        """Reserve capital for a strategy before an order is submitted."""
        if self._capital is None:
            return True
        available_cash = self.get_capital_view().available_cash
        return self._capital.reserve_intent(
            key,
            owner=owner,
            ticker=ticker,
            category=category or "unknown",
            contracts=contracts,
            price=price,
            available_cash=available_cash,
        )

    def release_trade_intent(self, key: str) -> None:
        """Release a previously reserved order intent."""
        if self._capital is None:
            return
        self._capital.release(key)

    def sync_open_order_reservations(
        self,
        orders: list,
        *,
        category_lookup: dict[str, str] | None = None,
    ) -> None:
        """Rebuild reserved capital from the currently tracked open orders."""
        if self._capital is None:
            return
        merged_lookup = dict(self._categories)
        if category_lookup:
            merged_lookup.update(category_lookup)
        self._capital.sync_open_orders(orders, category_lookup=merged_lookup)

    def pause_trading(self, source: str, reason: str) -> None:
        """Pause all new trading from all strategy paths."""
        if self._capital is None:
            return
        self._capital.pause_trading(source, reason)

    def resume_trading(self, source: str) -> None:
        """Resume trading for a previously paused source."""
        if self._capital is None:
            return
        self._capital.resume_trading(source)

    def is_trading_paused(self) -> bool:
        """Return True when any shared pause source is active."""
        if self._capital is None:
            return False
        return self._capital.snapshot().paused

    @property
    def pause_reasons(self) -> dict[str, str]:
        """Current shared trading pause reasons."""
        if self._capital is None:
            return {}
        return self._capital.snapshot().pause_reasons

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_portfolio_state(self) -> PortfolioState:
        """Build a PortfolioState snapshot from current tracker state."""
        positions: dict[str, int] = {}
        gross_positions: dict[str, int] = {}
        position_costs: dict[str, Decimal] = {}
        tail_exposure = _ZERO
        for ticker, pos in self._portfolio.positions.items():
            positions[ticker] = pos.net_count
            gross_positions[ticker] = pos.yes_count + pos.no_count
            position_costs[ticker] = pos.cost_basis
            if pos.yes_count > 0:
                yes_avg_cost = pos.yes_cost_basis / Decimal(pos.yes_count)
                if yes_avg_cost < _TAIL_LOW or yes_avg_cost > _TAIL_HIGH:
                    tail_exposure += pos.yes_cost_basis
            if pos.no_count > 0:
                no_avg_cost = pos.no_cost_basis / Decimal(pos.no_count)
                if no_avg_cost < _TAIL_LOW or no_avg_cost > _TAIL_HIGH:
                    tail_exposure += pos.no_cost_basis

        reserved_snapshot = (
            self._capital.snapshot() if self._capital is not None else None
        )
        category_exposure = self._portfolio.get_exposure_by_category(
            self._categories
        )
        total_exposure = self._portfolio.get_total_exposure()
        available_cash = self._portfolio.cash
        if reserved_snapshot is not None:
            for ticker, contracts in reserved_snapshot.reserved_contracts_by_ticker.items():
                gross_positions[ticker] = gross_positions.get(ticker, 0) + contracts
            for category, dollars in reserved_snapshot.reserved_by_category.items():
                category_exposure[category] = (
                    category_exposure.get(category, _ZERO) + dollars
                )
            total_exposure += reserved_snapshot.total_reserved
            tail_exposure += reserved_snapshot.tail_reserved
            available_cash = max(_ZERO, available_cash - reserved_snapshot.total_reserved)

        equity_snapshot = self._portfolio.get_effective_equity_snapshot()
        equity = equity_snapshot.equity

        return PortfolioState(
            positions=positions,
            gross_positions=gross_positions,
            position_costs=position_costs,
            category_exposure=category_exposure,
            total_exposure=total_exposure,
            bankroll=equity,
            daily_pnl=self._daily_pnl,
            peak_equity=self._drawdown.peak_equity,
            current_equity=equity,
            tail_exposure=tail_exposure,
            available_cash=available_cash,
            reserved_exposure=(
                reserved_snapshot.total_reserved
                if reserved_snapshot is not None
                else _ZERO
            ),
        )

    def _get_category_tickers(self) -> dict[str, list[str]]:
        """Invert the categories dict to get category -> list of tickers."""
        result: dict[str, list[str]] = {}
        for ticker, cat in self._categories.items():
            result.setdefault(cat, []).append(ticker)
        return result

    def _maybe_reset_daily_pnl(self) -> None:
        """Reset daily PnL if the UTC trading date has rolled over."""
        today = datetime.now(timezone.utc).date()
        if today != self._session_date:
            logger.info(
                "risk.daily_pnl_reset",
                previous_date=self._session_date.isoformat(),
                new_date=today.isoformat(),
                previous_daily_pnl=str(self._daily_pnl),
            )
            self._daily_pnl = _ZERO
            self._limits.reset_daily()
            self._session_date = today

    def reset_daily_pnl(self) -> None:
        """Manually reset daily PnL tracking (e.g. at session start)."""
        self._daily_pnl = _ZERO
        self._limits.reset_daily()
        self._session_date = datetime.now(timezone.utc).date()
        logger.info("risk.daily_pnl_manual_reset")

    def _sync_daily_pnl_from_portfolio(self) -> None:
        """Accumulate realized PnL deltas into the session daily PnL."""
        self._maybe_reset_daily_pnl()
        current_realized = self._portfolio.realized_pnl
        realized_delta = current_realized - self._last_realized_pnl
        if realized_delta != _ZERO:
            self._daily_pnl += realized_delta
            self._limits.update_daily_pnl(realized_delta)
        self._last_realized_pnl = current_realized
