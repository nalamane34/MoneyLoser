"""Simulated exchange for backtesting.

Provides a synchronous exchange simulation that processes orders against
historical orderbook snapshots and tracks portfolio state (positions,
cash, PnL) without any async or network operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from moneygone.exchange.types import (
    Action,
    Fill,
    MarketResult,
    OrderRequest,
    OrderbookSnapshot,
    Settlement,
    Side,
)
from moneygone.execution.simulator import FillSimulator, SimulatedFill
from moneygone.signals.fees import KalshiFeeCalculator

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


# ---------------------------------------------------------------------------
# Simulated portfolio
# ---------------------------------------------------------------------------


@dataclass
class SimulatedPosition:
    """A position in the simulated portfolio."""

    ticker: str
    side: Side
    contracts: int
    avg_price: Decimal
    cost_basis: Decimal


@dataclass
class SimulatedPortfolio:
    """Tracks cash, positions, and PnL for backtesting.

    Unlike PortfolioTracker, this is fully synchronous and does not
    interact with any exchange APIs.

    Parameters
    ----------
    initial_cash:
        Starting cash balance.
    """

    cash: Decimal = Decimal("10000")
    positions: dict[str, SimulatedPosition] = field(default_factory=dict)
    realized_pnl: Decimal = _ZERO
    total_fees: Decimal = _ZERO
    peak_equity: Decimal = _ZERO
    trade_count: int = 0

    def __post_init__(self) -> None:
        self.peak_equity = self.cash

    def get_equity(self) -> Decimal:
        """Total equity: cash + sum of position cost bases."""
        position_value = sum(
            (p.cost_basis for p in self.positions.values()),
            _ZERO,
        )
        return self.cash + position_value

    def get_total_exposure(self) -> Decimal:
        """Total dollar exposure across all positions."""
        return sum(
            (p.cost_basis for p in self.positions.values()),
            _ZERO,
        )

    def get_position_count(self, ticker: str) -> int:
        """Get the net contract count for a ticker."""
        pos = self.positions.get(ticker)
        if pos is None:
            return 0
        return pos.contracts

    def update_peak(self) -> None:
        """Update peak equity if current equity is a new high."""
        equity = self.get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity


# ---------------------------------------------------------------------------
# Simulated exchange
# ---------------------------------------------------------------------------


class SimulatedExchange:
    """Simulates order execution and settlement for backtesting.

    Parameters
    ----------
    initial_cash:
        Starting cash balance for the simulated portfolio.
    fee_calculator:
        Fee calculator for cost estimation.
    """

    def __init__(
        self,
        initial_cash: Decimal = Decimal("10000"),
        fee_calculator: KalshiFeeCalculator | None = None,
    ) -> None:
        self._portfolio = SimulatedPortfolio(cash=initial_cash)
        self._fees = fee_calculator or KalshiFeeCalculator()
        self._fill_id_counter = 0

    @property
    def portfolio(self) -> SimulatedPortfolio:
        """The simulated portfolio state."""
        return self._portfolio

    # ------------------------------------------------------------------
    # Order processing
    # ------------------------------------------------------------------

    def process_order(
        self,
        order: OrderRequest,
        orderbook: OrderbookSnapshot,
        fill_simulator: FillSimulator,
    ) -> SimulatedFill:
        """Process an order against a historical orderbook snapshot.

        Parameters
        ----------
        order:
            The order to simulate.
        orderbook:
            The orderbook state at the time of the order.
        fill_simulator:
            The fill simulation model to use.

        Returns
        -------
        SimulatedFill
            The simulated fill result.
        """
        sim_fill = fill_simulator.simulate_fill(order, orderbook)

        if not sim_fill.filled or sim_fill.filled_contracts <= 0:
            return sim_fill

        # Update portfolio state
        self._apply_fill(order, sim_fill)
        self._portfolio.trade_count += 1
        self._portfolio.update_peak()

        logger.debug(
            "sim_exchange.order_filled",
            ticker=order.ticker,
            side=order.side.value,
            contracts=sim_fill.filled_contracts,
            fill_price=str(sim_fill.fill_price),
            fees=str(sim_fill.fees),
            slippage=str(sim_fill.slippage),
        )

        return sim_fill

    def _apply_fill(self, order: OrderRequest, fill: SimulatedFill) -> None:
        """Apply a simulated fill to the portfolio."""
        ticker = order.ticker
        fill_price = fill.fill_price
        contracts = fill.filled_contracts
        fees = fill.fees

        self._portfolio.total_fees += fees

        if order.action == Action.BUY:
            # Buying: pay price * contracts + fees
            cost = Decimal(contracts) * fill_price + fees
            self._portfolio.cash -= cost

            existing = self._portfolio.positions.get(ticker)
            if existing is not None and existing.side == order.side:
                # Add to existing position
                total_contracts = existing.contracts + contracts
                total_cost = existing.cost_basis + Decimal(contracts) * fill_price
                existing.contracts = total_contracts
                existing.cost_basis = total_cost
                existing.avg_price = total_cost / Decimal(total_contracts)
            else:
                # New position (or reversing)
                self._portfolio.positions[ticker] = SimulatedPosition(
                    ticker=ticker,
                    side=order.side,
                    contracts=contracts,
                    avg_price=fill_price,
                    cost_basis=Decimal(contracts) * fill_price,
                )
        else:
            # Selling: receive price * contracts - fees
            revenue = Decimal(contracts) * fill_price - fees
            self._portfolio.cash += revenue

            existing = self._portfolio.positions.get(ticker)
            if existing is not None:
                # Calculate PnL
                avg_cost = existing.avg_price
                pnl = (fill_price - avg_cost) * Decimal(contracts)
                if existing.side == Side.NO:
                    # For NO positions, profit when NO price goes up (YES price goes down)
                    pnl = (fill_price - avg_cost) * Decimal(contracts)

                self._portfolio.realized_pnl += pnl

                # Reduce position
                remaining = existing.contracts - contracts
                if remaining <= 0:
                    self._portfolio.positions.pop(ticker, None)
                else:
                    existing.contracts = remaining
                    existing.cost_basis = existing.avg_price * Decimal(remaining)

    # ------------------------------------------------------------------
    # Settlement processing
    # ------------------------------------------------------------------

    def process_settlement(
        self,
        ticker: str,
        result: MarketResult,
    ) -> Decimal:
        """Process a market settlement.

        Parameters
        ----------
        ticker:
            The market ticker that settled.
        result:
            The settlement outcome.

        Returns
        -------
        Decimal
            PnL from the settlement.
        """
        pos = self._portfolio.positions.get(ticker)
        if pos is None:
            return _ZERO

        pnl = _ZERO
        contracts = pos.contracts
        cost_basis = pos.cost_basis

        if result in (MarketResult.YES, MarketResult.ALL_YES):
            if pos.side == Side.YES:
                # Long YES wins: payout $1 per contract
                payout = Decimal(contracts) * _ONE
                pnl = payout - cost_basis
                self._portfolio.cash += payout
            else:
                # Long NO loses: payout $0
                pnl = -cost_basis
        elif result in (MarketResult.NO, MarketResult.ALL_NO):
            if pos.side == Side.NO:
                # Long NO wins: payout $1 per contract
                payout = Decimal(contracts) * _ONE
                pnl = payout - cost_basis
                self._portfolio.cash += payout
            else:
                # Long YES loses: payout $0
                pnl = -cost_basis
        elif result == MarketResult.VOIDED:
            # Voided: return cost basis
            self._portfolio.cash += cost_basis
            pnl = _ZERO
        else:
            logger.warning(
                "sim_exchange.unknown_settlement_result",
                ticker=ticker,
                result=result.value,
            )

        self._portfolio.realized_pnl += pnl
        self._portfolio.positions.pop(ticker, None)
        self._portfolio.update_peak()

        logger.info(
            "sim_exchange.settlement",
            ticker=ticker,
            result=result.value,
            contracts=contracts,
            pnl=str(pnl),
            equity=str(self._portfolio.get_equity()),
        )

        return pnl

    def get_equity(self) -> Decimal:
        """Current portfolio equity."""
        return self._portfolio.get_equity()
