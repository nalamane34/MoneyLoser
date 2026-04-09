"""Portfolio tracker -- maintains local position and cash state.

Keeps an in-memory ledger of positions, cash balance, and PnL that is
updated on every fill and settlement.  Provides a ``sync_with_exchange``
method to reconcile with the exchange's authoritative state.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

from moneygone.exchange.types import (
    Action,
    Fill,
    Position,
    Settlement,
    Side,
)

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass
class LocalPosition:
    """Mutable local position state for a single market."""

    ticker: str
    yes_count: int = 0
    no_count: int = 0
    cost_basis: Decimal = _ZERO
    realized_pnl: Decimal = _ZERO

    @property
    def net_count(self) -> int:
        """Net contract count (yes - no)."""
        return self.yes_count - self.no_count

    @property
    def is_flat(self) -> bool:
        """True if no contracts are held on either side."""
        return self.yes_count == 0 and self.no_count == 0


class PortfolioTracker:
    """Tracks positions, cash balance, and PnL.

    Parameters
    ----------
    initial_cash:
        Starting cash balance.
    """

    def __init__(self, initial_cash: Decimal = _ZERO) -> None:
        self._positions: dict[str, LocalPosition] = {}
        self._cash: Decimal = initial_cash
        self._realized_pnl: Decimal = _ZERO

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, LocalPosition]:
        """All tracked positions (ticker -> LocalPosition)."""
        return dict(self._positions)

    @property
    def cash(self) -> Decimal:
        """Current cash balance."""
        return self._cash

    @property
    def realized_pnl(self) -> Decimal:
        """Total realized PnL across all markets."""
        return self._realized_pnl

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def on_fill(self, fill: Fill) -> None:
        """Update position and cash when a fill is received.

        For a BUY:
            - Cash decreases by fill.count * fill.price
            - Position count increases
        For a SELL:
            - Cash increases by fill.count * fill.price
            - Position count decreases
            - Realized PnL is recorded
        """
        pos = self._positions.setdefault(
            fill.ticker, LocalPosition(ticker=fill.ticker)
        )
        cost = Decimal(fill.count) * fill.price

        if fill.action == Action.BUY:
            if fill.side == Side.YES:
                pos.yes_count += fill.count
            else:
                pos.no_count += fill.count
            pos.cost_basis += cost
            self._cash -= cost
            logger.debug(
                "fill_buy",
                ticker=fill.ticker,
                side=fill.side.value,
                count=fill.count,
                price=str(fill.price),
                cash_remaining=str(self._cash),
            )
        elif fill.action == Action.SELL:
            total_count = pos.yes_count + pos.no_count
            if fill.side == Side.YES:
                if pos.yes_count > 0 and total_count > 0:
                    avg_cost = pos.cost_basis / Decimal(total_count)
                    pnl = cost - avg_cost * Decimal(fill.count)
                    pos.realized_pnl += pnl
                    self._realized_pnl += pnl
                pos.yes_count = max(0, pos.yes_count - fill.count)
            else:
                if pos.no_count > 0 and total_count > 0:
                    avg_cost = pos.cost_basis / Decimal(total_count)
                    pnl = cost - avg_cost * Decimal(fill.count)
                    pos.realized_pnl += pnl
                    self._realized_pnl += pnl
                pos.no_count = max(0, pos.no_count - fill.count)
            self._cash += cost
            # Reduce cost basis proportionally
            total_before = pos.yes_count + pos.no_count + fill.count
            if total_before > 0:
                fraction_sold = Decimal(fill.count) / Decimal(total_before)
                pos.cost_basis -= pos.cost_basis * fraction_sold
            logger.debug(
                "fill_sell",
                ticker=fill.ticker,
                side=fill.side.value,
                count=fill.count,
                price=str(fill.price),
                cash_remaining=str(self._cash),
            )

        # Clean up flat positions
        if pos.is_flat:
            pos.cost_basis = _ZERO

    def on_settlement(self, settlement: Settlement) -> None:
        """Update realized PnL when a market settles.

        Settlement payout is added to cash; cost basis is written off.
        """
        pos = self._positions.get(settlement.ticker)
        payout = settlement.payout

        self._cash += payout
        self._realized_pnl += settlement.revenue

        if pos is not None:
            pos.realized_pnl += settlement.revenue
            pos.yes_count = 0
            pos.no_count = 0
            pos.cost_basis = _ZERO

        logger.info(
            "settlement",
            ticker=settlement.ticker,
            result=settlement.market_result.value,
            payout=str(payout),
            revenue=str(settlement.revenue),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> LocalPosition | None:
        """Get position for a specific ticker, or None if not held."""
        return self._positions.get(ticker)

    def get_total_exposure(self) -> Decimal:
        """Sum of all position cost bases."""
        return sum(
            (p.cost_basis for p in self._positions.values()), _ZERO
        )

    def get_exposure_by_category(
        self, categories: dict[str, str]
    ) -> dict[str, Decimal]:
        """Group exposure by category.

        Parameters
        ----------
        categories:
            Mapping of ticker -> category name.

        Returns
        -------
        dict
            category -> total cost basis in that category.
        """
        result: dict[str, Decimal] = {}
        for ticker, pos in self._positions.items():
            cat = categories.get(ticker, "unknown")
            result[cat] = result.get(cat, _ZERO) + pos.cost_basis
        return result

    def get_equity(self) -> Decimal:
        """Total equity = cash + sum of cost bases (proxy for unrealized).

        For a true mark-to-market, call ``get_marked_equity`` with
        current market prices.
        """
        return self._cash + self.get_total_exposure()

    def get_marked_equity(self, market_prices: dict[str, Decimal]) -> Decimal:
        """Mark-to-market equity using current mid prices.

        Parameters
        ----------
        market_prices:
            ticker -> current yes mid price.

        Returns
        -------
        Decimal
            cash + sum(position_value at current prices).
        """
        unrealized = _ZERO
        for ticker, pos in self._positions.items():
            price = market_prices.get(ticker, _ZERO)
            yes_value = Decimal(pos.yes_count) * price
            no_value = Decimal(pos.no_count) * (_ONE - price)
            unrealized += yes_value + no_value
        return self._cash + unrealized

    async def sync_with_exchange(self, rest_client: object) -> None:
        """Reconcile local state with exchange positions.

        Parameters
        ----------
        rest_client:
            A Kalshi REST client with ``get_positions()`` and
            ``get_balance()`` methods.

        Fetches the authoritative position and balance data from the
        exchange and overwrites local state where discrepancies are found.
        """
        try:
            positions: list[Position] = await rest_client.get_positions()  # type: ignore[attr-defined]
            balance = await rest_client.get_balance()  # type: ignore[attr-defined]

            # Reconcile positions
            exchange_tickers: set[str] = set()
            for expos in positions:
                exchange_tickers.add(expos.ticker)
                local = self._positions.setdefault(
                    expos.ticker, LocalPosition(ticker=expos.ticker)
                )
                if (
                    local.yes_count != expos.yes_count
                    or local.no_count != expos.no_count
                ):
                    logger.warning(
                        "position_reconciliation",
                        ticker=expos.ticker,
                        local_yes=local.yes_count,
                        exchange_yes=expos.yes_count,
                        local_no=local.no_count,
                        exchange_no=expos.no_count,
                    )
                    local.yes_count = expos.yes_count
                    local.no_count = expos.no_count

            # Remove local positions that no longer exist on exchange
            for ticker in list(self._positions.keys()):
                if ticker not in exchange_tickers:
                    pos = self._positions[ticker]
                    if not pos.is_flat:
                        logger.warning(
                            "position_removed_on_sync",
                            ticker=ticker,
                            yes_count=pos.yes_count,
                            no_count=pos.no_count,
                        )
                    del self._positions[ticker]

            # Reconcile cash
            if self._cash != balance.available:  # type: ignore[attr-defined]
                logger.warning(
                    "cash_reconciliation",
                    local=str(self._cash),
                    exchange=str(balance.available),  # type: ignore[attr-defined]
                )
                self._cash = balance.available  # type: ignore[attr-defined]

            logger.info("portfolio_synced", n_positions=len(self._positions))

        except Exception:
            logger.exception("portfolio_sync_failed")
            raise
