"""Exposure analysis across markets, categories, and tail positions.

Provides mark-to-market exposure calculations used by the RiskManager
to enforce concentration and tail-risk limits.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from moneygone.risk.portfolio import LocalPosition

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")

# Tail contract thresholds (prices below/above these are considered tail)
_TAIL_LOW = Decimal("0.15")
_TAIL_HIGH = Decimal("0.85")


class ExposureCalculator:
    """Computes portfolio exposure breakdowns for risk limit checks."""

    def compute_market_exposure(
        self,
        positions: dict[str, LocalPosition],
        market_prices: dict[str, Decimal],
    ) -> dict[str, Decimal]:
        """Compute mark-to-market exposure per market.

        Parameters
        ----------
        positions:
            Current positions keyed by ticker.
        market_prices:
            Current YES mid prices keyed by ticker.

        Returns
        -------
        dict
            ticker -> dollar exposure (sum of YES value + NO value).
        """
        result: dict[str, Decimal] = {}
        for ticker, pos in positions.items():
            price = market_prices.get(ticker, _ZERO)
            yes_exposure = Decimal(pos.yes_count) * price
            no_exposure = Decimal(pos.no_count) * (_ONE - price)
            result[ticker] = yes_exposure + no_exposure
        return result

    def compute_category_exposure(
        self,
        positions: dict[str, LocalPosition],
        categories: dict[str, str],
        market_prices: dict[str, Decimal],
    ) -> dict[str, Decimal]:
        """Compute mark-to-market exposure grouped by category.

        Parameters
        ----------
        positions:
            Current positions keyed by ticker.
        categories:
            Mapping of ticker -> category name.
        market_prices:
            Current YES mid prices keyed by ticker.

        Returns
        -------
        dict
            category -> total dollar exposure in that category.
        """
        market_exp = self.compute_market_exposure(positions, market_prices)
        result: dict[str, Decimal] = {}
        for ticker, exposure in market_exp.items():
            cat = categories.get(ticker, "unknown")
            result[cat] = result.get(cat, _ZERO) + exposure
        return result

    def compute_tail_exposure(
        self,
        positions: dict[str, LocalPosition],
        market_prices: dict[str, Decimal],
    ) -> Decimal:
        """Compute total exposure to tail contracts.

        A tail contract is one priced below 0.15 or above 0.85,
        representing extreme probability events that carry outsized
        risk relative to their expected frequency.

        Parameters
        ----------
        positions:
            Current positions keyed by ticker.
        market_prices:
            Current YES mid prices keyed by ticker.

        Returns
        -------
        Decimal
            Total dollar exposure to tail contracts.
        """
        tail_exposure = _ZERO
        for ticker, pos in positions.items():
            price = market_prices.get(ticker, _ZERO)
            if price < _TAIL_LOW or price > _TAIL_HIGH:
                yes_exposure = Decimal(pos.yes_count) * price
                no_exposure = Decimal(pos.no_count) * (_ONE - price)
                tail_exposure += yes_exposure + no_exposure
        return tail_exposure

    def compute_correlation_penalty(
        self,
        tickers: list[str],
        category: str,  # noqa: ARG002
    ) -> float:
        """Compute a simple penalty factor for correlated positions.

        When multiple positions are in the same category, they are likely
        correlated.  This returns a penalty multiplier (0-1) that should
        reduce position sizes to account for hidden concentration risk.

        Heuristic:
            - 1 position in category: penalty = 1.0 (no reduction)
            - 2 positions: 0.85
            - 3 positions: 0.70
            - 4+ positions: 0.50

        Parameters
        ----------
        tickers:
            Tickers of positions in the same category.
        category:
            Category name (reserved for future use with explicit
            correlation matrices).

        Returns
        -------
        float
            Penalty multiplier, 0.0 to 1.0.
        """
        n = len(tickers)
        if n <= 1:
            return 1.0
        if n == 2:
            return 0.85
        if n == 3:
            return 0.70
        return 0.50
