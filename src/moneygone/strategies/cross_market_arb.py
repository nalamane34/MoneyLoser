"""Cross-market arbitrage strategy for Kalshi.

Detects and exploits mathematically inconsistent pricing across related
contracts.  Three arbitrage types are supported:

1. **Threshold ordering** -- contracts on the same underlying with different
   numeric thresholds must have monotonically ordered prices.
2. **Mutually exclusive** -- a complete set of outcomes (e.g. Fed rate
   decision buckets) must sum to ~100 cents.
3. **Complementary** -- YES + NO on the same contract must equal ~100 cents.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import structlog

from moneygone.exchange.types import (
    Action,
    Fill,
    Market,
    MarketStatus,
    Order,
    OrderRequest,
    OrderbookSnapshot,
    Side,
    TimeInForce,
)

if TYPE_CHECKING:
    from moneygone.execution.order_manager import OrderManager
    from moneygone.exchange.rest_client import KalshiRestClient
    from moneygone.signals.fees import KalshiFeeCalculator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")
_CENT = Decimal("0.01")

# Regex patterns for extracting numeric thresholds from tickers.
#
# Real Kalshi ticker formats use a -T{number} suffix for thresholds:
#   KXTEMPNYCH-26APR0909-T47.99  -> threshold 47.99
#   KXA100MON-26APR0912-T90.99   -> threshold 90.99
#   KXHIGHNY-26APR09-T80         -> threshold 80
#
# The primary pattern is -T{number} at the end of the ticker.
# Legacy patterns are kept as fallbacks.
_THRESHOLD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-T(\d+(?:\.\d+)?)$"),             # ...-T47.99 (end of ticker, primary)
    re.compile(r"[-_]T(\d+(?:\.\d+)?)\b"),         # ...-T90, ..._T90 (mid-ticker)
    re.compile(r"[-_](\d+(?:\.\d+)?)[-_]?T\b"),   # ...-80000-T (legacy)
    re.compile(r"ABOVE[-_](\d+(?:\.\d+)?)"),       # ...ABOVE-4200 (legacy)
    re.compile(r"BELOW[-_](\d+(?:\.\d+)?)"),       # ...BELOW-4200 (legacy)
    re.compile(r"[-_](\d{3,})(?:[-_]|$)"),         # ...-80000- (3+ digit numbers, legacy)
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbLeg:
    """A single leg in an arbitrage trade."""

    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy"
    price: Decimal
    contracts: int


@dataclass(frozen=True)
class ArbOpportunity:
    """A detected arbitrage opportunity."""

    arb_type: str  # "threshold_ordering", "mutual_exclusive", "complementary"
    tickers: list[str]
    prices: list[Decimal]
    violation: str  # human-readable description
    theoretical_profit: Decimal  # profit if both sides settle correctly
    net_profit_after_fees: Decimal
    legs: list[ArbLeg]  # what to buy/sell
    confidence: float
    detected_at: datetime


@dataclass
class ArbConfig:
    """Configuration for the cross-market arbitrage scanner."""

    min_profit_after_fees: float = 0.005  # minimum 0.5 cents per contract
    max_contracts_per_leg: int = 50
    scan_interval_seconds: float = 30.0
    max_legs: int = 5  # max legs in multi-leg arb
    threshold_tolerance: float = 0.02  # 2 cents tolerance for ordering violations


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class CrossMarketArbitrage:
    """Detects and exploits mathematically inconsistent pricing across
    related Kalshi contracts.

    Three types of arbitrage:

    1. THRESHOLD ORDERING: For the same underlying event with different
       thresholds, probabilities must be monotonically ordered.
       ``P(BTC > 85K) <= P(BTC > 80K) <= P(BTC > 75K)``
       If violated: buy the underpriced, sell the overpriced.

    2. MUTUALLY EXCLUSIVE: Outcomes that cover all possibilities must sum
       to ~100 cents.
       ``Fed decision: hold + 25bp + 50bp + 75bp ~= 100c``
       If sum < 100c: buy all sides (guaranteed profit).
       If sum > 100c: sell all sides (if possible).

    3. COMPLEMENTARY: YES + NO on the same contract must equal ~100 cents.
       ``If YES 55c + NO 40c = 95c: buy both for guaranteed 5c profit minus
       fees.``
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        fee_calculator: KalshiFeeCalculator,
        order_manager: OrderManager,
        config: ArbConfig,
    ) -> None:
        self._client = rest_client
        self._fees = fee_calculator
        self._orders = order_manager
        self._config = config
        self._running = False
        self._scan_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background scanning loop."""
        if self._running:
            logger.warning("cross_market_arb.already_running")
            return

        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info(
            "cross_market_arb.started",
            scan_interval=self._config.scan_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the background scanning loop."""
        self._running = False
        if self._scan_task is not None:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None
        logger.info("cross_market_arb.stopped")

    async def _scan_loop(self) -> None:
        """Periodically scan for arbitrage opportunities and execute them."""
        while self._running:
            try:
                opportunities = await self.scan_all()
                for opp in opportunities:
                    if opp.net_profit_after_fees >= Decimal(
                        str(self._config.min_profit_after_fees)
                    ):
                        logger.info(
                            "cross_market_arb.executing",
                            arb_type=opp.arb_type,
                            tickers=opp.tickers,
                            net_profit=str(opp.net_profit_after_fees),
                        )
                        try:
                            await self.execute_arb(opp)
                        except Exception:
                            logger.exception(
                                "cross_market_arb.execution_failed",
                                arb_type=opp.arb_type,
                                tickers=opp.tickers,
                            )
            except Exception:
                logger.exception("cross_market_arb.scan_error")

            await asyncio.sleep(self._config.scan_interval_seconds)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def scan_all(self) -> list[ArbOpportunity]:
        """Scan all active markets for arbitrage opportunities.

        Fetches open markets, groups them by event, and runs all three
        arbitrage scanners.

        Returns
        -------
        list[ArbOpportunity]
            Detected opportunities sorted by net profit descending.
        """
        markets = await self._client.get_markets(status="open")
        open_markets = [m for m in markets if m.status == MarketStatus.OPEN]

        if not open_markets:
            return []

        opportunities: list[ArbOpportunity] = []

        # Group by event for threshold and mutual-exclusive checks
        event_groups = self._group_by_event(open_markets)

        for event_ticker, event_markets in event_groups.items():
            threshold_opps = await self.scan_threshold_ordering(
                event_ticker, event_markets
            )
            opportunities.extend(threshold_opps)

            me_opps = await self.scan_mutually_exclusive(
                event_ticker, event_markets
            )
            opportunities.extend(me_opps)

        # Complementary checks run per-market
        comp_opps = await self.scan_complementary(open_markets)
        opportunities.extend(comp_opps)

        # Sort by profitability
        opportunities.sort(key=lambda o: o.net_profit_after_fees, reverse=True)

        logger.info(
            "cross_market_arb.scan_complete",
            total_markets=len(open_markets),
            events_scanned=len(event_groups),
            opportunities_found=len(opportunities),
        )

        return opportunities

    async def scan_threshold_ordering(
        self, event_ticker: str, markets: list[Market]
    ) -> list[ArbOpportunity]:
        """Check that markets with different thresholds on the same event
        are properly ordered.

        Groups markets by event_ticker, extracts numeric thresholds from
        ticker names, and verifies monotonic price ordering.  For "above"
        thresholds the YES price must decrease as the threshold rises; for
        "below" thresholds the YES price must increase.

        Parameters
        ----------
        event_ticker:
            The common event ticker for the market group.
        markets:
            Markets belonging to this event.

        Returns
        -------
        list[ArbOpportunity]
            Threshold ordering violations found.
        """
        # Extract thresholds and pair with markets
        threshold_markets: list[tuple[float, Market]] = []
        for m in markets:
            threshold = self._extract_threshold(m.ticker)
            if threshold is not None:
                threshold_markets.append((threshold, m))

        if len(threshold_markets) < 2:
            return []

        # Sort by threshold ascending
        threshold_markets.sort(key=lambda tm: tm[0])

        opportunities: list[ArbOpportunity] = []
        tolerance = Decimal(str(self._config.threshold_tolerance))

        # For "above" type thresholds: higher threshold => lower YES price
        # Check each adjacent pair
        for i in range(len(threshold_markets) - 1):
            lower_thresh, lower_mkt = threshold_markets[i]
            higher_thresh, higher_mkt = threshold_markets[i + 1]

            # P(above lower_thresh) >= P(above higher_thresh)
            # So lower_mkt.yes_ask should be >= higher_mkt.yes_bid
            # Violation: higher threshold priced ABOVE lower threshold
            if higher_mkt.yes_bid > lower_mkt.yes_ask + tolerance:
                violation_size = higher_mkt.yes_bid - lower_mkt.yes_ask

                # Buy YES on lower threshold (cheap), sell YES on higher (expensive)
                # Actually: buy YES on lower, buy NO on higher (equivalent to selling YES)
                contracts = self._config.max_contracts_per_leg
                legs = [
                    ArbLeg(
                        ticker=lower_mkt.ticker,
                        side="yes",
                        action="buy",
                        price=lower_mkt.yes_ask,
                        contracts=contracts,
                    ),
                    ArbLeg(
                        ticker=higher_mkt.ticker,
                        side="no",
                        action="buy",
                        price=_ONE - higher_mkt.yes_bid,
                        contracts=contracts,
                    ),
                ]

                theoretical = violation_size * contracts
                net_profit = self.calculate_arb_profit(legs, theoretical)

                if net_profit > _ZERO:
                    opp = ArbOpportunity(
                        arb_type="threshold_ordering",
                        tickers=[lower_mkt.ticker, higher_mkt.ticker],
                        prices=[lower_mkt.yes_ask, higher_mkt.yes_bid],
                        violation=(
                            f"P({event_ticker} above {higher_thresh}) at "
                            f"{higher_mkt.yes_bid} > P(above {lower_thresh}) "
                            f"at {lower_mkt.yes_ask}: violation of {violation_size}"
                        ),
                        theoretical_profit=theoretical,
                        net_profit_after_fees=net_profit,
                        legs=legs,
                        confidence=min(1.0, float(violation_size / tolerance)),
                        detected_at=datetime.now(timezone.utc),
                    )
                    opportunities.append(opp)
                    logger.info(
                        "cross_market_arb.threshold_violation",
                        event=event_ticker,
                        lower_thresh=lower_thresh,
                        higher_thresh=higher_thresh,
                        violation_size=str(violation_size),
                    )

        return opportunities

    async def scan_mutually_exclusive(
        self, event_ticker: str, markets: list[Market]
    ) -> list[ArbOpportunity]:
        """Check that mutually exclusive outcomes sum to ~100 cents.

        Uses ``series_ticker`` to group related markets (e.g. all Fed rate
        outcomes).  The sum of YES ask prices should be close to 100 cents.
        If the sum is significantly below 100c, buying all sides guarantees
        profit since exactly one must settle YES.

        Parameters
        ----------
        event_ticker:
            The common event ticker for the market group.
        markets:
            Markets belonging to this event.

        Returns
        -------
        list[ArbOpportunity]
            Mutually exclusive sum violations found.
        """
        # Group by series_ticker to find mutually exclusive sets
        series_groups: dict[str, list[Market]] = {}
        for m in markets:
            if m.series_ticker:
                series_groups.setdefault(m.series_ticker, []).append(m)

        opportunities: list[ArbOpportunity] = []

        for series_ticker, series_markets in series_groups.items():
            if len(series_markets) < 2:
                continue

            # Sum of YES ask prices (cost to buy all YES sides)
            yes_ask_sum = sum(
                (m.yes_ask for m in series_markets if m.yes_ask > _ZERO),
                _ZERO,
            )

            if yes_ask_sum <= _ZERO:
                continue

            # If sum < 1.00, buying all YES sides costs less than the
            # guaranteed $1 payout (one must win)
            if yes_ask_sum < _ONE:
                gap = _ONE - yes_ask_sum
                tolerance = Decimal(str(self._config.threshold_tolerance))

                if gap <= tolerance:
                    continue

                contracts = self._config.max_contracts_per_leg

                # Cap at max_legs
                if len(series_markets) > self._config.max_legs:
                    continue

                legs: list[ArbLeg] = []
                for m in series_markets:
                    if m.yes_ask > _ZERO:
                        legs.append(
                            ArbLeg(
                                ticker=m.ticker,
                                side="yes",
                                action="buy",
                                price=m.yes_ask,
                                contracts=contracts,
                            )
                        )

                theoretical = gap * contracts
                net_profit = self.calculate_arb_profit(legs, theoretical)

                if net_profit > _ZERO:
                    opp = ArbOpportunity(
                        arb_type="mutual_exclusive",
                        tickers=[m.ticker for m in series_markets],
                        prices=[m.yes_ask for m in series_markets],
                        violation=(
                            f"Mutually exclusive set {series_ticker}: "
                            f"sum of YES asks = {yes_ask_sum} < 1.00 "
                            f"(gap = {gap})"
                        ),
                        theoretical_profit=theoretical,
                        net_profit_after_fees=net_profit,
                        legs=legs,
                        confidence=min(1.0, float(gap / Decimal("0.05"))),
                        detected_at=datetime.now(timezone.utc),
                    )
                    opportunities.append(opp)
                    logger.info(
                        "cross_market_arb.mutual_exclusive_violation",
                        series=series_ticker,
                        yes_ask_sum=str(yes_ask_sum),
                        gap=str(gap),
                        num_markets=len(series_markets),
                    )

            # If sum > 1.00, selling all YES sides (buying all NO sides)
            # guarantees profit since at most one YES wins
            elif yes_ask_sum > _ONE:
                # For selling, we need to look at YES bid prices (what we
                # can sell at) -- but Kalshi doesn't support direct selling
                # for non-holders.  Instead: buy NO on all markets.
                no_cost_sum = sum(
                    (_ONE - m.yes_bid for m in series_markets if m.yes_bid > _ZERO),
                    _ZERO,
                )
                # Buying NO on all markets costs no_cost_sum.
                # Exactly (n-1) NO contracts settle YES (pay $1 each).
                n = len(series_markets)
                guaranteed_payout = _ONE * (n - 1)
                gap = guaranteed_payout - no_cost_sum

                if gap <= Decimal(str(self._config.threshold_tolerance)):
                    continue

                if n > self._config.max_legs:
                    continue

                contracts = self._config.max_contracts_per_leg
                legs = []
                for m in series_markets:
                    if m.yes_bid > _ZERO:
                        legs.append(
                            ArbLeg(
                                ticker=m.ticker,
                                side="no",
                                action="buy",
                                price=_ONE - m.yes_bid,
                                contracts=contracts,
                            )
                        )

                theoretical = gap * contracts
                net_profit = self.calculate_arb_profit(legs, theoretical)

                if net_profit > _ZERO:
                    opp = ArbOpportunity(
                        arb_type="mutual_exclusive",
                        tickers=[m.ticker for m in series_markets],
                        prices=[m.yes_bid for m in series_markets],
                        violation=(
                            f"Mutually exclusive set {series_ticker}: "
                            f"sum of YES asks = {yes_ask_sum} > 1.00, "
                            f"NO side gap = {gap}"
                        ),
                        theoretical_profit=theoretical,
                        net_profit_after_fees=net_profit,
                        legs=legs,
                        confidence=min(1.0, float(gap / Decimal("0.05"))),
                        detected_at=datetime.now(timezone.utc),
                    )
                    opportunities.append(opp)
                    logger.info(
                        "cross_market_arb.mutual_exclusive_oversupply",
                        series=series_ticker,
                        yes_ask_sum=str(yes_ask_sum),
                        no_cost_sum=str(no_cost_sum),
                        gap=str(gap),
                    )

        return opportunities

    async def scan_complementary(
        self, markets: list[Market]
    ) -> list[ArbOpportunity]:
        """Check YES + NO = ~100 cents for each market.

        For each market, if ``best_yes_ask + best_no_ask < 100c`` then buying
        both sides guarantees profit (one must settle at $1).  This is rare
        since Kalshi pairs are usually consistent, but worth checking.

        Parameters
        ----------
        markets:
            All open markets to check.

        Returns
        -------
        list[ArbOpportunity]
            Complementary pair violations found.
        """
        opportunities: list[ArbOpportunity] = []
        tolerance = Decimal(str(self._config.threshold_tolerance))

        for m in markets:
            if m.yes_ask <= _ZERO or m.yes_bid <= _ZERO:
                continue

            # NO ask = 1 - YES bid (the cost to buy the NO side)
            no_ask = _ONE - m.yes_bid
            total_cost = m.yes_ask + no_ask

            if total_cost < _ONE - tolerance:
                gap = _ONE - total_cost
                contracts = self._config.max_contracts_per_leg

                legs = [
                    ArbLeg(
                        ticker=m.ticker,
                        side="yes",
                        action="buy",
                        price=m.yes_ask,
                        contracts=contracts,
                    ),
                    ArbLeg(
                        ticker=m.ticker,
                        side="no",
                        action="buy",
                        price=no_ask,
                        contracts=contracts,
                    ),
                ]

                theoretical = gap * contracts
                net_profit = self.calculate_arb_profit(legs, theoretical)

                if net_profit > _ZERO:
                    opp = ArbOpportunity(
                        arb_type="complementary",
                        tickers=[m.ticker],
                        prices=[m.yes_ask, no_ask],
                        violation=(
                            f"YES ask {m.yes_ask} + NO ask {no_ask} = "
                            f"{total_cost} < 1.00 (gap = {gap})"
                        ),
                        theoretical_profit=theoretical,
                        net_profit_after_fees=net_profit,
                        legs=legs,
                        confidence=min(1.0, float(gap / Decimal("0.05"))),
                        detected_at=datetime.now(timezone.utc),
                    )
                    opportunities.append(opp)
                    logger.info(
                        "cross_market_arb.complementary_violation",
                        ticker=m.ticker,
                        yes_ask=str(m.yes_ask),
                        no_ask=str(no_ask),
                        gap=str(gap),
                    )

        return opportunities

    # ------------------------------------------------------------------
    # Profit calculation
    # ------------------------------------------------------------------

    def calculate_arb_profit(
        self, legs: list[ArbLeg], guaranteed_payout: Decimal
    ) -> Decimal:
        """Calculate net profit after fees for an arbitrage trade.

        Parameters
        ----------
        legs:
            The trade legs to execute.
        guaranteed_payout:
            The guaranteed payout if the arb settles correctly
            (e.g. ``gap * contracts``).

        Returns
        -------
        Decimal
            Net profit after subtracting total cost and taker fees.
            Negative means the arb is not profitable after fees.
        """
        total_cost = _ZERO
        total_fees = _ZERO

        for leg in legs:
            leg_cost = leg.price * leg.contracts
            total_cost += leg_cost

            # Assume taker execution for arb (aggressive fills)
            leg_fee = self._fees.taker_fee(leg.contracts, leg.price)
            total_fees += leg_fee

        net_profit = guaranteed_payout - total_cost - total_fees

        return net_profit.quantize(_CENT, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_arb(
        self, opportunity: ArbOpportunity
    ) -> list[Fill]:
        """Execute all legs of an arbitrage trade simultaneously.

        Uses aggressive (IOC) execution on all legs.  Submits all leg orders
        concurrently and collects the results.  If any leg fails to fill,
        the remaining resting orders are cancelled.

        Parameters
        ----------
        opportunity:
            The arbitrage opportunity to execute.

        Returns
        -------
        list[Fill]
            Fills received from successfully executed legs.

        Raises
        ------
        RuntimeError
            If any leg fails to execute and the arb cannot be completed.
        """
        logger.info(
            "cross_market_arb.executing_arb",
            arb_type=opportunity.arb_type,
            tickers=opportunity.tickers,
            legs=len(opportunity.legs),
            net_profit=str(opportunity.net_profit_after_fees),
        )

        # Build order requests for all legs
        order_requests: list[OrderRequest] = []
        for leg in opportunity.legs:
            side = Side.YES if leg.side == "yes" else Side.NO

            # Calculate the yes_price for the order.  For YES legs, the
            # yes_price is the leg price.  For NO legs, yes_price is
            # 1 - leg.price because the API always takes yes_price.
            if side == Side.YES:
                yes_price = leg.price
            else:
                yes_price = _ONE - leg.price

            req = OrderRequest(
                ticker=leg.ticker,
                side=side,
                action=Action.BUY,
                count=leg.contracts,
                yes_price=yes_price,
                time_in_force=TimeInForce.IOC,
            )
            order_requests.append(req)

        # Submit all legs concurrently
        submit_tasks = [
            self._orders.submit_order(req) for req in order_requests
        ]
        results = await asyncio.gather(*submit_tasks, return_exceptions=True)

        # Check for failures
        submitted_orders: list[Order] = []
        failed_legs: list[int] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "cross_market_arb.leg_failed",
                    leg_index=i,
                    ticker=opportunity.legs[i].ticker,
                    error=str(result),
                )
                failed_legs.append(i)
            else:
                submitted_orders.append(result)

        # If any leg failed, cancel all successfully submitted orders
        if failed_legs:
            logger.warning(
                "cross_market_arb.partial_failure_cancelling",
                failed_legs=failed_legs,
                submitted_count=len(submitted_orders),
            )
            cancel_tasks = [
                self._orders.cancel_order(order.order_id)
                for order in submitted_orders
                if order.remaining_count > 0
            ]
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)

            raise RuntimeError(
                f"Arb execution failed: {len(failed_legs)} of "
                f"{len(opportunity.legs)} legs failed"
            )

        # Collect fills for successfully executed legs
        all_fills: list[Fill] = []
        for order in submitted_orders:
            try:
                fills = await self._client.get_fills(
                    ticker=order.ticker, order_id=order.order_id
                )
                all_fills.extend(fills)
            except Exception:
                logger.exception(
                    "cross_market_arb.fill_fetch_failed",
                    order_id=order.order_id,
                )

        logger.info(
            "cross_market_arb.arb_executed",
            arb_type=opportunity.arb_type,
            tickers=opportunity.tickers,
            fills=len(all_fills),
            total_contracts=sum(f.count for f in all_fills),
        )

        return all_fills

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_threshold(self, ticker: str) -> float | None:
        """Parse numeric threshold from ticker string.

        Tries multiple regex patterns to extract the threshold value from
        Kalshi ticker naming conventions.  The primary format uses a
        ``-T{number}`` suffix (e.g. ``-T47.99``).

        Examples
        --------
        >>> arb._extract_threshold("KXTEMPNYCH-26APR0909-T47.99")
        47.99
        >>> arb._extract_threshold("KXA100MON-26APR0912-T90.99")
        90.99
        >>> arb._extract_threshold("KXHIGHNY-26APR09-T80")
        80.0
        >>> arb._extract_threshold("INX-ABOVE-4200")
        4200.0
        """
        for pattern in _THRESHOLD_PATTERNS:
            match = pattern.search(ticker)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def _group_by_event(
        self, markets: list[Market]
    ) -> dict[str, list[Market]]:
        """Group markets by ``event_ticker`` for cross-market comparison.

        Parameters
        ----------
        markets:
            List of markets to group.

        Returns
        -------
        dict[str, list[Market]]
            Mapping from event_ticker to list of markets in that event.
            Events with only a single market are excluded.
        """
        groups: dict[str, list[Market]] = {}
        for m in markets:
            if m.event_ticker:
                groups.setdefault(m.event_ticker, []).append(m)

        # Only keep groups with 2+ markets (can't arb a single market
        # against itself in this context)
        return {k: v for k, v in groups.items() if len(v) >= 2}
