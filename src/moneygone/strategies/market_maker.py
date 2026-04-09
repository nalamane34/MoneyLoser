"""Market-making strategy for illiquid Kalshi contracts.

Posts two-sided quotes (bid and ask) on contracts with wide spreads to
capture the bid-ask spread.  Resting limit orders on Kalshi are fee-exempt,
making this strategy viable even on thin margins.
"""

from __future__ import annotations

import asyncio
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
    OrderbookLevel,
    OrderbookSnapshot,
    OrderStatus,
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
_MIN_PRICE = Decimal("0.01")  # 1 cent
_MAX_PRICE = Decimal("0.99")  # 99 cents

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMQuote:
    """A two-sided quote to be posted on a market."""

    ticker: str
    bid_price: Decimal  # price to buy YES
    ask_price: Decimal  # price to sell YES (= buy NO at 1-ask)
    spread: Decimal
    size: int  # contracts per side
    edge_per_contract: Decimal  # half-spread minus any residual risk
    fair_value: Decimal  # estimated fair price


@dataclass
class MMState:
    """Live state for a single market being made."""

    ticker: str
    bid_order: Order | None = None
    ask_order: Order | None = None
    net_position: int = 0  # positive = long YES, negative = short YES
    realized_spread_pnl: Decimal = _ZERO
    num_bid_fills: int = 0
    num_ask_fills: int = 0


@dataclass
class MMConfig:
    """Configuration for the market-making strategy."""

    min_spread: float = 0.05  # 5 cents minimum spread to make
    spread_fraction: float = 0.8  # fraction of current spread to capture
    max_inventory: int = 50  # max net position per market
    skew_per_contract: float = 0.005  # 0.5 cent skew per contract of inventory
    max_markets: int = 10  # max simultaneous markets
    quote_refresh_seconds: float = 30.0  # how often to refresh quotes
    min_hours_to_expiry: float = 24.0  # don't make markets expiring soon
    max_volume_24h: int = 5000  # too competitive above this
    size_per_side: int = 10  # contracts per side
    pull_on_volatility: bool = True  # pull quotes during high vol


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MarketMaker:
    """Posts two-sided quotes on illiquid Kalshi contracts to capture the
    spread.

    Key advantages on Kalshi:

    - Maker (resting limit) orders are FEE-EXEMPT
    - Many markets have 5--15 cent spreads
    - Low competition on illiquid contracts

    Strategy:

    1. Identify markets with wide spreads and low volume.
    2. Estimate fair value (midpoint or model-based).
    3. Post bid at ``fair_value - half_spread`` and ask at
       ``fair_value + half_spread``.
    4. When one side fills, adjust the other side.
    5. Manage inventory: if too long/short, skew quotes to reduce position.

    Risk management:

    - Max position per market (net long or short).
    - Inventory skew (if long, lower bid/ask to encourage sells).
    - Pull quotes on high volatility or news events.
    - Max number of markets to make simultaneously.
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        order_manager: OrderManager,
        fee_calculator: KalshiFeeCalculator,
        config: MMConfig,
    ) -> None:
        self._client = rest_client
        self._orders = order_manager
        self._fees = fee_calculator
        self._config = config
        self._states: dict[str, MMState] = {}
        self._running = False
        self._refresh_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, tickers: list[str] | None = None) -> None:
        """Start market making on specified tickers.

        Initialises state for each ticker, computes initial quotes, and
        starts the background refresh loop.

        Parameters
        ----------
        tickers:
            Market tickers to make.  Capped at ``config.max_markets``.
        """
        if self._running:
            logger.warning("market_maker.already_running")
            return

        if tickers is None:
            tickers = await self.select_markets()

        active_tickers = tickers[: self._config.max_markets]

        for ticker in active_tickers:
            self._states[ticker] = MMState(ticker=ticker)

        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        # Post initial quotes
        for ticker in active_tickers:
            try:
                await self.update_quotes(ticker)
            except Exception:
                logger.exception(
                    "market_maker.initial_quote_failed", ticker=ticker
                )

        logger.info(
            "market_maker.started",
            tickers=active_tickers,
            num_markets=len(active_tickers),
        )

    async def stop(self) -> None:
        """Cancel all resting orders and stop."""
        self._running = False

        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

        # Cancel all outstanding orders
        for ticker in list(self._states):
            try:
                await self._orders.cancel_all(ticker=ticker)
            except Exception:
                logger.exception(
                    "market_maker.cancel_failed", ticker=ticker
                )

        self._states.clear()
        logger.info("market_maker.stopped")

    async def _refresh_loop(self) -> None:
        """Periodically refresh quotes on all active markets."""
        while self._running:
            await asyncio.sleep(self._config.quote_refresh_seconds)

            for ticker in list(self._states):
                try:
                    await self.update_quotes(ticker)
                except Exception:
                    logger.exception(
                        "market_maker.refresh_failed", ticker=ticker
                    )

    # ------------------------------------------------------------------
    # Market selection
    # ------------------------------------------------------------------

    async def select_markets(self) -> list[str]:
        """Auto-select markets good for market making.

        Criteria:

        - Spread > ``min_spread`` (default 5 cents)
        - Volume > 0 (someone trades here)
        - Volume < ``max_volume_24h`` (not too competitive)
        - Time to expiry > ``min_hours_to_expiry``
        - Market is open

        Returns
        -------
        list[str]
            Tickers of suitable markets, sorted by spread descending.
        """
        markets = await self._client.get_all_markets(status="open", limit=100)
        now = datetime.now(timezone.utc)
        min_spread = Decimal(str(self._config.min_spread))
        candidates: list[tuple[Decimal, str]] = []

        for m in markets:
            if m.status != MarketStatus.OPEN:
                continue

            # Check spread
            if m.yes_ask <= _ZERO or m.yes_bid <= _ZERO:
                continue
            spread = m.yes_ask - m.yes_bid
            if spread < min_spread:
                continue

            # Check volume
            if m.volume <= 0:
                continue
            if m.volume > self._config.max_volume_24h:
                continue

            # Check time to expiry
            hours_to_expiry = (
                m.close_time - now
            ).total_seconds() / 3600.0
            if hours_to_expiry < self._config.min_hours_to_expiry:
                continue

            candidates.append((spread, m.ticker))

        # Sort by widest spread first (most profitable)
        candidates.sort(key=lambda c: c[0], reverse=True)

        selected = [ticker for _, ticker in candidates][
            : self._config.max_markets
        ]

        logger.info(
            "market_maker.selected_markets",
            total_candidates=len(candidates),
            selected=len(selected),
            tickers=selected,
        )
        return selected

    # ------------------------------------------------------------------
    # Fair value & quoting
    # ------------------------------------------------------------------

    def compute_fair_value(
        self, market: Market, orderbook: OrderbookSnapshot
    ) -> Decimal:
        """Estimate fair value using a volume-weighted mid price.

        Uses the orderbook levels to compute a weighted midpoint.  If the
        orderbook is too thin, falls back to the simple midpoint of the
        top-of-book bid and ask.

        Parameters
        ----------
        market:
            Market snapshot with bid/ask/last_price.
        orderbook:
            Current orderbook snapshot.

        Returns
        -------
        Decimal
            Estimated fair value in the 0.01--0.99 range.
        """
        # Try volume-weighted mid from orderbook
        bid_value = _ZERO
        bid_size = _ZERO
        for level in orderbook.yes_levels:
            bid_value += level.price * level.contracts
            bid_size += level.contracts

        ask_value = _ZERO
        ask_size = _ZERO
        for level in orderbook.no_levels:
            # NO levels represent the ask side: ask_price = 1 - no_price
            implied_ask = _ONE - level.price
            ask_value += implied_ask * level.contracts
            ask_size += level.contracts

        if bid_size > _ZERO and ask_size > _ZERO:
            vwap_bid = bid_value / bid_size
            vwap_ask = ask_value / ask_size
            fair = (vwap_bid + vwap_ask) / 2
        elif market.yes_bid > _ZERO and market.yes_ask > _ZERO:
            # Fallback to simple mid
            fair = (market.yes_bid + market.yes_ask) / 2
        elif market.last_price > _ZERO:
            fair = market.last_price
        else:
            fair = Decimal("0.50")

        # Clamp to valid range
        fair = max(_MIN_PRICE, min(_MAX_PRICE, fair))

        return fair.quantize(_CENT, rounding=ROUND_HALF_UP)

    def compute_quotes(
        self,
        ticker: str,
        fair_value: Decimal,
        orderbook: OrderbookSnapshot,
    ) -> MMQuote:
        """Compute bid and ask prices with inventory skew.

        Base spread is the larger of ``min_spread`` and
        ``current_spread * spread_fraction``.

        Inventory skew adjusts both bid and ask:

        - If ``net_position > 0`` (long): lower bid and ask to encourage
          sells and discourage buys.
        - If ``net_position < 0`` (short): raise bid and ask to encourage
          buys and discourage sells.

        Prices are clamped to the 1--99 cent range and bid is forced to be
        strictly less than ask.

        Parameters
        ----------
        ticker:
            Market ticker.
        fair_value:
            Estimated fair value.
        orderbook:
            Current orderbook snapshot (used to read current spread).

        Returns
        -------
        MMQuote
            The computed two-sided quote.
        """
        min_spread = Decimal(str(self._config.min_spread))

        # Determine current market spread from orderbook
        best_bid = _ZERO
        if orderbook.yes_levels:
            best_bid = orderbook.yes_levels[0].price

        best_ask = _ONE
        if orderbook.no_levels:
            best_ask = _ONE - orderbook.no_levels[0].price

        current_spread = best_ask - best_bid if best_ask > best_bid else min_spread

        # Use the wider of min_spread and a fraction of the current spread
        target_spread = max(
            min_spread,
            (current_spread * Decimal(str(self._config.spread_fraction))).quantize(
                _CENT, rounding=ROUND_HALF_UP
            ),
        )

        half_spread = (target_spread / 2).quantize(_CENT, rounding=ROUND_HALF_UP)

        # Inventory skew
        state = self._states.get(ticker)
        skew = _ZERO
        if state is not None and state.net_position != 0:
            skew_per = Decimal(str(self._config.skew_per_contract))
            skew = skew_per * state.net_position
            # Positive position => negative skew (lower prices to sell)
            # Negative position => positive skew (raise prices to buy)

        bid_price = fair_value - half_spread - skew
        ask_price = fair_value + half_spread - skew

        # Clamp to valid range
        bid_price = max(_MIN_PRICE, min(_MAX_PRICE, bid_price))
        ask_price = max(_MIN_PRICE, min(_MAX_PRICE, ask_price))

        # Ensure bid < ask (at least 1 cent apart)
        if bid_price >= ask_price:
            mid = (bid_price + ask_price) / 2
            bid_price = (mid - _CENT).quantize(_CENT, rounding=ROUND_HALF_UP)
            ask_price = (mid + _CENT).quantize(_CENT, rounding=ROUND_HALF_UP)
            bid_price = max(_MIN_PRICE, bid_price)
            ask_price = min(_MAX_PRICE, ask_price)

        # Quantize to cents
        bid_price = bid_price.quantize(_CENT, rounding=ROUND_HALF_UP)
        ask_price = ask_price.quantize(_CENT, rounding=ROUND_HALF_UP)

        spread = ask_price - bid_price
        edge = (spread / 2).quantize(_CENT, rounding=ROUND_HALF_UP)

        return MMQuote(
            ticker=ticker,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            size=self._config.size_per_side,
            edge_per_contract=edge,
            fair_value=fair_value,
        )

    # ------------------------------------------------------------------
    # Quote management
    # ------------------------------------------------------------------

    async def update_quotes(self, ticker: str) -> None:
        """Recompute and update resting orders for a ticker.

        Fetches the current market and orderbook, computes new quotes, and
        cancels/replaces existing orders if the price has changed.

        Parameters
        ----------
        ticker:
            The market ticker to update.
        """
        state = self._states.get(ticker)
        if state is None:
            logger.warning("market_maker.unknown_ticker", ticker=ticker)
            return

        # Check inventory limits before quoting
        await self.manage_inventory(ticker)

        market = await self._client.get_market(ticker)
        if market.status != MarketStatus.OPEN:
            logger.info(
                "market_maker.market_not_open",
                ticker=ticker,
                status=market.status.value,
            )
            # Pull all quotes for this closed market
            await self._orders.cancel_all(ticker=ticker)
            state.bid_order = None
            state.ask_order = None
            return

        orderbook = await self._client.get_orderbook(ticker)
        fair_value = self.compute_fair_value(market, orderbook)
        quote = self.compute_quotes(ticker, fair_value, orderbook)

        logger.debug(
            "market_maker.quote_computed",
            ticker=ticker,
            bid=str(quote.bid_price),
            ask=str(quote.ask_price),
            spread=str(quote.spread),
            fair_value=str(quote.fair_value),
            net_position=state.net_position,
        )

        # Update bid side
        await self._update_side(
            state=state,
            side="bid",
            new_price=quote.bid_price,
            size=quote.size,
            ticker=ticker,
        )

        # Update ask side
        await self._update_side(
            state=state,
            side="ask",
            new_price=quote.ask_price,
            size=quote.size,
            ticker=ticker,
        )

    async def _update_side(
        self,
        *,
        state: MMState,
        side: str,
        new_price: Decimal,
        size: int,
        ticker: str,
    ) -> None:
        """Cancel and replace an order on one side if the price changed.

        Parameters
        ----------
        state:
            The market-making state for this ticker.
        side:
            ``"bid"`` or ``"ask"``.
        new_price:
            The new desired price for this side.
        size:
            Number of contracts.
        ticker:
            The market ticker.
        """
        existing_order: Order | None
        if side == "bid":
            existing_order = state.bid_order
        else:
            existing_order = state.ask_order

        # Check if existing order is still resting and at the right price
        if existing_order is not None:
            if (
                existing_order.status
                in (OrderStatus.RESTING, OrderStatus.PARTIAL)
                and existing_order.price == new_price
                and existing_order.remaining_count >= size
            ):
                # No change needed
                return

            # Cancel stale order
            try:
                await self._orders.cancel_order(existing_order.order_id)
            except Exception:
                logger.exception(
                    "market_maker.cancel_failed",
                    ticker=ticker,
                    side=side,
                    order_id=existing_order.order_id,
                )

        # Check if inventory limit blocks this side
        if not self._should_quote_side(state, side):
            if side == "bid":
                state.bid_order = None
            else:
                state.ask_order = None
            return

        # Submit new order
        if side == "bid":
            # Buy YES at bid_price
            req = OrderRequest(
                ticker=ticker,
                side=Side.YES,
                action=Action.BUY,
                count=size,
                yes_price=new_price,
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )
        else:
            # Sell YES = buy NO at (1 - ask_price)
            req = OrderRequest(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                count=size,
                yes_price=_ONE - new_price,
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )

        try:
            order = await self._orders.submit_order(req)
            if side == "bid":
                state.bid_order = order
            else:
                state.ask_order = order

            logger.debug(
                "market_maker.order_posted",
                ticker=ticker,
                side=side,
                price=str(new_price),
                size=size,
                order_id=order.order_id,
            )
        except Exception:
            logger.exception(
                "market_maker.order_failed",
                ticker=ticker,
                side=side,
                price=str(new_price),
            )
            if side == "bid":
                state.bid_order = None
            else:
                state.ask_order = None

    def _should_quote_side(self, state: MMState, side: str) -> bool:
        """Determine whether to post a quote on the given side based on
        inventory limits.

        If at max long inventory, do not post another bid (would increase
        long position).  If at max short inventory, do not post another ask
        (would increase short position).

        Parameters
        ----------
        state:
            Current market-making state.
        side:
            ``"bid"`` or ``"ask"``.

        Returns
        -------
        bool
            ``True`` if the side should be quoted.
        """
        max_inv = self._config.max_inventory

        if side == "bid" and state.net_position >= max_inv:
            return False
        if side == "ask" and state.net_position <= -max_inv:
            return False
        return True

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    async def on_fill(self, fill: Fill) -> None:
        """Handle a fill on one side.

        Updates net position, recalculates quotes (applying inventory
        skew), and updates the other side's order.

        Parameters
        ----------
        fill:
            The fill event received from the exchange.
        """
        state = self._states.get(fill.ticker)
        if state is None:
            return

        # Update position based on fill side
        if fill.side == Side.YES and fill.action == Action.BUY:
            # Bid filled: we bought YES
            state.net_position += fill.count
            state.num_bid_fills += 1
            state.realized_spread_pnl -= fill.price * fill.count
            state.bid_order = None  # Order consumed

            logger.info(
                "market_maker.bid_filled",
                ticker=fill.ticker,
                count=fill.count,
                price=str(fill.price),
                net_position=state.net_position,
            )

        elif fill.side == Side.NO and fill.action == Action.BUY:
            # Ask filled: we sold YES (bought NO)
            state.net_position -= fill.count
            state.num_ask_fills += 1
            # Revenue from selling YES at ask_price = 1 - no_price
            ask_price = _ONE - fill.price
            state.realized_spread_pnl += ask_price * fill.count
            state.ask_order = None  # Order consumed

            logger.info(
                "market_maker.ask_filled",
                ticker=fill.ticker,
                count=fill.count,
                price=str(fill.price),
                net_position=state.net_position,
            )

        # Forward the fill to the order manager for bookkeeping
        self._orders.on_fill(fill)

        # Refresh quotes with updated inventory skew
        try:
            await self.update_quotes(fill.ticker)
        except Exception:
            logger.exception(
                "market_maker.post_fill_update_failed",
                ticker=fill.ticker,
            )

    # ------------------------------------------------------------------
    # Inventory management
    # ------------------------------------------------------------------

    async def manage_inventory(self, ticker: str) -> None:
        """Check if inventory is too large and adjust quotes accordingly.

        If ``abs(net_position)`` exceeds ``max_inventory``, pull quotes on
        the side that would increase the position and keep only the reducing
        side.

        Parameters
        ----------
        ticker:
            The market ticker to check.
        """
        state = self._states.get(ticker)
        if state is None:
            return

        max_inv = self._config.max_inventory

        if abs(state.net_position) <= max_inv:
            return

        if state.net_position > max_inv:
            # Too long -- pull bid (would add more longs)
            if state.bid_order is not None:
                try:
                    await self._orders.cancel_order(state.bid_order.order_id)
                    logger.info(
                        "market_maker.inventory_pull_bid",
                        ticker=ticker,
                        net_position=state.net_position,
                    )
                except Exception:
                    logger.exception(
                        "market_maker.inventory_cancel_failed",
                        ticker=ticker,
                        side="bid",
                    )
                state.bid_order = None

        elif state.net_position < -max_inv:
            # Too short -- pull ask (would add more shorts)
            if state.ask_order is not None:
                try:
                    await self._orders.cancel_order(state.ask_order.order_id)
                    logger.info(
                        "market_maker.inventory_pull_ask",
                        ticker=ticker,
                        net_position=state.net_position,
                    )
                except Exception:
                    logger.exception(
                        "market_maker.inventory_cancel_failed",
                        ticker=ticker,
                        side="ask",
                    )
                state.ask_order = None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self, ticker: str) -> MMState:
        """Return the current state for a single market.

        Parameters
        ----------
        ticker:
            The market ticker.

        Returns
        -------
        MMState
            The live market-making state.

        Raises
        ------
        KeyError
            If the ticker is not being actively made.
        """
        if ticker not in self._states:
            raise KeyError(f"Not making market on {ticker}")
        return self._states[ticker]

    def get_all_states(self) -> list[MMState]:
        """Return states for all actively made markets.

        Returns
        -------
        list[MMState]
            All current market-making states.
        """
        return list(self._states.values())

    def get_pnl_summary(self) -> dict[str, Decimal | int]:
        """Compute aggregate P&L across all markets being made.

        Returns
        -------
        dict
            Summary with keys:
            - ``total_realized_pnl``: Sum of spread P&L across all markets.
            - ``total_bid_fills``: Total bid fills across all markets.
            - ``total_ask_fills``: Total ask fills across all markets.
            - ``total_net_position``: Sum of net positions (signed).
            - ``num_markets``: Number of markets being made.
        """
        total_pnl = _ZERO
        total_bid_fills = 0
        total_ask_fills = 0
        total_net_position = 0

        for state in self._states.values():
            total_pnl += state.realized_spread_pnl
            total_bid_fills += state.num_bid_fills
            total_ask_fills += state.num_ask_fills
            total_net_position += state.net_position

        return {
            "total_realized_pnl": total_pnl,
            "total_bid_fills": total_bid_fills,
            "total_ask_fills": total_ask_fills,
            "total_net_position": total_net_position,
            "num_markets": len(self._states),
        }
