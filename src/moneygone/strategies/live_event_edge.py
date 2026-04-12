"""Live event edge detection strategy.

Detects mid-event edge by comparing real-time data (sports scores,
weather observations, crypto prices) to Kalshi market prices.

Unlike the resolution sniper (which waits for outcomes to be known),
this strategy identifies situations where the CURRENT STATE of an event
strongly implies an outcome but the market hasn't caught up.

Examples
--------
- NBA team up 25 in Q4 -> ~99% win prob, market at 80c -> 19c edge
- Temperature already exceeded threshold at 10am -> ~95%+ prob, market at 70c
- Bitcoin at $82K and rising, "above $80K" contract at 70c
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import structlog

from moneygone.data.crypto.ccxt_feed import CryptoDataFeed
from moneygone.data.sports.espn import ESPNLiveFeed, GameState
from moneygone.data.sports.live_weather import LiveWeatherFeed, WeatherObservation
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import (
    Action,
    Market,
    OrderRequest,
    Side,
    TimeInForce,
)
from moneygone.execution.fill_tracker import FillTracker
from moneygone.execution.order_manager import OrderManager
from moneygone.risk.manager import RiskManager
from moneygone.signals.edge import EdgeCalculator, EdgeResult
from moneygone.signals.fees import KalshiFeeCalculator
from moneygone.sizing.kelly import KellySizer, SizeResult
from moneygone.sizing.risk_limits import ProposedTrade
from moneygone.utils.logging import get_logger
from moneygone.utils.time import now_utc

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveEdgeSignal:
    """A detected edge opportunity from live event data."""

    ticker: str
    category: str  # "sports", "weather", "crypto"
    implied_probability: float  # from real-time data
    market_probability: float  # current Kalshi price
    edge: float  # implied - market (for YES side)
    fee_adjusted_edge: float
    confidence: float
    reasoning: str  # human-readable explanation
    data_snapshot: dict[str, Any]  # raw data that produced this signal
    detected_at: datetime
    urgency: str  # "high", "medium", "low"


@dataclass(frozen=True)
class TradeDecision:
    """Result of evaluating a LiveEdgeSignal through the full pipeline."""

    signal: LiveEdgeSignal
    edge_result: EdgeResult
    size_result: SizeResult
    contracts: int
    side: str  # "yes" or "no"
    price: Decimal
    should_trade: bool
    rejection_reason: str | None = None


@dataclass
class LiveEdgeConfig:
    """Configuration for the live event edge strategy."""

    min_edge: float = 0.05  # 5 cents minimum edge for live events
    min_confidence: float = 0.90
    scan_interval_seconds: float = 15.0  # scan every 15 seconds
    max_contracts_per_trade: int = 100
    sports_enabled: bool = True
    weather_enabled: bool = True
    crypto_enabled: bool = True
    # Sports-specific
    sports_leagues: list[str] = field(
        default_factory=lambda: ["nba", "nfl", "mlb", "nhl"]
    )
    # Weather-specific
    weather_stations: list[dict[str, Any]] = field(default_factory=list)
    # Crypto-specific
    crypto_symbols: list[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT"]
    )
    # Volatility window for crypto probability estimation (hours)
    crypto_vol_window_hours: float = 24.0


# ---------------------------------------------------------------------------
# Sports probability estimator
# ---------------------------------------------------------------------------


class SportsProbabilityEstimator:
    """Estimates win probability from current game state.

    Uses conservative lookup tables based on historical data.
    NOT a ML model -- just hardcoded conservative rules.
    This is intentionally simple because:
    1. We only trade at extreme edges (20+ point leads)
    2. Overcomplicating this adds model risk
    3. Historical base rates for blowouts are very stable
    """

    def estimate_win_prob(self, game: GameState) -> float | None:
        """Return estimated P(leader wins) or None if too uncertain to trade.

        Returns None when the game state does not meet our conservative
        thresholds for high-confidence estimation.
        """
        sport = game.sport
        if not sport:
            # Try to infer from league field
            league_to_sport = {
                "nba": "nba",
                "nfl": "nfl",
                "mlb": "mlb",
                "nhl": "nhl",
                "eng.1": "soccer",
                "esp.1": "soccer",
                "ger.1": "soccer",
                "ita.1": "soccer",
                "fra.1": "soccer",
                "usa.1": "soccer",
                "soccer_epl": "soccer",
                "soccer_spain_la_liga": "soccer",
                "soccer_germany_bundesliga": "soccer",
                "soccer_italy_serie_a": "soccer",
                "soccer_france_ligue_one": "soccer",
                "soccer_usa_mls": "soccer",
                "mls": "soccer",
                "ufc": "mma",
                "atp": "tennis",
                "wta": "tennis",
                "tennis_atp": "tennis",
                "tennis_wta": "tennis",
                "pga": "golf",
                "golf_pga": "golf",
            }
            sport = league_to_sport.get(game.league, "")

        lead = abs(game.home_score - game.away_score)
        if lead == 0:
            return None  # tied games are too uncertain

        # Parse period and clock
        try:
            period = int(game.period)
        except (ValueError, TypeError):
            return None

        minutes_left = self._parse_clock_minutes(game.clock)

        if sport in ("basketball", "nba"):
            return self._nba_win_prob(lead, period, minutes_left)
        elif sport in ("football", "nfl"):
            return self._nfl_win_prob(lead, period, minutes_left)
        elif sport in ("baseball", "mlb"):
            is_top = "top" in game.detail.lower() if game.detail else True
            return self._mlb_win_prob(lead, period, is_top)
        elif sport in ("hockey", "nhl"):
            return self._nhl_win_prob(lead, period, minutes_left)
        elif sport in ("soccer",) or game.league in (
            "eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "usa.1",
            "soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga",
            "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_usa_mls",
            "mls",
        ):
            return self._soccer_win_prob(lead, period, minutes_left)
        elif sport in ("mma",) or game.league in ("ufc",):
            return self._mma_win_prob(game)
        elif sport in ("tennis",) or game.league in ("atp", "wta", "tennis_atp", "tennis_wta"):
            return self._tennis_win_prob(game)
        elif sport in ("golf",) or game.league in ("pga", "golf_pga"):
            return self._golf_win_prob(game)

        return None

    def _nba_win_prob(
        self, lead: int, period: int, minutes_left: float
    ) -> float | None:
        """NBA win probability based on lead, quarter, and time remaining.

        Conservative thresholds -- only returns a probability when we are
        very confident, i.e. large leads late in the game.
        """
        if period < 3:
            # Too early to estimate with high confidence
            return None

        # Q4 or OT
        if period >= 4:
            if minutes_left < 5:
                if lead >= 20:
                    return 0.99
                if lead >= 15:
                    return 0.97
                if lead >= 10:
                    return 0.95
                if lead >= 7:
                    return 0.92
            elif minutes_left < 8:
                if lead >= 20:
                    return 0.97
                if lead >= 15:
                    return 0.95
                if lead >= 10:
                    return 0.92
            elif minutes_left <= 12:
                if lead >= 20:
                    return 0.95
                if lead >= 15:
                    return 0.92

        # Q3
        if period == 3:
            if lead >= 25:
                return 0.92
            if lead >= 20:
                return 0.90

        return None

    def _nfl_win_prob(
        self, lead: int, quarter: int, minutes_left: float
    ) -> float | None:
        """NFL win probability from current game state.

        NFL leads are more durable because scoring is harder (possessions
        are longer, field goals worth 3, TDs worth 6-8).
        """
        if quarter < 3:
            return None

        if quarter >= 4:
            if minutes_left < 5:
                if lead >= 17:
                    return 0.99
                if lead >= 14:
                    return 0.97
                if lead >= 11:
                    return 0.95
                if lead >= 8:
                    return 0.92
            elif minutes_left < 10:
                if lead >= 17:
                    return 0.97
                if lead >= 14:
                    return 0.95
            elif minutes_left <= 15:
                if lead >= 21:
                    return 0.95
                if lead >= 17:
                    return 0.92

        if quarter == 3:
            if lead >= 21:
                return 0.90

        return None

    def _mlb_win_prob(
        self, lead: int, inning: int, is_top: bool
    ) -> float | None:
        """MLB win probability from current game state.

        MLB games have 9 innings; leads in late innings are very sticky
        because scoring is relatively rare per inning.
        """
        if inning < 7:
            return None

        if inning >= 9:
            if lead >= 5:
                return 0.99
            if lead >= 3:
                return 0.97
            if lead >= 2:
                return 0.95
        elif inning == 8:
            if lead >= 5:
                return 0.97
            if lead >= 3:
                return 0.95
        elif inning == 7:
            if lead >= 6:
                return 0.95
            if lead >= 5:
                return 0.92

        return None

    def _nhl_win_prob(
        self, lead: int, period: int, minutes_left: float
    ) -> float | None:
        """NHL win probability from current game state.

        Hockey goals are rare (avg ~3 per game per team), so multi-goal
        leads in the 3rd period are very safe.
        """
        if period < 3:
            return None

        if period >= 3:
            if minutes_left < 10:
                if lead >= 4:
                    return 0.99
                if lead >= 3:
                    return 0.98
                if lead >= 2:
                    return 0.96
            elif minutes_left <= 20:
                if lead >= 4:
                    return 0.98
                if lead >= 3:
                    return 0.96

        return None

    def _soccer_win_prob(
        self, lead: int, period: int, minutes_left: float
    ) -> float | None:
        """Soccer win probability from current game state.

        Soccer games are 90 minutes (two 45-min halves).  Goals are rare
        (~2.7 per game total), so multi-goal leads late in the match are
        extremely durable.

        ESPN reports period=1 (1st half) or period=2 (2nd half).
        Clock counts UP in soccer, so we convert: minutes_played = clock,
        minutes_left = 90 - minutes_played (approximately).

        Historical conversion rates for leads:
        - 2-goal lead after 70 min → ~97% win rate
        - 2-goal lead after 80 min → ~99%
        - 1-goal lead after 85 min → ~93%
        - 3-goal lead any time 2nd half → ~99%
        """
        # In ESPN soccer data, the clock is minutes played (counts up)
        # and period=2 means second half.
        if period < 2:
            # First half — only trade 3+ goal leads
            if lead >= 3:
                return 0.92
            return None

        # Second half — estimate minutes remaining
        # ESPN clock for soccer = minutes played in current half
        # Total minutes played ≈ 45 + clock_in_2nd_half
        mins_played = 45 + minutes_left  # minutes_left here is actually clock
        mins_remaining = max(0.0, 95.0 - mins_played)  # include ~5 min stoppage

        if lead >= 3:
            return 0.99
        if lead >= 2:
            if mins_remaining < 10:
                return 0.99
            if mins_remaining < 20:
                return 0.97
            if mins_remaining < 30:
                return 0.95
        if lead >= 1:
            if mins_remaining < 5:
                return 0.93
            if mins_remaining < 10:
                return 0.90

        return None

    def _mma_win_prob(self, game: GameState) -> float | None:
        """MMA/UFC win probability from current fight state.

        UFC fights are 3 rounds (non-title) or 5 rounds (title).
        If a fighter is significantly ahead on the scorecards going
        into the final round, the probability of winning is very high
        (would need a finish to lose).

        ESPN provides limited real-time scoring for MMA, so we're
        conservative — only trade when fight goes to decision
        (i.e., is_final or near-final).
        """
        # MMA is harder to estimate mid-fight without scorecard data.
        # We'll only snipe after the fight is over (resolution sniper),
        # not mid-fight via live edge.
        return None

    def _tennis_win_prob(self, game: GameState) -> float | None:
        """Tennis win probability from current match state.

        Tennis matches are best of 3 or 5 sets.  Key near-decided states:
        - Up 2 sets to 0 in best of 3 → match over (100%)
        - Up 2 sets to 0 in best of 5 → ~95% (historical)
        - Up 2 sets to 1 in best of 5 and serving for match → ~92%

        ESPN detail string shows set scores, e.g. "6-3, 6-4, 3-2"
        We parse the detail to count sets won.
        """
        detail = game.detail or ""
        detail_lower = detail.lower()

        # Try to count completed sets from the detail string
        # ESPN format: "6-3, 7-5, 4-2" or "Final: 6-3, 6-4"
        import re
        set_scores = re.findall(r"(\d+)-(\d+)", detail)
        if not set_scores:
            return None

        # Count sets won by each side
        # We need to figure out which player is leading
        home_sets = 0
        away_sets = 0
        for s1, s2 in set_scores:
            s1, s2 = int(s1), int(s2)
            if s1 > s2 and s1 >= 6:
                home_sets += 1
            elif s2 > s1 and s2 >= 6:
                away_sets += 1
            # else: set still in progress

        sets_leader = max(home_sets, away_sets)
        sets_trailer = min(home_sets, away_sets)

        # Detect best-of-5 (Grand Slams) from detail string or set count
        is_best_of_5 = (
            "5 sets" in detail_lower
            or "best of 5" in detail_lower
            or len(set_scores) > 3
            or sets_leader == 3
        )

        # Best of 5 (Grand Slams) — check first to avoid early return
        if is_best_of_5:
            if sets_leader == 3:
                return 0.99  # match over
            if sets_leader == 2 and sets_trailer == 0:
                # Up 2-0 in best of 5 — historically ~95%
                return 0.95
            if sets_leader == 2 and sets_trailer == 1:
                # Up 2-1 in best of 5
                return 0.90
            return None

        # Best of 3 (most ATP/WTA except Grand Slams)
        if sets_leader == 2 and sets_trailer == 0:
            # Match is over or about to be
            return 0.99
        if sets_leader == 2 and sets_trailer == 1:
            # 2-1 in sets, in deciding set — still competitive
            return None

        return None

    def _golf_win_prob(self, game: GameState) -> float | None:
        """Golf tournament leader probability.

        Golf scoring is relative (strokes under/over par).  Historical
        conversion rates for stroke leads going into the final round:
        - 6+ stroke lead after 54 holes → ~98%
        - 4-5 stroke lead after 54 holes → ~90%
        - 3 stroke lead after 54 holes → ~80%

        For mid-round leads, the probabilities are higher because
        there are fewer holes remaining.

        ESPN provides the leaderboard; we use home_score/away_score
        as a proxy for the score differential.
        """
        lead = abs(game.home_score - game.away_score)
        if lead == 0:
            return None

        # Golf uses period for the round number (1-4)
        try:
            round_num = int(game.period)
        except (ValueError, TypeError):
            return None

        # Round 4 (final round) or later
        if round_num >= 4:
            if lead >= 8:
                return 0.99
            if lead >= 6:
                return 0.98
            if lead >= 4:
                return 0.95
            if lead >= 3:
                return 0.92
            if lead >= 2:
                return 0.90

        # Round 3 (going into final round)
        if round_num == 3:
            if lead >= 7:
                return 0.97
            if lead >= 5:
                return 0.93
            if lead >= 4:
                return 0.90

        return None

    @staticmethod
    def _parse_clock_minutes(clock: str) -> float:
        """Parse a clock string like '5:32' or '0:45' into decimal minutes."""
        if not clock or clock == "--" or clock.lower() == "end":
            return 0.0
        try:
            parts = clock.strip().split(":")
            if len(parts) == 2:
                mins = float(parts[0])
                secs = float(parts[1])
                return mins + secs / 60.0
            elif len(parts) == 1:
                # Just seconds or just minutes
                val = float(parts[0])
                if val > 59:
                    return val / 60.0  # was seconds
                return val
        except (ValueError, IndexError):
            pass
        return 0.0


# ---------------------------------------------------------------------------
# Crypto price edge estimator
# ---------------------------------------------------------------------------


class CryptoPriceEdgeEstimator:
    """Estimates probability that crypto price stays above/below threshold.

    Uses current price, time remaining, and recent realized volatility
    to estimate P(price > threshold at expiry).

    Simple model: assumes log-normal price movement.

        P(S_T > K) = N(d2)

    where::

        d2 = (ln(S/K) + (r - sigma^2 / 2) * T) / (sigma * sqrt(T))

    - r = 0 for crypto (no risk-free rate assumption)
    - sigma = realized vol from last 24h
    - T = time to expiry in years
    """

    def __init__(self, crypto_feed: CryptoDataFeed) -> None:
        self._feed = crypto_feed
        self._vol_cache: dict[str, tuple[float, datetime]] = {}
        self._vol_cache_ttl_seconds = 300.0  # refresh vol every 5 min

    async def estimate_prob(
        self,
        current_price: float,
        threshold: float,
        hours_remaining: float,
        symbol: str = "BTC/USDT",
    ) -> float:
        """Estimate P(price > threshold at expiry).

        Parameters
        ----------
        current_price:
            Current spot price.
        threshold:
            The threshold level from the Kalshi contract.
        hours_remaining:
            Hours until contract expiry.
        symbol:
            The trading pair (used for volatility estimation).

        Returns
        -------
        float
            Estimated probability between 0 and 1.
        """
        if current_price <= 0 or threshold <= 0 or hours_remaining <= 0:
            # Edge cases
            if current_price >= threshold:
                return 0.99  # already above and no time to fall
            return 0.01

        sigma = await self._get_realized_vol(symbol)
        if sigma <= 0:
            # Fallback: if no vol data, use a conservative estimate
            # based purely on current price vs threshold
            if current_price >= threshold:
                return 0.90
            return 0.10

        # Time to expiry in years
        t = hours_remaining / (365.25 * 24.0)
        if t <= 0:
            return 1.0 if current_price >= threshold else 0.0

        # d2 = (ln(S/K) - sigma^2/2 * T) / (sigma * sqrt(T))
        log_moneyness = math.log(current_price / threshold)
        vol_term = sigma * math.sqrt(t)

        if vol_term <= 0:
            return 1.0 if current_price >= threshold else 0.0

        d2 = (log_moneyness - 0.5 * sigma * sigma * t) / vol_term

        # N(d2) using the normal CDF
        prob = _normal_cdf(d2)

        # Clamp to avoid extreme values
        return max(0.01, min(0.99, prob))

    async def _get_realized_vol(self, symbol: str) -> float:
        """Compute annualized realized volatility from recent trades.

        Uses the last ~24h of trades to estimate hourly returns standard
        deviation, then annualizes.
        """
        # Check cache
        cached = self._vol_cache.get(symbol)
        if cached is not None:
            vol, cached_at = cached
            age = (now_utc() - cached_at).total_seconds()
            if age < self._vol_cache_ttl_seconds:
                return vol

        try:
            trades = await self._feed.get_recent_trades(symbol, limit=100)
        except Exception:
            log.warning(
                "crypto_vol.fetch_error", symbol=symbol, exc_info=True
            )
            # Fallback: BTC annualized vol ~60%, ETH ~80%
            if "BTC" in symbol:
                return 0.60
            elif "ETH" in symbol:
                return 0.80
            return 0.70

        if len(trades) < 10:
            # Not enough data, use fallback
            if "BTC" in symbol:
                return 0.60
            elif "ETH" in symbol:
                return 0.80
            return 0.70

        # Compute log returns between consecutive trades
        prices = [t.price for t in sorted(trades, key=lambda x: x.timestamp)]
        log_returns: list[float] = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))

        if len(log_returns) < 5:
            if "BTC" in symbol:
                return 0.60
            return 0.70

        # Standard deviation of log returns
        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (
            len(log_returns) - 1
        )
        std_ret = math.sqrt(variance)

        # Estimate the average time between trades to annualize
        time_span = (
            trades[-1].timestamp - trades[0].timestamp
        ).total_seconds()
        if time_span <= 0:
            if "BTC" in symbol:
                return 0.60
            return 0.70

        n_intervals = len(log_returns)
        avg_interval_seconds = time_span / n_intervals
        intervals_per_year = (365.25 * 24 * 3600) / avg_interval_seconds

        # Annualize: sigma_annual = sigma_interval * sqrt(intervals_per_year)
        annualized_vol = std_ret * math.sqrt(intervals_per_year)

        # Sanity clamp: crypto vol is typically 30-150% annualized
        annualized_vol = max(0.10, min(2.0, annualized_vol))

        # Cache it
        self._vol_cache[symbol] = (annualized_vol, now_utc())

        log.debug(
            "crypto_vol.computed",
            symbol=symbol,
            annualized_vol=round(annualized_vol, 4),
            n_trades=len(trades),
        )

        return annualized_vol


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------


class LiveEventEdge:
    """Detects mid-event edge by comparing real-time data to market prices.

    Unlike the resolution sniper (which waits for outcomes to be known),
    this strategy identifies situations where the CURRENT STATE of an event
    strongly implies an outcome but the market hasn't caught up.

    Examples:
    - NBA team up 25 in Q4 -> ~99% win prob, market at 80c -> 19c edge
    - Temperature already exceeded threshold at 10am -> ~95%+ prob,
      market at 70c
    - Bitcoin at $82K and rising, "above $80K" contract at 70c
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        espn_feed: ESPNLiveFeed,
        weather_feed: LiveWeatherFeed,
        fee_calculator: KalshiFeeCalculator,
        edge_calculator: EdgeCalculator,
        sizer: KellySizer,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        config: LiveEdgeConfig,
        crypto_feed: CryptoDataFeed | None = None,
        fill_tracker: FillTracker | None = None,
    ) -> None:
        self._client = rest_client
        self._espn = espn_feed
        self._weather = weather_feed
        self._fees = fee_calculator
        self._edge_calc = edge_calculator
        self._sizer = sizer
        self._risk = risk_manager
        self._orders = order_manager
        self._config = config
        self._crypto_feed = crypto_feed
        self._fill_tracker = fill_tracker
        self._sports_estimator = SportsProbabilityEstimator()
        self._crypto_estimator = (
            CryptoPriceEdgeEstimator(crypto_feed) if crypto_feed else None
        )
        self._running = False
        self._scan_task: asyncio.Task[None] | None = None
        self._traded_tickers: set[str] = set()  # avoid re-trading same signal
        self._signals_detected: int = 0
        self._trades_executed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the continuous scan loop."""
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        log.info(
            "live_edge.started",
            sports=self._config.sports_enabled,
            weather=self._config.weather_enabled,
            crypto=self._config.crypto_enabled,
            interval=self._config.scan_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the scan loop."""
        self._running = False
        if self._scan_task is not None:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None
        log.info(
            "live_edge.stopped",
            signals_detected=self._signals_detected,
            trades_executed=self._trades_executed,
        )

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Continuously scan for edge opportunities."""
        while self._running:
            try:
                signals: list[LiveEdgeSignal] = []

                # Run all enabled scanners concurrently
                tasks: list[asyncio.Task[list[LiveEdgeSignal]]] = []

                if self._config.sports_enabled:
                    tasks.append(
                        asyncio.create_task(self._safe_scan(self.scan_sports_edge))
                    )
                if self._config.weather_enabled:
                    tasks.append(
                        asyncio.create_task(self._safe_scan(self.scan_weather_edge))
                    )
                if self._config.crypto_enabled and self._crypto_estimator:
                    tasks.append(
                        asyncio.create_task(self._safe_scan(self.scan_crypto_edge))
                    )

                if tasks:
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        signals.extend(result)

                # Process all signals
                for signal in signals:
                    self._signals_detected += 1
                    log.info(
                        "live_edge.signal_detected",
                        ticker=signal.ticker,
                        category=signal.category,
                        edge=round(signal.edge, 4),
                        fee_adjusted_edge=round(signal.fee_adjusted_edge, 4),
                        confidence=round(signal.confidence, 4),
                        urgency=signal.urgency,
                        reasoning=signal.reasoning,
                    )

                    decision = await self.evaluate_signal(signal)
                    if decision is not None and decision.should_trade:
                        await self.execute_signal(signal, decision)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("live_edge.scan_error")

            await asyncio.sleep(self._config.scan_interval_seconds)

    async def _safe_scan(
        self,
        scanner: Any,
    ) -> list[LiveEdgeSignal]:
        """Run a scanner and return empty list on error."""
        try:
            result = await scanner()
            return result  # type: ignore[no-any-return]
        except Exception:
            log.warning("live_edge.scanner_error", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Sports edge scanner
    # ------------------------------------------------------------------

    async def scan_sports_edge(self) -> list[LiveEdgeSignal]:
        """Scan live sports games for mid-game edge.

        Win probability estimation rules (conservative):
        - Lead of 20+ pts in NBA Q4 with < 5 min -> 99%
        - Lead of 15+ pts in NBA Q4 -> 97%
        - Lead of 10+ pts in NBA Q4 -> 92%
        - Lead of 20+ in Q3 -> 90%
        - NFL lead of 17+ in Q4 with < 5 min -> 97%
        - NFL lead of 14+ in Q4 -> 92%
        - MLB lead of 5+ in 9th inning -> 97%
        - NHL lead of 3+ in 3rd period -> 96%

        These are intentionally conservative -- we only trade when
        extremely high confidence.
        """
        signals: list[LiveEdgeSignal] = []

        # Fetch live games from ESPN for each league
        for league_key in self._config.sports_leagues:
            from moneygone.data.sports.espn import SUPPORTED_LEAGUES

            sport_path = SUPPORTED_LEAGUES.get(league_key)
            if sport_path is None:
                continue

            sport, league = sport_path.split("/")

            try:
                games = await self._espn.get_live_scores(sport, league)
            except Exception:
                log.warning(
                    "live_edge.sports_fetch_error",
                    league=league_key,
                    exc_info=True,
                )
                continue

            for game in games:
                if game.is_final or game.status != "in":
                    continue

                win_prob = self._sports_estimator.estimate_win_prob(game)
                if win_prob is None:
                    continue  # Not confident enough

                if win_prob < self._config.min_confidence:
                    continue

                # Find matching Kalshi markets for this game
                matching_markets = await self._find_sports_markets(game)
                for market in matching_markets:
                    signal = await self._build_sports_signal(
                        game, win_prob, market
                    )
                    if signal is not None:
                        signals.append(signal)

        return signals

    async def _find_sports_markets(self, game: GameState) -> list[Market]:
        """Find Kalshi markets that correspond to a live game.

        Searches for markets matching the teams involved.
        """
        try:
            # Search by team names in the market title
            # Kalshi sports markets typically include team names
            markets = await self._client.get_all_markets(
                status="open",
                series_ticker=game.league.upper() if game.league else None,
                limit=100,
            )

            matching: list[Market] = []
            home_lower = game.home_team.lower()
            away_lower = game.away_team.lower()

            for m in markets:
                title_lower = m.title.lower()
                # Match if the market title contains either team name
                if home_lower in title_lower or away_lower in title_lower:
                    matching.append(m)

            return matching

        except Exception:
            log.warning(
                "live_edge.market_search_error",
                game_id=game.game_id,
                exc_info=True,
            )
            return []

    async def _build_sports_signal(
        self,
        game: GameState,
        win_prob: float,
        market: Market,
    ) -> LiveEdgeSignal | None:
        """Build a LiveEdgeSignal for a sports game + market pair."""
        if market.ticker in self._traded_tickers:
            return None

        # Determine which side the win_prob applies to
        # We need to figure out if this market is for the leading team
        leader = (
            game.home_team
            if game.home_score >= game.away_score
            else game.away_team
        )
        title_lower = market.title.lower()
        leader_lower = leader.lower()

        # If the market is about the leading team winning, use win_prob directly
        # If it's about the trailing team, invert
        if leader_lower in title_lower:
            model_prob = win_prob
        else:
            model_prob = 1.0 - win_prob

        # Market price is the current YES ask or mid price
        market_price = float(market.yes_ask) if market.yes_ask > 0 else float(market.last_price)
        if market_price <= 0 or market_price >= 1:
            return None

        # Compute raw edge
        edge = model_prob - market_price

        # Fee adjustment
        fee = float(
            self._fees.fee_per_contract(Decimal(str(market_price)), is_maker=False)
        )
        fee_adjusted = edge - fee

        if fee_adjusted < self._config.min_edge:
            return None

        # Determine urgency based on game state
        try:
            period = int(game.period)
        except (ValueError, TypeError):
            period = 0

        if period >= 4 and game.sport in ("basketball", "nba", "football", "nfl"):
            urgency = "high"
        elif period >= 3:
            urgency = "medium"
        else:
            urgency = "low"

        lead = abs(game.home_score - game.away_score)

        return LiveEdgeSignal(
            ticker=market.ticker,
            category="sports",
            implied_probability=model_prob,
            market_probability=market_price,
            edge=round(edge, 6),
            fee_adjusted_edge=round(fee_adjusted, 6),
            confidence=win_prob,
            reasoning=(
                f"{leader} leads by {lead} in {game.detail}. "
                f"Estimated win prob {win_prob:.0%}, market at "
                f"{market_price:.0%} ({edge:+.1%} edge)"
            ),
            data_snapshot={
                "game_id": game.game_id,
                "sport": game.sport,
                "league": game.league,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "period": game.period,
                "clock": game.clock,
                "detail": game.detail,
                "leader": leader,
                "lead": lead,
            },
            detected_at=now_utc(),
            urgency=urgency,
        )

    # ------------------------------------------------------------------
    # Weather edge scanner
    # ------------------------------------------------------------------

    async def scan_weather_edge(self) -> list[LiveEdgeSignal]:
        """Scan live weather observations for edge.

        Key scenarios:
        - Temperature already exceeded threshold -> P(exceeds) = very high
          (only fails if measurement is revised, which is rare)
        - Temperature well below threshold with only hours left and
          falling -> P(exceeds) = very low
        - Current temp within 2 deg F of threshold with hours left -> less
          certain, use forecast

        Confidence rules:
        - Already exceeded: 0.98
        - 3+ degrees above threshold: 0.95
        - Within 2 degrees, rising trend: 0.70 (skip, too uncertain)
        """
        signals: list[LiveEdgeSignal] = []

        # Find open weather markets on Kalshi
        try:
            weather_markets = await self._client.get_markets(
                status="open",
            )
        except Exception:
            log.warning("live_edge.weather_markets_error", exc_info=True)
            return signals

        # Filter to weather-related markets
        weather_markets = [
            m
            for m in weather_markets
            if _is_weather_market(m)
        ]

        if not weather_markets:
            return signals

        for market in weather_markets:
            if market.ticker in self._traded_tickers:
                continue

            signal = await self._evaluate_weather_market(market)
            if signal is not None:
                signals.append(signal)

        return signals

    async def _evaluate_weather_market(
        self, market: Market
    ) -> LiveEdgeSignal | None:
        """Evaluate a single weather market for edge."""
        # Try parsing from the ticker first (more reliable), then title
        parsed = _parse_weather_ticker(market.ticker)
        if parsed is None:
            parsed = _parse_weather_market_title(market.title)
        if parsed is None:
            return None

        location, threshold_f, direction = parsed

        # Fetch current observation for the location
        for station_cfg in self._config.weather_stations:
            station_name = station_cfg.get("name", "")
            if station_name.lower() not in location.lower():
                continue

            station_id = station_cfg.get("station_id", "")
            lat = station_cfg.get("lat")
            lon = station_cfg.get("lon")

            obs: WeatherObservation | None = None
            if lat is not None and lon is not None:
                obs = await self._weather.get_current_observation_open_meteo(
                    lat, lon, station_id=station_id
                )
            elif station_id:
                obs = await self._weather.get_current_observation(station_id)

            if obs is None or obs.temperature_f is None:
                continue

            return self._build_weather_signal(
                market, obs, threshold_f, direction, location
            )

        return None

    def _build_weather_signal(
        self,
        market: Market,
        obs: WeatherObservation,
        threshold_f: float,
        direction: str,
        location: str,
    ) -> LiveEdgeSignal | None:
        """Build a weather edge signal from observation data."""
        assert obs.temperature_f is not None
        current_temp = obs.temperature_f
        margin = current_temp - threshold_f  # positive = above threshold

        # Calculate hours remaining in the day
        now = now_utc()
        hours_remaining = max(0.0, 24.0 - now.hour - now.minute / 60.0)

        # Determine implied probability
        if direction == "above":
            if margin >= 3.0:
                # Already well above threshold
                implied_prob = 0.98
                confidence = 0.98
            elif margin >= 0.0:
                # Currently at or above threshold
                implied_prob = 0.95
                confidence = 0.95
            elif margin >= -2.0:
                # Within 2 degrees, could still reach threshold
                # Too uncertain to trade
                return None
            else:
                # Well below threshold
                if hours_remaining < 3.0:
                    # Little time left to warm up
                    implied_prob = 0.05
                    confidence = 0.92
                else:
                    return None  # too uncertain
        else:
            # "below" direction -- market pays if temp stays below threshold
            if margin <= -3.0:
                # Well below threshold
                implied_prob = 0.98
                confidence = 0.98
            elif margin <= 0.0:
                implied_prob = 0.95
                confidence = 0.95
            elif margin <= 2.0:
                return None  # too close
            else:
                if hours_remaining < 3.0:
                    implied_prob = 0.05
                    confidence = 0.92
                else:
                    return None

        market_price = float(
            market.yes_ask if market.yes_ask > 0 else market.last_price
        )
        if market_price <= 0 or market_price >= 1:
            return None

        edge = implied_prob - market_price
        fee = float(
            self._fees.fee_per_contract(
                Decimal(str(market_price)), is_maker=False
            )
        )
        fee_adjusted = edge - fee

        if fee_adjusted < self._config.min_edge:
            return None

        if confidence < self._config.min_confidence:
            return None

        # Urgency based on hours remaining
        if hours_remaining < 2:
            urgency = "high"
        elif hours_remaining < 6:
            urgency = "medium"
        else:
            urgency = "low"

        return LiveEdgeSignal(
            ticker=market.ticker,
            category="weather",
            implied_probability=implied_prob,
            market_probability=market_price,
            edge=round(edge, 6),
            fee_adjusted_edge=round(fee_adjusted, 6),
            confidence=confidence,
            reasoning=(
                f"{location} temp is {current_temp:.1f}F "
                f"(threshold {threshold_f:.0f}F {direction}). "
                f"Margin {margin:+.1f}F, {hours_remaining:.1f}h remaining. "
                f"Implied prob {implied_prob:.0%}, market {market_price:.0%} "
                f"({edge:+.1%} edge)"
            ),
            data_snapshot={
                "location": location,
                "current_temp_f": current_temp,
                "threshold_f": threshold_f,
                "direction": direction,
                "margin_f": round(margin, 1),
                "hours_remaining": round(hours_remaining, 1),
                "station_id": obs.station_id,
                "observed_at": obs.observation_time.isoformat(),
            },
            detected_at=now_utc(),
            urgency=urgency,
        )

    # ------------------------------------------------------------------
    # Crypto edge scanner
    # ------------------------------------------------------------------

    async def scan_crypto_edge(self) -> list[LiveEdgeSignal]:
        """Scan live crypto prices for edge on crypto prediction markets.

        If BTC is at $82K and "above $80K by EOD" is at 70c, that's edge.

        Confidence based on:
        - Current price distance from threshold (farther = more confident)
        - Time remaining (less time = price less likely to reverse)
        - Recent volatility (high vol = less confident)
        - Direction of trend
        """
        if self._crypto_feed is None or self._crypto_estimator is None:
            return []

        signals: list[LiveEdgeSignal] = []

        # Find open crypto markets on Kalshi
        try:
            all_markets = await self._client.get_markets(status="open")
        except Exception:
            log.warning("live_edge.crypto_markets_error", exc_info=True)
            return signals

        crypto_markets = [m for m in all_markets if _is_crypto_market(m)]

        if not crypto_markets:
            return signals

        # Fetch current prices for tracked symbols
        current_prices: dict[str, float] = {}
        for symbol in self._config.crypto_symbols:
            try:
                trades = await self._crypto_feed.get_recent_trades(
                    symbol, limit=1
                )
                if trades:
                    current_prices[symbol] = trades[0].price
            except Exception:
                log.debug(
                    "live_edge.crypto_price_error",
                    symbol=symbol,
                    exc_info=True,
                )

        if not current_prices:
            return signals

        for market in crypto_markets:
            if market.ticker in self._traded_tickers:
                continue

            signal = await self._evaluate_crypto_market(
                market, current_prices
            )
            if signal is not None:
                signals.append(signal)

        return signals

    async def _evaluate_crypto_market(
        self,
        market: Market,
        current_prices: dict[str, float],
    ) -> LiveEdgeSignal | None:
        """Evaluate a single crypto market for edge."""
        assert self._crypto_estimator is not None

        parsed = _parse_crypto_market_title(market.title)
        if parsed is None:
            return None

        asset, threshold, direction = parsed

        # Map asset to trading symbol
        symbol = _asset_to_symbol(asset)
        if symbol is None or symbol not in current_prices:
            return None

        current_price = current_prices[symbol]

        # Compute hours remaining
        hours_remaining = max(
            0.0,
            (market.close_time - now_utc()).total_seconds() / 3600.0,
        )

        if hours_remaining <= 0:
            return None

        # Estimate probability
        if direction == "above":
            implied_prob = await self._crypto_estimator.estimate_prob(
                current_price, threshold, hours_remaining, symbol
            )
        else:
            # P(price < threshold) = 1 - P(price > threshold)
            prob_above = await self._crypto_estimator.estimate_prob(
                current_price, threshold, hours_remaining, symbol
            )
            implied_prob = 1.0 - prob_above

        if implied_prob < self._config.min_confidence:
            return None

        market_price = float(
            market.yes_ask if market.yes_ask > 0 else market.last_price
        )
        if market_price <= 0 or market_price >= 1:
            return None

        edge = implied_prob - market_price
        fee = float(
            self._fees.fee_per_contract(
                Decimal(str(market_price)), is_maker=False
            )
        )
        fee_adjusted = edge - fee

        if fee_adjusted < self._config.min_edge:
            return None

        # Distance from threshold as percentage
        pct_distance = abs(current_price - threshold) / threshold * 100

        # Urgency based on hours remaining and distance
        if hours_remaining < 2 and pct_distance > 3:
            urgency = "high"
        elif hours_remaining < 6:
            urgency = "medium"
        else:
            urgency = "low"

        return LiveEdgeSignal(
            ticker=market.ticker,
            category="crypto",
            implied_probability=implied_prob,
            market_probability=market_price,
            edge=round(edge, 6),
            fee_adjusted_edge=round(fee_adjusted, 6),
            confidence=implied_prob,
            reasoning=(
                f"{asset} at ${current_price:,.0f}, threshold "
                f"${threshold:,.0f} {direction}. "
                f"{pct_distance:.1f}% away, {hours_remaining:.1f}h remaining. "
                f"Implied prob {implied_prob:.0%}, market {market_price:.0%} "
                f"({edge:+.1%} edge)"
            ),
            data_snapshot={
                "asset": asset,
                "symbol": symbol,
                "current_price": current_price,
                "threshold": threshold,
                "direction": direction,
                "pct_distance": round(pct_distance, 2),
                "hours_remaining": round(hours_remaining, 2),
            },
            detected_at=now_utc(),
            urgency=urgency,
        )

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    async def evaluate_signal(
        self, signal: LiveEdgeSignal
    ) -> TradeDecision | None:
        """Run the full pipeline: fee adjustment, sizing, risk check.

        Only trade if:
        - fee_adjusted_edge > min_edge (default 5c for live events)
        - confidence > 0.90
        - sufficient liquidity
        - within risk limits
        """
        # Recheck edge threshold (in case config was updated)
        if signal.fee_adjusted_edge < self._config.min_edge:
            return None

        if signal.confidence < self._config.min_confidence:
            return None

        # Get the orderbook for precise pricing
        try:
            orderbook = await self._client.get_orderbook(signal.ticker)
        except Exception:
            log.warning(
                "live_edge.orderbook_error",
                ticker=signal.ticker,
                exc_info=True,
            )
            return None

        # Compute edge via the standard edge calculator
        edge_result = self._edge_calc.compute_edge(
            signal.implied_probability,
            orderbook,
            is_maker=False,  # live events need IOC for speed
        )

        if not edge_result.is_actionable:
            log.debug(
                "live_edge.not_actionable",
                ticker=signal.ticker,
                fee_adjusted_edge=edge_result.fee_adjusted_edge,
            )
            return TradeDecision(
                signal=signal,
                edge_result=edge_result,
                size_result=SizeResult(
                    kelly_fraction=0.0,
                    adjusted_fraction=0.0,
                    contracts=0,
                    dollar_risk=Decimal("0"),
                    dollar_ev=Decimal("0"),
                    capped_by="not_actionable",
                ),
                contracts=0,
                side=edge_result.side,
                price=edge_result.target_price,
                should_trade=False,
                rejection_reason="Edge not actionable after orderbook check",
            )

        capital_view = self._risk.get_capital_view()
        bankroll = capital_view.bankroll

        # Size the position
        size_result = self._sizer.size(
            edge_result=edge_result,
            bankroll=bankroll,
            model_confidence=signal.confidence,
            existing_exposure=capital_view.total_exposure,
            regime_adjustment=1.0,  # live events always trade in any regime
        )

        contracts = min(
            size_result.contracts, self._config.max_contracts_per_trade
        )

        if contracts <= 0:
            return TradeDecision(
                signal=signal,
                edge_result=edge_result,
                size_result=size_result,
                contracts=0,
                side=edge_result.side,
                price=edge_result.target_price,
                should_trade=False,
                rejection_reason=f"Kelly sizer returned 0 contracts (capped_by: {size_result.capped_by})",
            )

        # Risk check
        proposed = ProposedTrade(
            ticker=signal.ticker,
            category=signal.category,
            side=edge_result.side,
            action=edge_result.action,
            contracts=contracts,
            price=edge_result.target_price,
        )

        risk_result = self._risk.pre_trade_check(proposed)

        if not risk_result.approved:
            log.info(
                "live_edge.risk_rejected",
                ticker=signal.ticker,
                reason=risk_result.rejection_reason,
            )
            return TradeDecision(
                signal=signal,
                edge_result=edge_result,
                size_result=size_result,
                contracts=0,
                side=edge_result.side,
                price=edge_result.target_price,
                should_trade=False,
                rejection_reason=risk_result.rejection_reason,
            )

        # Apply risk-adjusted size
        final_contracts = risk_result.adjusted_size or contracts

        return TradeDecision(
            signal=signal,
            edge_result=edge_result,
            size_result=size_result,
            contracts=final_contracts,
            side=edge_result.side,
            price=edge_result.target_price,
            should_trade=True,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_signal(
        self, signal: LiveEdgeSignal, decision: TradeDecision
    ) -> None:
        """Execute using aggressive IOC strategy since timing matters.

        For live events, we use IOC (immediate-or-cancel) orders because:
        1. The edge is time-sensitive -- it can disappear quickly
        2. We want immediate fills, not resting orders
        3. The spread is less important than speed of execution
        """
        if not decision.should_trade or decision.contracts <= 0:
            return

        if self._risk.is_trading_paused():
            log.warning(
                "live_edge.global_pause_active",
                ticker=signal.ticker,
                reasons=self._risk.pause_reasons,
            )
            return

        # Mark the ticker to avoid duplicate trades
        self._traded_tickers.add(signal.ticker)

        side = Side.YES if decision.side == "yes" else Side.NO
        price = decision.price

        # If buying NO side, compute the YES price equivalent
        # Kalshi orders always specify yes_price
        if side == Side.NO:
            yes_price = Decimal("1") - price
        else:
            yes_price = price

        # Quantize to Kalshi's cent precision
        yes_price = yes_price.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        request = OrderRequest(
            ticker=signal.ticker,
            side=side,
            action=Action.BUY,
            count=decision.contracts,
            yes_price=yes_price,
            time_in_force=TimeInForce.IOC,  # aggressive fill
            post_only=False,
        )

        log.info(
            "live_edge.executing",
            ticker=signal.ticker,
            side=side.value,
            contracts=decision.contracts,
            price=str(yes_price),
            category=signal.category,
            edge=round(signal.edge, 4),
            reasoning=signal.reasoning,
        )

        reservation_key = (
            f"live_edge:{signal.ticker}:{signal.detected_at.isoformat()}:"
            f"{decision.side}"
        )
        reserved = self._risk.reserve_trade_intent(
            reservation_key,
            owner="live_edge",
            ticker=signal.ticker,
            category=signal.category,
            contracts=decision.contracts,
            price=decision.price,
        )
        if not reserved:
            log.warning(
                "live_edge.capital_reservation_rejected",
                ticker=signal.ticker,
                category=signal.category,
                contracts=decision.contracts,
                price=str(decision.price),
            )
            self._traded_tickers.discard(signal.ticker)
            return

        try:
            order = await self._orders.submit_order(request)
            self._trades_executed += 1
            log.info(
                "live_edge.executed",
                ticker=signal.ticker,
                order_id=order.order_id,
                status=order.status.value,
                contracts=decision.contracts,
                price=str(yes_price),
            )

            # Record fill in tracker for audit trail
            if self._fill_tracker is not None:
                try:
                    fills = await self._client.get_fills(ticker=signal.ticker)
                    for f in fills:
                        if (
                            f.side == side
                            and f.action == Action.BUY
                            and abs(f.price - yes_price) < Decimal("0.01")
                        ):
                            self._fill_tracker.on_closer_fill(
                                f,
                                strategy="live_edge",
                                signal_source=signal.category,
                                confidence=signal.confidence,
                                expected_profit=float(decision.edge_result.expected_value),
                                entry_price=float(yes_price),
                                contracts=decision.contracts,
                                category=signal.category,
                                signal_data={
                                    "edge": signal.edge,
                                    "fee_adjusted_edge": signal.fee_adjusted_edge,
                                    "reasoning": signal.reasoning,
                                    "implied_probability": signal.implied_probability,
                                },
                            )
                            break
                except Exception:
                    log.warning(
                        "live_edge.fill_tracker_error",
                        ticker=signal.ticker,
                        exc_info=True,
                    )
        except Exception:
            log.error(
                "live_edge.execution_error",
                ticker=signal.ticker,
                exc_info=True,
            )
            # Remove from traded set so we can retry
            self._traded_tickers.discard(signal.ticker)
        finally:
            self._risk.release_trade_intent(reservation_key)

        try:
            self._risk.sync_open_order_reservations(
                self._orders.get_open_orders(),
                category_lookup={signal.ticker: signal.category},
            )
        except Exception:
            log.warning("live_edge.capital_sync_failed", ticker=signal.ticker, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function.

    Avoids importing scipy for a single function.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _is_weather_market(market: Market) -> bool:
    """Check if a market is weather-related using ticker patterns and keywords.

    Real Kalshi weather tickers follow patterns like:
    - KXTEMPNYCH-26APR0909-T47.99  (temperature)
    - KXHIGHNY-26APR09-T80         (high temp)
    - KXA100MON-26APR0912-T90.99   (threshold markets)
    """
    ticker_upper = market.ticker.upper()
    # Check ticker prefixes for known weather patterns
    weather_ticker_prefixes = ("KXTEMP", "KXHIGH", "KXLOW", "KXPRECIP", "KXWEATHER")
    if any(ticker_upper.startswith(p) for p in weather_ticker_prefixes):
        return True

    # Also match general threshold tickers with T-suffix that relate to weather
    # by checking the category or title
    title = market.title.lower()
    weather_keywords = [
        "temperature",
        "degrees",
        "fahrenheit",
        "celsius",
        "weather",
        "rain",
        "precipitation",
        "wind speed",
        "snow",
        "heat",
        "cold",
        "high temp",
        "low temp",
    ]
    category = market.category.lower()
    return (
        any(kw in title for kw in weather_keywords)
        or "weather" in category
        or "climate" in category
    )


def _is_crypto_market(market: Market) -> bool:
    """Check if a market is crypto-related using ticker patterns and keywords.

    Crypto tickers on Kalshi may use prefixes like KXBTC, KXETH, KXSOL,
    or contain crypto-related keywords in the title/category.
    """
    ticker_upper = market.ticker.upper()
    crypto_ticker_prefixes = ("KXBTC", "KXETH", "KXSOL", "KXCRYPTO")
    if any(ticker_upper.startswith(p) for p in crypto_ticker_prefixes):
        return True

    title = market.title.lower()
    crypto_keywords = [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "solana",
        "sol",
    ]
    category = market.category.lower()
    return (
        any(kw in title for kw in crypto_keywords)
        or "crypto" in category
    )


def _parse_weather_ticker(
    ticker: str,
) -> tuple[str, float, str] | None:
    """Extract (location, threshold, direction) from a Kalshi weather ticker.

    Real ticker formats:
    - KXTEMPNYCH-26APR0909-T47.99  -> ("New York", 47.99, "above")
    - KXHIGHNY-26APR09-T80        -> ("New York", 80.0, "above")
    - KXLOWCHI-26APR09-T32        -> ("Chicago", 32.0, "below")

    Returns None if the ticker doesn't match a known weather pattern.
    """
    ticker_upper = ticker.upper()

    # Match KXTEMP{location}-{date}-T{threshold} or KXHIGH{loc}-{date}-T{thresh}
    m = re.match(
        r"^KX(TEMP|HIGH|LOW)(\w+?)-(\d{2}[A-Z]{3}\d{2,6})-T([\d.]+)$",
        ticker_upper,
    )
    if m is None:
        # Try general threshold pattern: KX{anything}-{date}-T{threshold}
        m = re.match(
            r"^KX(\w+)-(\d{2}[A-Z]{3}\d{2,6})-T([\d.]+)$",
            ticker_upper,
        )
        if m is None:
            return None
        prefix = m.group(1)
        threshold = float(m.group(3))
        # Infer location from prefix if possible
        location = _infer_location_from_ticker_prefix(prefix)
        if location is None:
            return None
        return location, threshold, "above"

    measure_type = m.group(1)  # TEMP, HIGH, or LOW
    location_code = m.group(2)  # "NYCH", "NY", "CHI", etc.
    threshold = float(m.group(4))

    # Map location code to city name
    location = _infer_location_from_ticker_prefix(location_code)
    if location is None:
        location = location_code.title()

    # HIGH -> "above", LOW -> "below", TEMP -> "above" (default)
    direction = "below" if measure_type == "LOW" else "above"

    return location, threshold, direction


def _infer_location_from_ticker_prefix(prefix: str) -> str | None:
    """Map a ticker location code to a city name.

    Examples: NYCH -> New York, NY -> New York, CHI -> Chicago
    """
    prefix_upper = prefix.upper()
    location_map = {
        "NYCH": "New York",
        "NYC": "New York",
        "NY": "New York",
        "CHI": "Chicago",
        "LA": "Los Angeles",
        "MIA": "Miami",
        "HOU": "Houston",
        "PHX": "Phoenix",
        "DAL": "Dallas",
        "DEN": "Denver",
        "ATL": "Atlanta",
        "BOS": "Boston",
        "SEA": "Seattle",
        "SF": "San Francisco",
        "DC": "Washington",
        "PHL": "Philadelphia",
        "MSP": "Minneapolis",
        "DET": "Detroit",
        "TPA": "Tampa",
        "ORL": "Orlando",
        "AUS": "Austin",
        "LAS": "Las Vegas",
        "PDX": "Portland",
    }
    for code, city in location_map.items():
        if prefix_upper.startswith(code) or prefix_upper.endswith(code):
            return city
    return None


def _parse_weather_market_title(
    title: str,
) -> tuple[str, float, str] | None:
    """Extract (location, threshold_f, direction) from a weather market title.

    Examples of Kalshi weather market titles:
    - "Will the high in Chicago be above 90F on April 10?"
    - "NYC high temperature above 80 degrees?"

    Returns None if the title can't be parsed.
    """
    title_lower = title.lower()

    # Common US city names in Kalshi weather markets
    cities = [
        "chicago",
        "new york",
        "nyc",
        "los angeles",
        "la",
        "miami",
        "houston",
        "phoenix",
        "dallas",
        "denver",
        "atlanta",
        "boston",
        "seattle",
        "san francisco",
        "washington",
        "dc",
        "philadelphia",
        "minneapolis",
        "detroit",
        "tampa",
        "orlando",
        "austin",
        "nashville",
        "charlotte",
        "las vegas",
        "portland",
        "milwaukee",
        "sacramento",
    ]

    location: str | None = None
    for city in cities:
        if city in title_lower:
            location = city.title()
            break

    if location is None:
        return None

    # Extract temperature threshold
    # Match patterns like "above 90", "90F", "90 degrees", "above 80"
    temp_pattern = r"(above|below|over|under|exceed)\s+(\d+)"
    match = re.search(temp_pattern, title_lower)
    if match is None:
        # Try just a number followed by F or degrees
        match = re.search(r"(\d+)\s*(?:f|degrees|fahrenheit|\u00b0)", title_lower)
        if match is None:
            return None
        threshold = float(match.group(1))
        # Default to "above" for temp markets
        direction = "above"
    else:
        direction_word = match.group(1)
        threshold = float(match.group(2))
        direction = (
            "above"
            if direction_word in ("above", "over", "exceed")
            else "below"
        )

    return location, threshold, direction


def _parse_crypto_market_title(
    title: str,
) -> tuple[str, float, str] | None:
    """Extract (asset, threshold, direction) from a crypto market title.

    Examples:
    - "Will Bitcoin be above $80,000 on April 10?"
    - "ETH price above $3,000?"

    Returns None if the title can't be parsed.
    """
    title_lower = title.lower()

    # Identify asset
    asset: str | None = None
    asset_map = {
        "bitcoin": "BTC",
        "btc": "BTC",
        "ethereum": "ETH",
        "eth": "ETH",
        "solana": "SOL",
        "sol": "SOL",
    }
    for keyword, symbol in asset_map.items():
        if keyword in title_lower:
            asset = symbol
            break

    if asset is None:
        return None

    # Extract price threshold
    # Match patterns like "$80,000", "$80000", "80,000", "80K"
    price_pattern = r"\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?"
    direction_pattern = r"(above|below|over|under|exceed|higher|lower)\s+"

    dir_match = re.search(direction_pattern, title_lower)
    direction = "above"  # default
    if dir_match:
        dir_word = dir_match.group(1)
        direction = (
            "above"
            if dir_word in ("above", "over", "exceed", "higher")
            else "below"
        )

    # Find dollar amounts
    amounts: list[float] = []
    for m in re.finditer(r"\$?([\d,]+(?:\.\d+)?)\s*(k|K)?", title):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
            if m.group(2):  # K suffix
                val *= 1000
            if val > 100:  # Filter out small numbers that aren't prices
                amounts.append(val)
        except ValueError:
            continue

    if not amounts:
        return None

    threshold = amounts[0]
    return asset, threshold, direction


def _asset_to_symbol(asset: str) -> str | None:
    """Map a crypto asset code to a ccxt trading symbol."""
    mapping = {
        "BTC": "BTC/USDT",
        "ETH": "ETH/USDT",
        "SOL": "SOL/USDT",
    }
    return mapping.get(asset)
