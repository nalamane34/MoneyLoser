"""Resolution sniping strategy for Kalshi prediction markets.

When a real-world event outcome becomes KNOWN (game ends, weather threshold
hit, economic number released), Kalshi markets take seconds to minutes to
fully reprice to 99c/1c.  If we detect the outcome first via real-time data
feeds, we buy the winning side at 92-97c for near-risk-free profit minus fees.

This module ties together the ESPN, weather, and economic data feeds with
Kalshi market execution.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Union

import structlog

from moneygone.data.economic.releases import EconomicSignal
from moneygone.data.sports.espn import ESPNLiveFeed, GameState, OutcomeSignal
from moneygone.data.sports.live_weather import (
    LiveWeatherFeed,
    ThresholdSignal,
    WeatherObservation,
)
from moneygone.exchange.types import (
    Action,
    Fill,
    Market,
    MarketStatus,
    OrderRequest,
    Side,
    TimeInForce,
)
from moneygone.utils.time import now_utc

if TYPE_CHECKING:
    from moneygone.exchange.rest_client import KalshiRestClient
    from moneygone.execution.fill_tracker import FillTracker
    from moneygone.execution.order_manager import OrderManager
    from moneygone.signals.fees import KalshiFeeCalculator

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")

# Type alias for any signal from our data feeds.
AnySignal = Union[OutcomeSignal, ThresholdSignal, EconomicSignal]


# ---------------------------------------------------------------------------
# Configuration and mapping dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractMapping:
    """Maps a Kalshi contract to a real-world data source."""

    ticker: str
    category: str  # "sports", "weather", "economic"
    data_source: str  # "espn", "noaa", "open_meteo", "fred"
    source_params: dict[str, Any]
    resolution_field: str  # "winner", "temperature", "cpi", etc.
    threshold: float | None = None
    direction: str | None = None  # "above" or "below"


@dataclass
class SnipeOpportunity:
    """A detected opportunity to snipe a resolving contract."""

    ticker: str
    outcome_known: bool
    predicted_resolution: str  # "yes" or "no"
    confidence: float
    current_market_price: Decimal  # current YES price (best ask)
    expected_payout: Decimal  # $1.00 if correct
    expected_profit: Decimal  # payout - cost - fees
    signal_source: str
    signal_data: dict[str, Any]
    detected_at: datetime


@dataclass
class SnipeConfig:
    """Configuration parameters for the resolution sniper."""

    min_confidence: float = 0.95
    max_entry_price: float = 0.97
    min_entry_price: float = 0.80
    min_profit_after_fees: float = 0.005
    max_contracts_per_snipe: int = 50
    max_daily_snipes: int = 20
    cooldown_seconds: float = 5.0
    max_exposure_per_event: float = 20.0  # max $ exposure per game/event


@dataclass
class SnipeRecord:
    """Record of a snipe attempt for tracking and analysis."""

    ticker: str
    side: str
    contracts: int
    entry_price: Decimal
    expected_profit: Decimal
    signal_source: str
    confidence: float
    executed_at: datetime
    fill: Fill | None = None
    settled_correctly: bool | None = None  # populated after settlement
    actual_profit: Decimal | None = None


# ---------------------------------------------------------------------------
# Ticker pattern matching for auto-discovery
# ---------------------------------------------------------------------------

# Common Kalshi ticker patterns.  These regex patterns attempt to extract
# the category and relevant identifiers from Kalshi tickers.
#
# Real Kalshi ticker formats discovered from the live API:
#   Weather:  KXTEMPNYCH-26APR0909-T47.99  (NYC temperature)
#             KXHIGHNY-26APR09-T80         (NYC high temp)
#             KXA100MON-26APR0912-T90.99   (other threshold markets)
#   Sports:   KXMVESPORTSMULTIGAMEEXTENDED-S{hash}-{hash}  (multivariate sports)
#             KXMLBHRR-26APR091210CINMIA-{player}-{threshold}  (MLB hits+runs+RBIs)
#             KXMLBTB-26APR091210CINMIA-{player}-{threshold}   (MLB total bases)
#   Cross-cat: KXMVECROSSCATEGORY-S{hash}-{hash}  (multivariate cross-category)
_TICKER_PATTERNS: dict[str, dict[str, Any]] = {
    # Weather temperature: KXTEMPNYCH-26APR0909-T47.99
    r"^KX(TEMP\w+)-(\d{2}[A-Z]{3}\d{4})-T([\d.]+)": {
        "category": "weather",
        "data_source": "noaa",
    },
    # Weather high/low: KXHIGHNY-26APR09-T80
    r"^KX(HIGH|LOW)\w*-(\d{2}[A-Z]{3}\d{2,4})-T([\d.]+)": {
        "category": "weather",
        "data_source": "noaa",
    },
    # MLB player props: KXMLBHRR-26APR091210CINMIA-{player}-{threshold}
    r"^KX(MLB(?:HRR|TB|HR|SO|H|BB))-(\d{2}[A-Z]{3}\d+[A-Z]+)-([A-Za-z]+)-": {
        "category": "sports",
        "data_source": "espn",
    },
    # Multivariate sports: KXMVESPORTSMULTIGAMEEXTENDED-S{hash}-{hash}
    r"^KX(MVESPORTS\w+)-S": {
        "category": "sports",
        "data_source": "espn",
    },
    # Multivariate cross-category: KXMVECROSSCATEGORY-S{hash}-{hash}
    r"^KX(MVECROSSCATEGORY)-S": {
        "category": "sports",
        "data_source": "espn",
    },
    # General threshold markets: KXA100MON-26APR0912-T90.99
    r"^KX(\w+)-(\d{2}[A-Z]{3}\d{2,4})-T([\d.]+)": {
        "category": "weather",
        "data_source": "noaa",
    },
    # Legacy sports: KXNBA-*, KXNFL-*, KXMLB-*, KXNHL-*
    r"^KX(NBA|NFL|MLB|NHL|MLS|UFC)": {
        "category": "sports",
        "data_source": "espn",
    },
    # Economic: KXCPI-*, KXUNRATE-*, KXGDP-*, KXFED-*
    r"^KX(CPI|UNRATE|GDP|FED|ECON|JOBS)": {
        "category": "economic",
        "data_source": "fred",
    },
}

# League name mapping for ESPN.
_LEAGUE_TO_ESPN: dict[str, tuple[str, str]] = {
    "NBA": ("basketball", "nba"),
    "NFL": ("football", "nfl"),
    "MLB": ("baseball", "mlb"),
    "NHL": ("hockey", "nhl"),
    "MLS": ("soccer", "usa.1"),
    "UFC": ("mma", "ufc"),
}


# ---------------------------------------------------------------------------
# Resolution Sniper
# ---------------------------------------------------------------------------


class ResolutionSniper:
    """Monitors real-time data feeds and snipes Kalshi contracts when outcomes
    are known but markets have not fully repriced.

    Strategy: When an outcome is known with high confidence (>95%), buy the
    winning side if the market price is below 97c.
    Expected profit = $1.00 - price - taker_fee.

    Parameters
    ----------
    rest_client:
        Authenticated Kalshi REST client.
    order_manager:
        Order lifecycle manager.
    fee_calculator:
        Kalshi fee calculator for profit estimation.
    contract_mappings:
        Initial list of contract-to-data-source mappings.  Additional
        mappings can be auto-discovered from market tickers.
    config:
        Snipe strategy parameters.
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        order_manager: OrderManager,
        fee_calculator: KalshiFeeCalculator,
        contract_mappings: list[ContractMapping] | None = None,
        config: SnipeConfig | None = None,
        fill_tracker: FillTracker | None = None,
    ) -> None:
        self._client = rest_client
        self._order_manager = order_manager
        self._fees = fee_calculator
        self._config = config or SnipeConfig()
        self._fill_tracker = fill_tracker
        self._mappings: dict[str, ContractMapping] = {
            m.ticker: m for m in (contract_mappings or [])
        }

        # Data feeds.
        self._espn = ESPNLiveFeed()
        self._weather = LiveWeatherFeed()

        # State tracking.
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._snipe_history: list[SnipeRecord] = []
        self._daily_snipe_count = 0
        self._daily_reset_date: str = ""
        self._last_snipe_time: dict[str, datetime] = {}
        self._opportunity_log: list[SnipeOpportunity] = []
        # Per-event exposure tracking: event_ticker -> total $ spent
        self._event_exposure: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start monitoring all data feeds and scanning for opportunities."""
        if self._running:
            logger.warning("sniper.already_running")
            return

        self._running = True
        logger.info(
            "sniper.starting",
            mappings=len(self._mappings),
            config_min_confidence=self._config.min_confidence,
            config_max_price=self._config.max_entry_price,
        )

        # Auto-discover mappings from current Kalshi markets.
        await self._auto_discover_mappings()

        # Group mappings by data source and start appropriate feeds.
        sports_leagues: set[tuple[str, str]] = set()
        weather_stations: list[dict[str, Any]] = []

        for mapping in self._mappings.values():
            if mapping.category == "sports" and mapping.data_source == "espn":
                sport = mapping.source_params.get("sport", "")
                league = mapping.source_params.get("league", "")
                if sport and league:
                    sports_leagues.add((sport, league))

            elif mapping.category == "weather":
                station_cfg: dict[str, Any] = {}
                station_id = mapping.source_params.get("station_id")
                lat = mapping.source_params.get("lat")
                lon = mapping.source_params.get("lon")
                if station_id:
                    station_cfg["station_id"] = station_id
                if lat is not None and lon is not None:
                    station_cfg["lat"] = lat
                    station_cfg["lon"] = lon
                if mapping.threshold is not None:
                    station_cfg["thresholds"] = [
                        {
                            "variable": mapping.resolution_field,
                            "threshold": mapping.threshold,
                            "direction": mapping.direction or "above",
                        }
                    ]
                if station_cfg:
                    weather_stations.append(station_cfg)

        # Start ESPN polling for each league.
        for sport, league in sports_leagues:
            task = await self._espn.start_polling(
                sport=sport,
                league=league,
                interval_seconds=10.0,
                callback=self._on_espn_update,
            )
            self._tasks.append(task)

        # Start weather polling.
        if weather_stations:
            task = await self._weather.start_polling(
                stations=weather_stations,
                interval_seconds=60.0,
                callback=self._on_weather_update,
            )
            self._tasks.append(task)

        logger.info(
            "sniper.started",
            sports_leagues=len(sports_leagues),
            weather_stations=len(weather_stations),
        )

    async def stop(self) -> None:
        """Stop all monitoring and clean up."""
        self._running = False

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        await self._espn.close()
        await self._weather.close()

        logger.info(
            "sniper.stopped",
            total_snipes=len(self._snipe_history),
            opportunities_logged=len(self._opportunity_log),
        )

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    async def on_outcome_detected(self, signal: AnySignal) -> None:
        """Called when a data feed detects a known outcome.

        1. Match signal to contract mapping.
        2. Determine predicted resolution (yes/no).
        3. Check current market price.
        4. Calculate profit after fees.
        5. If profitable, execute immediately.

        Parameters
        ----------
        signal:
            An outcome signal from any data feed.
        """
        # Reset daily counter if needed.
        self._check_daily_reset()

        if self._daily_snipe_count >= self._config.max_daily_snipes:
            logger.warning(
                "sniper.daily_limit_reached",
                count=self._daily_snipe_count,
                limit=self._config.max_daily_snipes,
            )
            return

        # Find matching contract(s).
        matched_mappings = self._find_matching_mappings(signal)
        if not matched_mappings:
            logger.debug(
                "sniper.no_mapping_match",
                signal_type=type(signal).__name__,
            )
            return

        for mapping in matched_mappings:
            # Check cooldown.
            last_time = self._last_snipe_time.get(mapping.ticker)
            if last_time is not None:
                elapsed = (now_utc() - last_time).total_seconds()
                if elapsed < self._config.cooldown_seconds:
                    logger.debug(
                        "sniper.cooldown",
                        ticker=mapping.ticker,
                        elapsed=round(elapsed, 1),
                        cooldown=self._config.cooldown_seconds,
                    )
                    continue

            # Determine predicted resolution.
            predicted = self._predict_resolution(signal, mapping)
            if predicted is None:
                continue

            confidence = self._get_signal_confidence(signal)
            if confidence < self._config.min_confidence:
                logger.debug(
                    "sniper.low_confidence",
                    ticker=mapping.ticker,
                    confidence=confidence,
                    min_required=self._config.min_confidence,
                )
                continue

            # Get current market price.
            try:
                market = await self._client.get_market(mapping.ticker)
            except Exception:
                logger.warning(
                    "sniper.market_fetch_error",
                    ticker=mapping.ticker,
                    exc_info=True,
                )
                continue

            if market.status != MarketStatus.OPEN:
                logger.debug(
                    "sniper.market_not_open",
                    ticker=mapping.ticker,
                    status=market.status.value,
                )
                continue

            # Calculate opportunity.
            opportunity = self._evaluate_opportunity(
                ticker=mapping.ticker,
                predicted_resolution=predicted,
                market=market,
                confidence=confidence,
                signal=signal,
            )

            # Log every opportunity, even if not traded.
            self._opportunity_log.append(opportunity)
            logger.info(
                "sniper.opportunity_found",
                ticker=opportunity.ticker,
                predicted=opportunity.predicted_resolution,
                price=str(opportunity.current_market_price),
                profit=str(opportunity.expected_profit),
                confidence=opportunity.confidence,
                outcome_known=opportunity.outcome_known,
            )

            # Execute if profitable.
            if self._should_execute(opportunity):
                fill = await self.execute_snipe(opportunity)
                if fill is not None:
                    self._daily_snipe_count += 1
                    self._last_snipe_time[mapping.ticker] = now_utc()

    async def _on_espn_update(
        self,
        changed_games: list[GameState],
        new_outcomes: list[OutcomeSignal],
    ) -> None:
        """Callback from ESPN polling: process new outcome signals."""
        for outcome in new_outcomes:
            await self.on_outcome_detected(outcome)

    async def _on_weather_update(
        self,
        observations: list[WeatherObservation],
        signals: list[ThresholdSignal],
    ) -> None:
        """Callback from weather polling: process threshold signals."""
        for signal in signals:
            await self.on_outcome_detected(signal)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    async def scan_for_opportunities(self) -> list[SnipeOpportunity]:
        """Scan all mapped contracts for current snipe opportunities.

        This performs an immediate check across all data feeds rather than
        waiting for polling callbacks.

        Returns
        -------
        list[SnipeOpportunity]
            Currently actionable opportunities.
        """
        opportunities: list[SnipeOpportunity] = []

        # Check sports.
        checked_leagues: set[tuple[str, str]] = set()
        for mapping in self._mappings.values():
            if mapping.category != "sports":
                continue
            sport = mapping.source_params.get("sport", "")
            league = mapping.source_params.get("league", "")
            league_key = (sport, league)
            if league_key in checked_leagues or not sport or not league:
                continue
            checked_leagues.add(league_key)

            games = await self._espn.get_live_scores(sport, league)
            for game in games:
                signal = self._espn.detect_outcome(game)
                if signal is not None:
                    matched = self._find_matching_mappings(signal)
                    for m in matched:
                        predicted = self._predict_resolution(signal, m)
                        if predicted is None:
                            continue

                        try:
                            market = await self._client.get_market(m.ticker)
                        except Exception:
                            continue

                        if market.status != MarketStatus.OPEN:
                            continue

                        opp = self._evaluate_opportunity(
                            ticker=m.ticker,
                            predicted_resolution=predicted,
                            market=market,
                            confidence=signal.confidence,
                            signal=signal,
                        )
                        if self._should_execute(opp):
                            opportunities.append(opp)
                        self._opportunity_log.append(opp)

        # Check weather: done via polling, but we can force a single cycle.
        for mapping in self._mappings.values():
            if mapping.category != "weather":
                continue
            station_id = mapping.source_params.get("station_id")
            lat = mapping.source_params.get("lat")
            lon = mapping.source_params.get("lon")

            obs: WeatherObservation | None = None
            if lat is not None and lon is not None:
                obs = await self._weather.get_current_observation_open_meteo(
                    lat, lon, station_id=station_id or ""
                )
            elif station_id:
                obs = await self._weather.get_current_observation(station_id)

            if obs is None or mapping.threshold is None:
                continue

            signal = self._weather.check_threshold(
                obs,
                threshold=mapping.threshold,
                variable=mapping.resolution_field,
                direction=mapping.direction or "above",
            )
            if signal is None:
                continue

            predicted = self._predict_resolution(signal, mapping)
            if predicted is None:
                continue

            try:
                market = await self._client.get_market(mapping.ticker)
            except Exception:
                continue

            if market.status != MarketStatus.OPEN:
                continue

            opp = self._evaluate_opportunity(
                ticker=mapping.ticker,
                predicted_resolution=predicted,
                market=market,
                confidence=signal.confidence,
                signal=signal,
            )
            if self._should_execute(opp):
                opportunities.append(opp)
            self._opportunity_log.append(opp)

        logger.info(
            "sniper.scan_complete",
            total_opportunities=len(opportunities),
        )
        return opportunities

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_snipe(self, opportunity: SnipeOpportunity) -> Fill | None:
        """Execute a snipe trade.

        Uses aggressive execution (IOC limit order at best ask) since speed
        matters more than fee savings for resolution sniping.

        Sizes conservatively: ``max_contracts_per_snipe`` or risk limit,
        whichever is lower.

        Parameters
        ----------
        opportunity:
            The snipe opportunity to execute.

        Returns
        -------
        Fill | None
            The resulting fill, or ``None`` if execution failed.
        """
        contracts = min(
            self._config.max_contracts_per_snipe,
            self._estimate_safe_size(opportunity),
        )
        if contracts <= 0:
            logger.warning(
                "sniper.zero_contracts",
                ticker=opportunity.ticker,
            )
            return None

        # Determine side: if we predict "yes", buy YES; if "no", buy NO.
        if opportunity.predicted_resolution == "yes":
            side = Side.YES
            price = opportunity.current_market_price
        else:
            side = Side.NO
            # For buying NO, the price is (1 - yes_price).
            price = _ONE - opportunity.current_market_price

        # Validate price range.
        if price <= _ZERO or price >= _ONE:
            logger.warning(
                "sniper.invalid_price",
                ticker=opportunity.ticker,
                price=str(price),
            )
            return None

        request = OrderRequest(
            ticker=opportunity.ticker,
            side=side,
            action=Action.BUY,
            count=contracts,
            yes_price=price if side == Side.YES else _ONE - price,
            time_in_force=TimeInForce.IOC,
            post_only=False,
        )

        logger.info(
            "sniper.executing",
            ticker=opportunity.ticker,
            side=side.value,
            contracts=contracts,
            price=str(price),
            expected_profit=str(opportunity.expected_profit),
        )

        try:
            order = await self._order_manager.submit_order(request)
        except Exception:
            logger.error(
                "sniper.execution_failed",
                ticker=opportunity.ticker,
                exc_info=True,
            )
            return None

        # Attempt to get the fill for this order.
        fill: Fill | None = None
        try:
            fills = await self._client.get_fills(ticker=opportunity.ticker)
            for f in fills:
                if (
                    f.side == side
                    and f.action == Action.BUY
                    and abs(f.price - price) < Decimal("0.01")
                ):
                    fill = f
                    break
        except Exception:
            logger.warning(
                "sniper.fill_fetch_error",
                ticker=opportunity.ticker,
                exc_info=True,
            )

        # Track per-event exposure.
        event_ticker = self._event_ticker_from_ticker(opportunity.ticker)
        cost = float(price) * contracts
        self._event_exposure[event_ticker] = (
            self._event_exposure.get(event_ticker, 0.0) + cost
        )

        # Record the snipe.
        record = SnipeRecord(
            ticker=opportunity.ticker,
            side=side.value,
            contracts=contracts,
            entry_price=price,
            expected_profit=opportunity.expected_profit,
            signal_source=opportunity.signal_source,
            confidence=opportunity.confidence,
            executed_at=now_utc(),
            fill=fill,
        )
        self._snipe_history.append(record)

        # Record in the shared fill tracker for audit trail
        if fill is not None and self._fill_tracker is not None:
            self._fill_tracker.on_closer_fill(
                fill,
                strategy="sniper",
                signal_source=opportunity.signal_source,
                confidence=opportunity.confidence,
                expected_profit=float(opportunity.expected_profit),
                entry_price=float(price),
                contracts=contracts,
                category=opportunity.signal_data.get("category", "unknown"),
                signal_data=opportunity.signal_data,
            )

        logger.info(
            "sniper.executed",
            ticker=opportunity.ticker,
            side=side.value,
            contracts=contracts,
            price=str(price),
            order_id=order.order_id,
            fill_obtained=fill is not None,
            event_exposure=round(self._event_exposure.get(event_ticker, 0.0), 2),
        )

        return fill

    # ------------------------------------------------------------------
    # Mapping and matching
    # ------------------------------------------------------------------

    def match_signal_to_contract(self, signal: AnySignal) -> ContractMapping | None:
        """Find the first contract mapping that corresponds to a signal.

        Parameters
        ----------
        signal:
            An outcome, threshold, or economic signal.

        Returns
        -------
        ContractMapping | None
        """
        matched = self._find_matching_mappings(signal)
        return matched[0] if matched else None

    @staticmethod
    def _extract_ticker_date(ticker: str) -> str | None:
        """Extract the date portion from a Kalshi ticker.

        Examples:
            KXNHLTOTAL-26APR13DETTB-5  -> "26APR13"
            KXMLBGAME-26APR111310MIADET-MIA -> "26APR11"
            KXNHLGAME-26APR12PITWSH-PIT -> "26APR12"

        Returns the YYMONDD portion or None if not parseable.
        """
        # Look for YYMONDD pattern (e.g. 26APR13, 26APR11)
        match = re.search(r"(\d{2}[A-Z]{3}\d{2})", ticker.upper())
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _signal_date_str(signal: "OutcomeSignal") -> str | None:
        """Get the YYMONDD string for when a signal's game occurred."""
        # detected_at is when the outcome was observed (≈ game end time)
        dt = signal.detected_at
        if dt is None:
            return None
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        return f"{dt.year % 100}{months[dt.month - 1]}{dt.day:02d}"

    def _find_matching_mappings(self, signal: AnySignal) -> list[ContractMapping]:
        """Find all contract mappings that match a signal.

        SAFETY: For sports signals, requires BOTH:
          1. Same sport (basketball->basketball, hockey->hockey, etc.)
          2. Ticker date matches the signal date (game day)
        This prevents cross-sport and cross-date false matches.
        """
        results: list[ContractMapping] = []

        for mapping in self._mappings.values():
            if isinstance(signal, OutcomeSignal) and mapping.category == "sports":
                # ---- SPORT VALIDATION ----
                # The mapping's sport must match the signal's sport.
                # e.g. an NBA outcome must NOT match an NHL contract.
                mapping_sport = mapping.source_params.get("sport", "").lower()
                signal_sport = (signal.sport or "").lower()

                if not mapping_sport or not signal_sport:
                    continue  # Can't validate sport — skip for safety

                # Normalize sport names for comparison
                sport_aliases = {
                    "basketball": "basketball", "nba": "basketball",
                    "football": "football", "nfl": "football",
                    "baseball": "baseball", "mlb": "baseball",
                    "hockey": "hockey", "nhl": "hockey",
                    "soccer": "soccer", "mls": "soccer",
                    "mma": "mma", "ufc": "mma",
                    "tennis": "tennis",
                    "golf": "golf",
                }
                norm_mapping = sport_aliases.get(mapping_sport, mapping_sport)
                norm_signal = sport_aliases.get(signal_sport, signal_sport)

                if norm_mapping != norm_signal:
                    continue  # Different sport — skip

                # ---- DATE VALIDATION ----
                # The ticker's date must match the game date.
                # e.g. a game finishing today (APR11) must NOT match
                # a contract for APR13.
                ticker_date = self._extract_ticker_date(mapping.ticker)
                signal_date = self._signal_date_str(signal)

                if ticker_date and signal_date and ticker_date != signal_date:
                    logger.debug(
                        "sniper.date_mismatch",
                        ticker=mapping.ticker,
                        ticker_date=ticker_date,
                        signal_date=signal_date,
                    )
                    continue  # Wrong date — skip

                # ---- TEAM / GAME-ID MATCHING ----
                teams_in_signal = {
                    signal.home_team.lower(),
                    signal.away_team.lower(),
                }
                team_param = mapping.source_params.get("team", "").lower()
                game_id_param = mapping.source_params.get("game_id", "")

                if game_id_param and game_id_param == signal.game_id:
                    results.append(mapping)
                elif team_param and team_param in " ".join(teams_in_signal):
                    results.append(mapping)
                elif not team_param and not game_id_param:
                    # Generic sports mapping for this league.
                    league_param = mapping.source_params.get("league", "")
                    signal_league = (signal.league or "").lower()
                    if league_param and league_param == signal_league:
                        results.append(mapping)

            elif isinstance(signal, ThresholdSignal) and mapping.category == "weather":
                if mapping.source_params.get("station_id", "") == signal.station_id:
                    if mapping.resolution_field == signal.variable:
                        results.append(mapping)

            elif isinstance(signal, EconomicSignal) and mapping.category == "economic":
                if mapping.source_params.get("series_id", "") == signal.series_id:
                    results.append(mapping)

        return results

    # ------------------------------------------------------------------
    # Resolution prediction
    # ------------------------------------------------------------------

    def _predict_resolution(
        self, signal: AnySignal, mapping: ContractMapping
    ) -> str | None:
        """Predict whether a contract resolves YES or NO based on the signal.

        Returns
        -------
        str | None
            ``"yes"`` or ``"no"``, or ``None`` if indeterminate.
        """
        if isinstance(signal, OutcomeSignal):
            return self._predict_sports_resolution(signal, mapping)
        elif isinstance(signal, ThresholdSignal):
            return self._predict_weather_resolution(signal, mapping)
        elif isinstance(signal, EconomicSignal):
            return self._predict_economic_resolution(signal, mapping)
        return None

    def _predict_sports_resolution(
        self, signal: OutcomeSignal, mapping: ContractMapping
    ) -> str | None:
        """Predict sports contract resolution.

        The mapping's ``resolution_field`` is ``"winner"`` and ``source_params``
        contains a ``"team"`` key indicating which team the YES side represents.
        """
        team = mapping.source_params.get("team", "").lower()
        if not team:
            return None

        if signal.outcome == "home_win":
            winning_team = signal.home_team.lower()
        elif signal.outcome == "away_win":
            winning_team = signal.away_team.lower()
        else:
            # Draw -- neither team wins.
            return "no"

        # Does the contract's team match the winner?
        if team in winning_team or winning_team in team:
            return "yes"
        else:
            return "no"

    def _predict_weather_resolution(
        self, signal: ThresholdSignal, mapping: ContractMapping
    ) -> str | None:
        """Predict weather contract resolution.

        If the contract asks "will temperature exceed X?" and the threshold
        is exceeded, the contract resolves YES.
        """
        contract_direction = mapping.direction or "above"

        if signal.direction == contract_direction and signal.exceeded:
            return "yes"
        elif signal.direction != contract_direction and signal.exceeded:
            return "no"
        return None

    def _predict_economic_resolution(
        self, signal: EconomicSignal, mapping: ContractMapping
    ) -> str | None:
        """Predict economic contract resolution."""
        contract_direction = mapping.direction or "above"

        if signal.direction == contract_direction and signal.exceeded:
            return "yes"
        elif signal.direction != contract_direction and signal.exceeded:
            return "no"
        return None

    # ------------------------------------------------------------------
    # Profit calculation
    # ------------------------------------------------------------------

    def calculate_profit(
        self, price: Decimal, is_taker: bool = True
    ) -> Decimal:
        """Net profit per contract: $1.00 - price - fee.

        Parameters
        ----------
        price:
            Entry price per contract (0-1 range).
        is_taker:
            Whether the order crosses the spread (True for IOC).

        Returns
        -------
        Decimal
            Net profit per contract.  Must be positive for a viable snipe.
        """
        fee = self._fees.fee_per_contract(price, is_maker=not is_taker)
        return (_ONE - price - fee).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    def _evaluate_opportunity(
        self,
        ticker: str,
        predicted_resolution: str,
        market: Market,
        confidence: float,
        signal: AnySignal,
    ) -> SnipeOpportunity:
        """Build a SnipeOpportunity from market data and signal."""
        # Use the best ask for the side we want to buy.
        if predicted_resolution == "yes":
            entry_price = market.yes_ask
        else:
            # For buying NO, the cost is (1 - yes_bid).
            entry_price = _ONE - market.yes_bid if market.yes_bid > _ZERO else market.yes_ask

        profit = self.calculate_profit(entry_price, is_taker=True)

        signal_data: dict[str, Any] = {}
        if isinstance(signal, OutcomeSignal):
            signal_data = {
                "game_id": signal.game_id,
                "outcome": signal.outcome,
                "home_team": signal.home_team,
                "away_team": signal.away_team,
                "score": f"{signal.home_score}-{signal.away_score}",
            }
            source = "espn"
        elif isinstance(signal, ThresholdSignal):
            signal_data = {
                "variable": signal.variable,
                "value": signal.current_value,
                "threshold": signal.threshold,
                "station": signal.station_id,
            }
            source = "weather"
        elif isinstance(signal, EconomicSignal):
            signal_data = {
                "series_id": signal.series_id,
                "value": signal.value,
                "threshold": signal.threshold,
            }
            source = "fred"
        else:
            source = "unknown"

        return SnipeOpportunity(
            ticker=ticker,
            outcome_known=confidence >= self._config.min_confidence,
            predicted_resolution=predicted_resolution,
            confidence=confidence,
            current_market_price=entry_price,
            expected_payout=_ONE,
            expected_profit=profit,
            signal_source=source,
            signal_data=signal_data,
            detected_at=now_utc(),
        )

    @staticmethod
    def _event_ticker_from_ticker(ticker: str) -> str:
        """Extract the event ticker (game identifier) from a contract ticker.

        KXNHLTOTAL-26APR13DETTB-5  -> KXNHLTOTAL-26APR13DETTB
        KXMLBGAME-26APR111310MIADET-MIA -> KXMLBGAME-26APR111310MIADET
        """
        parts = ticker.rsplit("-", 1)
        return parts[0] if len(parts) > 1 else ticker

    def _should_execute(self, opp: SnipeOpportunity) -> bool:
        """Determine whether an opportunity should be executed."""
        if not opp.outcome_known:
            return False

        price_f = float(opp.current_market_price)
        if price_f > self._config.max_entry_price:
            logger.debug(
                "sniper.price_too_high",
                ticker=opp.ticker,
                price=price_f,
                max=self._config.max_entry_price,
            )
            return False

        if price_f < self._config.min_entry_price:
            logger.debug(
                "sniper.price_too_low",
                ticker=opp.ticker,
                price=price_f,
                min=self._config.min_entry_price,
            )
            return False

        profit_f = float(opp.expected_profit)
        if profit_f < self._config.min_profit_after_fees:
            logger.debug(
                "sniper.profit_too_low",
                ticker=opp.ticker,
                profit=profit_f,
                min=self._config.min_profit_after_fees,
            )
            return False

        # Per-event exposure cap: don't stack too much $ on one game
        event_ticker = self._event_ticker_from_ticker(opp.ticker)
        current_exposure = self._event_exposure.get(event_ticker, 0.0)
        new_cost = price_f * self._config.max_contracts_per_snipe
        if current_exposure + new_cost > self._config.max_exposure_per_event:
            logger.warning(
                "sniper.event_exposure_limit",
                ticker=opp.ticker,
                event_ticker=event_ticker,
                current_exposure=round(current_exposure, 2),
                new_cost=round(new_cost, 2),
                max_per_event=self._config.max_exposure_per_event,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signal_confidence(signal: AnySignal) -> float:
        """Extract confidence value from any signal type."""
        return signal.confidence

    def _estimate_safe_size(self, opp: SnipeOpportunity) -> int:
        """Estimate a safe position size for a snipe.

        Conservative sizing: uses the configured max or a fraction of
        available capital, whichever is smaller.
        """
        # For now, use the configured max.  In production, this would
        # query account balance and compute a risk-limited size.
        return self._config.max_contracts_per_snipe

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    async def _auto_discover_mappings(self) -> None:
        """Attempt to discover contract mappings from Kalshi market tickers.

        Fetches open markets and matches tickers against known patterns
        to create ContractMapping objects automatically.
        """
        try:
            markets = await self._client.get_all_markets(status="open", limit=100)
        except Exception:
            logger.warning(
                "sniper.auto_discover_failed",
                exc_info=True,
            )
            return

        discovered = 0
        for market in markets:
            if market.ticker in self._mappings:
                continue

            mapping = self._infer_mapping_from_ticker(market)
            if mapping is not None:
                self._mappings[mapping.ticker] = mapping
                discovered += 1

        if discovered:
            logger.info(
                "sniper.auto_discovered",
                new_mappings=discovered,
                total_mappings=len(self._mappings),
            )

    def _infer_mapping_from_ticker(self, market: Market) -> ContractMapping | None:
        """Try to infer a ContractMapping from a market's ticker and title.

        Parses real Kalshi ticker formats such as:
        - KXTEMPNYCH-26APR0909-T47.99  (weather temperature)
        - KXHIGHNY-26APR09-T80         (weather high temp)
        - KXMLBHRR-26APR091210CINMIA-PLAYER-5                (MLB player prop)
        - KXMVESPORTSMULTIGAMEEXTENDED-S{hash}-{hash}         (MVE sports)
        - KXA100MON-26APR0912-T90.99                          (general threshold)
        """
        ticker = market.ticker.upper()

        for pattern, meta in _TICKER_PATTERNS.items():
            match = re.match(pattern, ticker)
            if match is None:
                continue

            category = meta["category"]
            data_source = meta["data_source"]

            if category == "sports":
                groups = match.groups()
                prefix = groups[0] if groups else ""

                # MLB player props: KXMLBHRR-26APR091210CINMIA-PLAYER-5
                if prefix.startswith("MLB") and len(groups) >= 3:
                    date_teams = groups[1]  # "26APR091210CINMIA"
                    player_name = groups[2]  # "PLAYER"
                    # Parse the MLB stat type from prefix
                    stat_type = prefix[3:]  # "HRR", "TB", etc.
                    # Extract team codes from the date+teams string
                    # Format: YYMONDDHHMM{TEAM1}{TEAM2}
                    team_match = re.search(r"\d{2}[A-Z]{3}\d{4,6}([A-Z]{2,3})([A-Z]{2,3})$", date_teams)
                    teams = []
                    if team_match:
                        teams = [team_match.group(1), team_match.group(2)]

                    return ContractMapping(
                        ticker=market.ticker,
                        category="sports",
                        data_source="espn",
                        source_params={
                            "sport": "baseball",
                            "league": "mlb",
                            "player": player_name,
                            "stat_type": stat_type,
                            "teams": teams,
                        },
                        resolution_field=stat_type.lower(),
                    )

                # MVE sports / cross-category
                if prefix.startswith("MVE"):
                    return ContractMapping(
                        ticker=market.ticker,
                        category="sports",
                        data_source="espn",
                        source_params={
                            "sport": "multi",
                            "league": "multi",
                            "mve_type": prefix,
                        },
                        resolution_field="multivariate",
                    )

                # Legacy sports: KXNBA-*, KXNFL-*, etc.
                league_code = prefix
                espn_info = _LEAGUE_TO_ESPN.get(league_code)
                if espn_info is None:
                    continue

                sport, league = espn_info
                team = self._extract_team_from_title(market.title)

                return ContractMapping(
                    ticker=market.ticker,
                    category="sports",
                    data_source="espn",
                    source_params={
                        "sport": sport,
                        "league": league,
                        "team": team,
                    },
                    resolution_field="winner",
                )

            elif category == "weather":
                groups = match.groups()
                # Try to extract threshold from the ticker T-suffix first
                # (e.g. KXTEMPNYCH-26APR0909-T47.99 -> threshold=47.99)
                threshold: float | None = None
                direction: str | None = "above"
                date_str: str | None = None

                if len(groups) >= 3:
                    # Pattern matched with date and threshold groups
                    date_str = groups[1]   # "26APR0909"
                    try:
                        threshold = float(groups[2])  # "47.99"
                    except (ValueError, TypeError):
                        pass
                elif len(groups) >= 2:
                    # Maybe date + threshold or just date
                    try:
                        threshold = float(groups[1])
                    except (ValueError, TypeError):
                        date_str = groups[1]

                # If no threshold from ticker, fall back to title parsing
                if threshold is None:
                    threshold, direction = self._extract_weather_threshold(
                        market.title
                    )

                # Extract location hint from ticker prefix
                # KXTEMPNYCH -> NYC, KXHIGHNY -> NY
                prefix = groups[0] if groups else ""
                source_params: dict[str, Any] = {}
                if date_str:
                    source_params["date_code"] = date_str
                if "NYC" in prefix or "NY" in prefix:
                    source_params["location"] = "New York"
                elif "CHI" in prefix:
                    source_params["location"] = "Chicago"
                elif "LA" in prefix:
                    source_params["location"] = "Los Angeles"

                return ContractMapping(
                    ticker=market.ticker,
                    category="weather",
                    data_source="noaa",
                    source_params=source_params,
                    resolution_field="temperature",
                    threshold=threshold,
                    direction=direction,
                )

            elif category == "economic":
                series_id = self._extract_series_from_ticker(ticker)
                return ContractMapping(
                    ticker=market.ticker,
                    category="economic",
                    data_source="fred",
                    source_params={"series_id": series_id},
                    resolution_field=series_id.lower(),
                )

        return None

    # ------------------------------------------------------------------
    # Title parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_team_from_title(title: str) -> str:
        """Best-effort extraction of team name from a market title.

        Kalshi titles often look like "Will the Lakers win tonight?" or
        "NBA: Lakers vs Celtics - Lakers to win".
        """
        # Try patterns like "Will the <TEAM> win"
        m = re.search(r"[Ww]ill\s+(?:the\s+)?(.+?)\s+win", title)
        if m:
            return m.group(1).strip()

        # Try "<TEAM> to win"
        m = re.search(r"(.+?)\s+to\s+win", title)
        if m:
            return m.group(1).strip()

        # Try "<TEAM> vs <TEAM>" and take the first
        m = re.search(r"(.+?)\s+vs\.?\s+", title)
        if m:
            return m.group(1).strip()

        return ""

    @staticmethod
    def _extract_weather_threshold(title: str) -> tuple[float | None, str | None]:
        """Extract a numeric threshold and direction from a weather market title.

        Examples:
            "Will Chicago reach 90F today?" -> (90.0, "above")
            "Temperature above 85 in NYC?" -> (85.0, "above")
            "Will it be below 32F in Denver?" -> (32.0, "below")
        """
        # Look for "above/exceed/reach/over X" patterns.
        m = re.search(
            r"(?:above|exceed|reach|over|hit)\s+(\d+\.?\d*)", title, re.IGNORECASE
        )
        if m:
            return float(m.group(1)), "above"

        # Look for "below/under/drop to X" patterns.
        m = re.search(
            r"(?:below|under|drop\s+to|fall\s+to)\s+(\d+\.?\d*)", title, re.IGNORECASE
        )
        if m:
            return float(m.group(1)), "below"

        return None, None

    @staticmethod
    def _extract_series_from_ticker(ticker: str) -> str:
        """Map a Kalshi economic ticker prefix to a FRED series ID."""
        prefix_map = {
            "CPI": "CPIAUCSL",
            "UNRATE": "UNRATE",
            "GDP": "GDP",
            "FED": "FEDFUNDS",
            "JOBS": "PAYEMS",
            "ECON": "",
        }
        for prefix, series in prefix_map.items():
            if prefix in ticker:
                return series
        return ""

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self) -> None:
        """Reset the daily snipe counter if the date has changed."""
        today = now_utc().strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_snipe_count = 0

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @property
    def snipe_history(self) -> list[SnipeRecord]:
        """All snipe records for analysis."""
        return list(self._snipe_history)

    @property
    def opportunity_log(self) -> list[SnipeOpportunity]:
        """All opportunities found (traded and not traded)."""
        return list(self._opportunity_log)

    def hit_rate(self) -> float:
        """Fraction of snipes that settled correctly.

        Returns
        -------
        float
            Hit rate (0-1), or 0.0 if no settled snipes yet.
        """
        settled = [r for r in self._snipe_history if r.settled_correctly is not None]
        if not settled:
            return 0.0
        correct = sum(1 for r in settled if r.settled_correctly)
        return correct / len(settled)

    def total_profit(self) -> Decimal:
        """Sum of actual profits from all settled snipes."""
        return sum(
            (r.actual_profit for r in self._snipe_history if r.actual_profit is not None),
            _ZERO,
        )

    def expected_vs_actual(self) -> dict[str, Decimal]:
        """Compare expected vs actual profit across settled snipes."""
        settled = [r for r in self._snipe_history if r.actual_profit is not None]
        if not settled:
            return {"expected": _ZERO, "actual": _ZERO, "difference": _ZERO}

        expected = sum((r.expected_profit for r in settled), _ZERO)
        actual = sum((r.actual_profit for r in settled), _ZERO)

        return {
            "expected": expected,
            "actual": actual,
            "difference": actual - expected,
        }

    def mark_settlement(
        self, ticker: str, resolved_yes: bool
    ) -> None:
        """Mark a snipe's settlement outcome for tracking.

        Parameters
        ----------
        ticker:
            The contract ticker that settled.
        resolved_yes:
            Whether the contract resolved YES.
        """
        for record in self._snipe_history:
            if record.ticker != ticker:
                continue
            if record.settled_correctly is not None:
                continue  # already marked

            predicted_yes = record.side == "yes"
            record.settled_correctly = predicted_yes == resolved_yes

            if record.settled_correctly:
                # Profit = payout ($1) - entry_price - fee.
                record.actual_profit = self.calculate_profit(
                    record.entry_price, is_taker=True
                ) * Decimal(record.contracts)
            else:
                # Loss = entry_price * contracts (lost our stake).
                record.actual_profit = -(record.entry_price * Decimal(record.contracts))

            logger.info(
                "sniper.settlement_marked",
                ticker=ticker,
                correct=record.settled_correctly,
                actual_profit=str(record.actual_profit),
            )
