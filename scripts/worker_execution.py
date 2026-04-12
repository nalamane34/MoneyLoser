#!/usr/bin/env python3
"""Worker: Execution engine (the brain).

Reads sportsbook lines from ``collector.duckdb`` and market data from
``market_data.duckdb``.  Runs feature pipeline, model, edge calculation,
sizing, risk checks, and submits orders via Kalshi REST + WS.

Writes features, predictions, and fills to ``execution.duckdb``.

Usage::

    python scripts/worker_execution.py --config config/default.yaml --overlay config/paper-soak.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import default_weather_locations, load_config
from moneygone.data.schemas import EXECUTION_TABLES
from moneygone.data.sports.live_snapshots import (
    StoreBackedSportsSnapshotProvider,
    recommended_max_line_age,
)
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.ws_client import KalshiWebSocket
from moneygone.data.market_discovery import MarketCategory
from moneygone.execution.category_providers import CryptoDataProvider, WeatherDataProvider
from moneygone.execution.artifact_runtime import (
    build_universal_artifact_fallbacks,
    fallback_categories_for_config,
)
from moneygone.execution.engine import (
    CategoryProvider,
    ExecutionEngine,
)
from moneygone.execution.fill_tracker import FillTracker
from moneygone.execution.order_manager import OrderManager
from moneygone.execution.strategies import AggressiveStrategy, DryRunStrategy, PassiveStrategy
from moneygone.features import (
    BidAskSpread,
    DepthRatio,
    HomeFieldAdvantage,
    KalshiVsSportsbookEdge,
    MidPrice,
    MoneylineMovement,
    OrderbookImbalance,
    PinnacleVsMarketEdge,
    PinnacleWinProbability,
    PowerRatingEdge,
    SpreadImpliedWinProb,
    SportsbookWinProbability,
    TeamInjuryImpact,
    TimeToExpiry,
    WeightedMidPrice,
)
from moneygone.features.crypto_features import (
    ATR14,
    ATR24,
    BRTIDistanceToThreshold,
    BRTIPrice,
    BasisSpread,
    CryptoOrderbookImbalance,
    FundingRateSignal,
    FundingRateZScore,
    ImpliedVolatility,
    OpenInterestChange,
    RealizedVol7d,
    RealizedVol24h,
    RealizedVol30d,
    TrendRegime,
    TrendStrength,
    VolSpread,
    VolatilityRegime,
    WhaleFlowIndicator,
)
from moneygone.features.pipeline import FeaturePipeline
from moneygone.features.weather_features import (
    ClimatologicalAnomaly,
    EnsembleExceedanceProb,
    EnsembleMean,
    EnsembleSpread,
    ForecastHorizon,
    ForecastRevisionDirection,
    ForecastRevisionMagnitude,
    StationBiasExceedance,
    ModelDisagreement,
)
from moneygone.models.crypto_vol import CryptoVolModel
from moneygone.models.market_baseline import MarketBaselineModel
from moneygone.models.sharp_sportsbook import SharpSportsbookModel
from moneygone.models.weather_ensemble import WeatherEnsembleModel
from moneygone.risk.drawdown import DrawdownMonitor
from moneygone.risk.exposure import ExposureCalculator
from moneygone.risk.manager import RiskManager
from moneygone.risk.portfolio import PortfolioTracker
from moneygone.signals.edge import EdgeCalculator
from moneygone.signals.fees import KalshiFeeCalculator
from moneygone.sizing.kelly import KellySizer
from moneygone.sizing.risk_limits import RiskLimits
from moneygone.strategies.closer_killswitch import (
    CloserKillSwitch,
    KillSwitchConfig,
    tiered_min_confidence,
)
from moneygone.strategies.live_event_edge import LiveEdgeConfig, LiveEventEdge
from moneygone.strategies.resolution_sniper import ResolutionSniper, SnipeConfig
from moneygone.utils.env import load_repo_env
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.execution")
REPO_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    parser = argparse.ArgumentParser(description="Execution engine worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    args = parser.parse_args()

    loaded_env = load_repo_env(REPO_ROOT)
    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)
    if loaded_env:
        log.info("worker_execution.repo_env_loaded", keys=sorted(loaded_env))

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config.model.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir

    # Own database for writes (features, predictions, fills)
    store = DataStore(data_dir / "execution.duckdb")
    store.initialize_schema(EXECUTION_TABLES)

    # Try to attach collector DB for weather/crypto views.
    # This will fail if the collector process holds the lock — that's OK,
    # sports execution doesn't need those tables.
    collector_db = data_dir / "collector.duckdb"
    collector_attached = False
    try:
        store.attach_readonly("collector", collector_db)
        store.create_attached_views(
            {
                "collector": [
                    "forecast_ensembles",
                    "funding_rates",
                    "open_interest",
                ],
            }
        )
        collector_attached = True
        log.info("execution.collector_attached")
    except Exception:
        log.warning(
            "execution.collector_attach_failed",
            msg="collector DB locked — weather/crypto views unavailable, sports-only mode",
        )

    # Cross-DB reads: DuckDB doesn't support cross-process concurrent access.
    # The collector exports sportsbook data to a parquet file; we load it
    # into our own execution.duckdb on startup and periodically.
    sportsbook_parquet = data_dir / "sportsbook_lines.parquet"
    if sportsbook_parquet.exists():
        try:
            count = store.load_parquet_into_table("sportsbook_game_lines", sportsbook_parquet)
            log.info("execution.sportsbook_loaded_from_parquet", rows=count)
        except Exception:
            log.warning("execution.sportsbook_parquet_load_failed", exc_info=True)
    else:
        log.info("execution.no_sportsbook_parquet", msg="waiting for collector to export")

    # Build pipeline components
    model = SharpSportsbookModel()
    pipeline = FeaturePipeline(
        [
            SportsbookWinProbability(),
            PinnacleWinProbability(),
            KalshiVsSportsbookEdge(),
            PinnacleVsMarketEdge(),
            MoneylineMovement(),
            PowerRatingEdge(),
            HomeFieldAdvantage(),
            TeamInjuryImpact(),
            SpreadImpliedWinProb(),
        ],
        store=store,
    )

    fee_calculator = KalshiFeeCalculator()
    edge_calculator = EdgeCalculator(
        fee_calculator=fee_calculator,
        min_edge_threshold=config.execution.min_edge_threshold,
        max_edge_sanity=config.execution.max_edge_sanity,
    )
    kelly_sizer = KellySizer(
        kelly_fraction=config.risk.kelly_fraction,
        max_position_pct=config.risk.max_total_exposure_pct,
    )

    rest_client = KalshiRestClient(config.exchange)
    ws_client = KalshiWebSocket(config.exchange)
    order_manager = OrderManager(rest_client)
    fill_tracker = FillTracker(store=store)
    portfolio_tracker = PortfolioTracker()
    drawdown_monitor = DrawdownMonitor()
    exposure_calculator = ExposureCalculator()
    risk_limits = RiskLimits(config.risk)
    risk_manager = RiskManager(
        risk_config=config.risk,
        risk_limits=risk_limits,
        portfolio=portfolio_tracker,
        drawdown_monitor=drawdown_monitor,
        exposure_calculator=exposure_calculator,
    )

    # Sportsbook data is loaded into execution.duckdb from parquet (see above).
    sports_provider = StoreBackedSportsSnapshotProvider(
        store,
        leagues=config.sportsbook.leagues,
        rest_client=rest_client,
        max_line_age=recommended_max_line_age(
            config.sportsbook.fetch_interval_minutes,
        ),
    )

    inner_strategy = (
        PassiveStrategy(timeout_seconds=30.0)
        if config.execution.prefer_maker
        else AggressiveStrategy()
    )
    strategy = (
        DryRunStrategy(inner_strategy) if config.exchange.demo_mode
        else inner_strategy
    )

    # ---- Category providers: crypto, weather ----
    category_providers: dict[MarketCategory, CategoryProvider] = {}
    crypto_provider: CryptoDataProvider | None = None
    weather_provider: WeatherDataProvider | None = None

    if config.crypto.enabled:
        try:
            from moneygone.data.crypto.ccxt_feed import CryptoDataFeed
            from moneygone.data.crypto.volatility import CryptoVolatilityFeed

            crypto_feed = CryptoDataFeed(
                exchange_ids=config.crypto.exchanges,
            )
            vol_feed = CryptoVolatilityFeed(
                exchange_id=config.crypto.exchanges[0] if config.crypto.exchanges else "binanceus",
            )

            # Coinalyze for futures data (funding rates, OI) — works from US
            futures_feed = None
            import os
            coinalyze_key = config.crypto.coinalyze_api_key or os.environ.get("COINALYZE_API_KEY", "")
            if coinalyze_key:
                from moneygone.data.crypto.coinalyze_feed import CoinalyzeFeed
                futures_feed = CoinalyzeFeed(api_key=coinalyze_key)
                log.info("execution.coinalyze_feed_enabled")

            crypto_provider = CryptoDataProvider(crypto_feed, vol_feed, futures_feed=futures_feed)
            crypto_model = CryptoVolModel()
            crypto_pipeline = FeaturePipeline(
                [
                    FundingRateSignal(),
                    FundingRateZScore(),
                    OpenInterestChange(),
                    CryptoOrderbookImbalance(),
                    WhaleFlowIndicator(),
                    VolatilityRegime(),
                    BasisSpread(),
                    ATR14(),
                    ATR24(),
                    RealizedVol24h(),
                    RealizedVol7d(),
                    RealizedVol30d(),
                    ImpliedVolatility(),
                    VolSpread(),
                    TrendRegime(),
                    TrendStrength(),
                    BRTIPrice(),
                    BRTIDistanceToThreshold(),
                ],
                store=store,
            )
            category_providers[MarketCategory.CRYPTO] = CategoryProvider(
                category=MarketCategory.CRYPTO,
                model=crypto_model,
                pipeline=crypto_pipeline,
                get_context_data=crypto_provider.get_context,
            )
            log.info("execution.crypto_provider_enabled", symbols=config.crypto.symbols)
        except ImportError:
            log.warning("execution.crypto_provider_unavailable", msg="ccxt not installed")
        except Exception:
            log.warning("execution.crypto_provider_failed", exc_info=True)

    if config.weather.enabled:
        try:
            from moneygone.data.weather.ecmwf import ECMWFOpenDataFetcher
            from moneygone.data.weather.noaa import NOAAEnsembleFetcher
            from moneygone.data.weather.nws import NWSFetcher

            noaa_fetcher = NOAAEnsembleFetcher(
                api_key=config.weather.open_meteo_api_key,
            )
            ecmwf_fetcher = ECMWFOpenDataFetcher()
            nws_fetcher = NWSFetcher()
            weather_locations = [
                {"name": loc["name"], "lat": loc["lat"], "lon": loc["lon"]}
                for loc in (config.weather.locations or default_weather_locations())
            ]
            weather_provider = WeatherDataProvider(
                noaa_fetcher,
                ecmwf_fetcher,
                weather_locations,
                nws_fetcher=nws_fetcher,
            )
            weather_model = WeatherEnsembleModel()
            weather_pipeline = FeaturePipeline(
                [
                    EnsembleMean(),
                    EnsembleSpread(),
                    EnsembleExceedanceProb(),
                    StationBiasExceedance(),
                    ForecastRevisionMagnitude(),
                    ForecastRevisionDirection(),
                    ModelDisagreement(),
                    ForecastHorizon(),
                    ClimatologicalAnomaly(),
                ],
                store=store,
            )
            category_providers[MarketCategory.WEATHER] = CategoryProvider(
                category=MarketCategory.WEATHER,
                model=weather_model,
                pipeline=weather_pipeline,
                get_context_data=weather_provider.get_context,
            )
            log.info("execution.weather_provider_enabled", locations=[l["name"] for l in weather_locations])
        except ImportError:
            log.warning("execution.weather_provider_unavailable")
        except Exception:
            log.warning("execution.weather_provider_failed", exc_info=True)

    fallback_categories = fallback_categories_for_config(
        weather_enabled=config.weather.enabled,
        crypto_enabled=config.crypto.enabled,
    )
    missing_categories = [
        category
        for category in fallback_categories
        if category not in category_providers
    ]
    artifact_fallbacks = build_universal_artifact_fallbacks(
        model_dir,
        missing_categories,
    )
    if artifact_fallbacks:
        for category, (artifact_model, artifact_pipeline) in artifact_fallbacks.items():
            category_providers[category] = CategoryProvider(
                category=category,
                model=artifact_model,
                pipeline=artifact_pipeline,  # type: ignore[arg-type]
                get_context_data=None,
            )
        log.info(
            "execution.artifact_fallback_providers_enabled",
            categories=[category.value for category in artifact_fallbacks],
            model=next(iter(artifact_fallbacks.values()))[0].name,
        )

    remaining_categories = [
        category
        for category in fallback_categories
        if category not in category_providers
    ]

    # ---- Baseline model: ONLY in demo mode and only if artifacts are unavailable ----
    # The MarketBaselineModel has NO informational edge — it just mirrors
    # market pricing.  It must NEVER trade with real money.
    if remaining_categories and config.exchange.demo_mode:
        baseline_model = MarketBaselineModel()
        baseline_pipeline = FeaturePipeline(
            [
                BidAskSpread(),
                MidPrice(),
                OrderbookImbalance(),
                WeightedMidPrice(),
                DepthRatio(),
                TimeToExpiry(),
            ],
            store=store,
        )
        for category in remaining_categories:
            category_providers[category] = CategoryProvider(
                category=category,
                model=baseline_model,
                pipeline=baseline_pipeline,
                get_context_data=None,
            )
        log.info(
            "execution.market_baseline_provider_enabled",
            categories=[category.value for category in remaining_categories],
            model=baseline_model.name,
            version=baseline_model.version,
        )
    elif remaining_categories:
        log.warning(
            "execution.artifact_fallbacks_missing",
            categories=[category.value for category in remaining_categories],
            msg="No artifact-backed fallback available for some live categories",
        )

    log.info(
        "execution.category_providers",
        enabled=[c.value for c in category_providers.keys()],
    )

    # Shared discovery cache written by market_data worker
    discovery_cache_path = data_dir / "discovered_markets.json"

    engine = ExecutionEngine(
        rest_client=rest_client,
        ws_client=ws_client,
        feature_pipeline=pipeline,
        model=model,
        edge_calculator=edge_calculator,
        sizer=kelly_sizer,
        risk_manager=risk_manager,
        order_manager=order_manager,
        fill_tracker=fill_tracker,
        strategy=strategy,
        config=config.execution,
        watched_tickers=[],
        store=store,
        sports_snapshot_provider=sports_provider,
        recorder=None,  # Market data worker handles recording
        category_providers=category_providers,
        discovery_cache_path=discovery_cache_path,
        sportsbook_parquet_path=sportsbook_parquet,
        demo_mode=config.exchange.demo_mode,
    )

    # ---- Closer strategies: Resolution Sniper + Live Event Edge ----
    resolution_sniper: ResolutionSniper | None = None
    live_event_edge: LiveEventEdge | None = None
    kill_switch: CloserKillSwitch | None = None

    closer_cfg = getattr(config, "closer", None)
    # Parse closer config from raw overlay if not in typed config
    if closer_cfg is None:
        import yaml as _yaml
        with open(args.overlay) as _f:
            _raw = _yaml.safe_load(_f) or {}
        closer_cfg = _raw.get("closer", {})
    else:
        closer_cfg = {}

    closer_enabled = closer_cfg.get("enabled", False) if isinstance(closer_cfg, dict) else False

    if closer_enabled:
        # Kill switch (shared between both strategies)
        ks_cfg = closer_cfg.get("kill_switch", {})
        kill_switch = CloserKillSwitch(
            KillSwitchConfig(
                max_consecutive_losses=ks_cfg.get("max_consecutive_losses", 4),
                cooldown_hours=ks_cfg.get("cooldown_hours", 12.0),
            )
        )

        # Monkey-patch kill switch check into strategies.
        # The kill switch ONLY affects closer strategies — the main
        # ExecutionEngine continues trading normally even when the
        # kill switch is triggered.

        # Resolution Sniper
        sniper_cfg = closer_cfg.get("sniper", {})
        resolution_sniper = ResolutionSniper(
            rest_client=rest_client,
            order_manager=order_manager,
            fee_calculator=fee_calculator,
            fill_tracker=fill_tracker,
            portfolio=portfolio_tracker,
            config=SnipeConfig(
                min_confidence=sniper_cfg.get("min_confidence", 0.95),
                max_entry_price=sniper_cfg.get("max_entry_price", 0.95),
                min_entry_price=sniper_cfg.get("min_entry_price", 0.75),
                min_profit_after_fees=sniper_cfg.get("min_profit_after_fees", 0.005),
                max_contracts_per_snipe=sniper_cfg.get("max_contracts_per_snipe", 20),
                max_daily_snipes=sniper_cfg.get("max_daily_snipes", 40),
                cooldown_seconds=sniper_cfg.get("cooldown_seconds", 3.0),
            ),
        )

        # Wrap sniper's _should_execute to check kill switch first
        _original_should_execute = resolution_sniper._should_execute

        def _guarded_should_execute(opp):  # type: ignore
            if not kill_switch.is_active:
                log.warning(
                    "sniper.kill_switch_active",
                    ticker=opp.ticker,
                    paused_until=str(kill_switch.paused_until),
                    msg="Closer strategies paused — main engine unaffected",
                )
                return False
            return _original_should_execute(opp)

        resolution_sniper._should_execute = _guarded_should_execute  # type: ignore
        log.info(
            "execution.sniper_enabled",
            min_confidence=sniper_cfg.get("min_confidence", 0.95),
            max_entry_price=sniper_cfg.get("max_entry_price", 0.95),
            min_entry_price=sniper_cfg.get("min_entry_price", 0.75),
            max_contracts=sniper_cfg.get("max_contracts_per_snipe", 20),
        )

        # Live Event Edge
        edge_cfg = closer_cfg.get("live_edge", {})
        from moneygone.data.sports.espn import ESPNLiveFeed
        from moneygone.data.sports.live_weather import LiveWeatherFeed

        espn_feed = ESPNLiveFeed()
        weather_feed = LiveWeatherFeed()

        live_event_edge = LiveEventEdge(
            rest_client=rest_client,
            espn_feed=espn_feed,
            weather_feed=weather_feed,
            fee_calculator=fee_calculator,
            edge_calculator=edge_calculator,
            sizer=kelly_sizer,
            risk_manager=risk_manager,
            order_manager=order_manager,
            fill_tracker=fill_tracker,
            config=LiveEdgeConfig(
                min_edge=edge_cfg.get("min_edge", 0.05),
                min_confidence=edge_cfg.get("min_confidence_low", 0.90),
                scan_interval_seconds=edge_cfg.get("scan_interval_seconds", 15.0),
                max_contracts_per_trade=edge_cfg.get("max_contracts_per_trade", 20),
                sports_enabled=edge_cfg.get("sports_enabled", True),
                weather_enabled=edge_cfg.get("weather_enabled", True),
                crypto_enabled=edge_cfg.get("crypto_enabled", False),
                sports_leagues=edge_cfg.get("sports_leagues", ["nba", "nhl", "mlb", "ufc"]),
            ),
        )
        log.info(
            "execution.live_edge_enabled",
            min_edge=edge_cfg.get("min_edge", 0.05),
            scan_interval=edge_cfg.get("scan_interval_seconds", 15.0),
            sports_leagues=edge_cfg.get("sports_leagues", ["nba", "nhl", "mlb", "ufc"]),
            sports=edge_cfg.get("sports_enabled", True),
            weather=edge_cfg.get("weather_enabled", True),
        )

        # Wrap live_edge's evaluate_signal to check kill switch + tiered confidence
        _original_evaluate = live_event_edge.evaluate_signal

        async def _guarded_evaluate(signal):  # type: ignore
            if not kill_switch.is_active:
                log.warning(
                    "live_edge.kill_switch_active",
                    ticker=signal.ticker,
                    paused_until=str(kill_switch.paused_until),
                    msg="Closer strategies paused — main engine unaffected",
                )
                return None
            # Apply tiered confidence based on market price
            required_confidence = tiered_min_confidence(signal.market_probability)
            if signal.confidence < required_confidence:
                log.debug(
                    "live_edge.tiered_confidence_reject",
                    ticker=signal.ticker,
                    confidence=round(signal.confidence, 3),
                    required=required_confidence,
                    market_price=round(signal.market_probability, 3),
                )
                return None
            return await _original_evaluate(signal)

        live_event_edge.evaluate_signal = _guarded_evaluate  # type: ignore

        log.info(
            "execution.kill_switch_configured",
            max_consecutive_losses=ks_cfg.get("max_consecutive_losses", 4),
            cooldown_hours=ks_cfg.get("cooldown_hours", 12.0),
            note="Kill switch ONLY affects closer strategies — main engine unaffected",
        )
    else:
        log.info("execution.closer_strategies_disabled")

    # ---- Health status writer ----
    _engine_start_time = time.monotonic()
    HEALTH_PATH = Path("/tmp/moneygone_health.json")

    async def _health_writer_loop() -> None:
        """Write health status JSON every 60 seconds for dashboard polling."""
        while not shutdown.is_set():
            try:
                last_eval_times = engine._last_eval
                latest_eval = (
                    max(last_eval_times.values()).isoformat()
                    if last_eval_times
                    else None
                )
                health = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "engine_status": "running",
                    "uptime_seconds": round(time.monotonic() - _engine_start_time, 1),
                    "markets_watched": len(engine._watched),
                    "open_orders": len(order_manager._open_orders)
                    if hasattr(order_manager, "_open_orders")
                    else 0,
                    "total_trades": len(fill_tracker._fills),
                    "last_evaluation_time": latest_eval,
                    "closer_status": {
                        "resolution_sniper": resolution_sniper is not None,
                        "live_event_edge": live_event_edge is not None,
                        "kill_switch_active": kill_switch.is_active
                        if kill_switch is not None
                        else None,
                    },
                }
                tmp = HEALTH_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(health, indent=2))
                tmp.rename(HEALTH_PATH)
            except asyncio.CancelledError:
                return
            except Exception:
                log.debug("execution.health_write_error", exc_info=True)
            await asyncio.sleep(60)

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("execution.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    log.info(
        "execution.starting",
        model=model.name,
        leagues=config.sportsbook.leagues,
        demo_mode=config.exchange.demo_mode,
        closer_enabled=closer_enabled,
    )

    # Settlement checker for kill switch — polls settlements and marks wins/losses
    async def _settlement_checker_loop() -> None:
        """Periodically check settlements and update kill switch."""
        if kill_switch is None:
            return
        seen_settlements: set[str] = set()
        # Collect tickers from both closer strategies
        while not shutdown.is_set():
            try:
                settlements = await rest_client.get_settlements(limit=50)
                for s in settlements:
                    if s.ticker in seen_settlements:
                        continue
                    seen_settlements.add(s.ticker)

                    # Check if this settlement is from a closer strategy trade
                    is_closer_trade = False
                    strategy_name = "unknown"

                    # Check sniper history
                    if resolution_sniper is not None:
                        for record in resolution_sniper._snipe_history:
                            if record.ticker == s.ticker:
                                is_closer_trade = True
                                strategy_name = "sniper"
                                break

                    # Check live edge (it records trades via kill_switch.record_trade)
                    if not is_closer_trade and kill_switch is not None:
                        for record in kill_switch._trade_log:
                            if record.ticker == s.ticker:
                                is_closer_trade = True
                                strategy_name = record.strategy
                                break

                    if not is_closer_trade:
                        continue

                    # Determine win/loss: revenue > 0 means profitable
                    won = float(s.revenue) > 0
                    await kill_switch.mark_resolution(s.ticker, won)
                    log.info(
                        "kill_switch.settlement_tracked",
                        ticker=s.ticker,
                        won=won,
                        revenue=float(s.revenue),
                        strategy=strategy_name,
                        consecutive_losses=kill_switch.consecutive_losses,
                    )
            except asyncio.CancelledError:
                return
            except Exception:
                log.warning("kill_switch.settlement_check_error", exc_info=True)

            await asyncio.sleep(60)  # Check every 60 seconds

    # ---- Periodic P&L reconciliation ----
    async def _reconciliation_loop() -> None:
        """Compare local P&L against exchange balance every 5 minutes."""
        while not shutdown.is_set():
            await asyncio.sleep(300)  # 5 minutes
            if shutdown.is_set():
                return
            try:
                result = await fill_tracker.reconcile_with_exchange(
                    rest_client, portfolio_tracker,
                )
                log.info("execution.reconciliation_complete", **result)
            except asyncio.CancelledError:
                return
            except Exception:
                log.warning("execution.reconciliation_failed", exc_info=True)

    settlement_task: asyncio.Task | None = None
    health_task: asyncio.Task | None = None
    reconciliation_task: asyncio.Task | None = None

    try:
        await engine.start()

        # Start closer strategies alongside the main engine
        if resolution_sniper is not None:
            await resolution_sniper.start()
            log.info("execution.sniper_started")
        if live_event_edge is not None:
            await live_event_edge.start()
            log.info("execution.live_edge_started")

        # Start settlement checker for kill switch
        if kill_switch is not None:
            settlement_task = asyncio.create_task(
                _settlement_checker_loop(),
                name="settlement_checker",
            )
            log.info("execution.settlement_checker_started")

        # Start health status writer
        health_task = asyncio.create_task(
            _health_writer_loop(),
            name="health_writer",
        )
        log.info("execution.health_writer_started", path=str(HEALTH_PATH))

        # Start P&L reconciliation loop
        reconciliation_task = asyncio.create_task(
            _reconciliation_loop(),
            name="pnl_reconciliation",
        )
        log.info("execution.reconciliation_started", interval_seconds=300)

        await shutdown.wait()
    finally:
        # Stop health writer
        if health_task is not None:
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
            # Write final stopped status
            try:
                stopped = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "engine_status": "stopped",
                    "uptime_seconds": round(time.monotonic() - _engine_start_time, 1),
                    "markets_watched": 0,
                    "open_orders": 0,
                    "total_trades": len(fill_tracker._fills),
                    "last_evaluation_time": None,
                    "closer_status": {
                        "resolution_sniper": False,
                        "live_event_edge": False,
                        "kill_switch_active": None,
                    },
                }
                HEALTH_PATH.write_text(json.dumps(stopped, indent=2))
            except Exception:
                pass

        # Stop reconciliation loop
        if reconciliation_task is not None:
            reconciliation_task.cancel()
            try:
                await reconciliation_task
            except asyncio.CancelledError:
                pass

        # Stop settlement checker
        if settlement_task is not None:
            settlement_task.cancel()
            try:
                await settlement_task
            except asyncio.CancelledError:
                pass
        # Stop closer strategies
        if live_event_edge is not None:
            await live_event_edge.stop()
            log.info("execution.live_edge_stopped")
        if resolution_sniper is not None:
            await resolution_sniper.stop()
            log.info("execution.sniper_stopped")
        if kill_switch is not None:
            stats = kill_switch.get_stats()
            log.info("execution.kill_switch_final_stats", **stats)

        await engine.stop()
        await sports_provider.close()
        if crypto_provider is not None:
            await crypto_provider.close()
        if weather_provider is not None:
            await weather_provider.close()
        await rest_client.close()
        store.close()
        log.info("execution.stopped")


if __name__ == "__main__":
    asyncio.run(main())
