"""Application wiring and lifecycle management.

The :func:`build_app` factory constructs all system components in the
correct dependency order and returns an :class:`Application` instance
that manages their startup and shutdown.

Component initialization order:
    1. Logging
    2. Authentication
    3. REST / WS clients
    4. Data store and recorder
    5. Feature pipeline
    6. Model loading
    7. Signal generation (fees, edge, filter)
    8. Monitoring (drift, calibration, PnL, alerts)
    9. Strategies (resolution sniper, live event edge, cross-market arb, market maker)
"""

from __future__ import annotations

import asyncio
import pickle
import signal
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from moneygone.config import AppConfig
from moneygone.data.crypto.ccxt_feed import CryptoDataFeed
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.market_discovery import MarketCategory
from moneygone.data.sports.espn import ESPNLiveFeed
from moneygone.data.sports.live_snapshots import (
    StoreBackedSportsSnapshotProvider,
    recommended_max_line_age,
)
from moneygone.data.sports.live_weather import LiveWeatherFeed
from moneygone.data.store import DataStore
from moneygone.data.weather import NOAAEnsembleFetcher, ECMWFOpenDataFetcher
from moneygone.data.weather.nws import NWSFetcher
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.ws_client import KalshiWebSocket
from moneygone.execution.category_providers import WeatherDataProvider
from moneygone.execution.engine import CategoryProvider, ExecutionEngine
from moneygone.execution.fill_tracker import FillTracker
from moneygone.execution.order_manager import OrderManager
from moneygone.execution.strategies import AggressiveStrategy, DryRunStrategy, PassiveStrategy
from moneygone.features import (
    BidAskSpread,
    ClimatologicalAnomaly,
    DepthRatio,
    EnsembleExceedanceProb,
    EnsembleMean,
    EnsembleSpread,
    ForecastHorizon,
    ForecastRevisionDirection,
    ForecastRevisionMagnitude,
    HomeFieldAdvantage,
    KalshiVsSportsbookEdge,
    MidPrice,
    ModelDisagreement,
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
from moneygone.features.pipeline import FeaturePipeline
from moneygone.monitoring.alerts import AlertManager
from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker
from moneygone.monitoring.regime_detector import RegimeDetector
from moneygone.models.market_baseline import MarketBaselineModel
from moneygone.models.sharp_sportsbook import SharpSportsbookModel
from moneygone.models.weather_ensemble import WeatherEnsembleModel
from moneygone.risk.manager import RiskManager
from moneygone.risk.drawdown import DrawdownMonitor
from moneygone.risk.exposure import ExposureCalculator
from moneygone.risk.portfolio import PortfolioTracker
from moneygone.signals.edge import EdgeCalculator
from moneygone.signals.fees import KalshiFeeCalculator
from moneygone.signals.filter import SignalFilter
from moneygone.sizing.kelly import KellySizer
from moneygone.sizing.risk_limits import RiskLimits
from moneygone.strategies.cross_market_arb import ArbConfig, CrossMarketArbitrage
from moneygone.strategies.live_event_edge import LiveEdgeConfig, LiveEventEdge
from moneygone.strategies.market_maker import MarketMaker, MMConfig
from moneygone.strategies.resolution_sniper import ResolutionSniper, SnipeConfig
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)


class Application:
    """Top-level application container that owns all components.

    Manages the startup and shutdown lifecycle so that all async
    resources are properly initialized and cleaned up.

    Parameters
    ----------
    config:
        The resolved application configuration.
    rest_client:
        Kalshi REST API client.
    store:
        DuckDB data store.
    recorder:
        Market data recorder.
    fee_calculator:
        Exchange fee calculator.
    edge_calculator:
        Edge computation engine.
    signal_filter:
        Pre-trade signal quality filter.
    drift_detector:
        Model drift detector.
    calibration_monitor:
        Rolling calibration tracker.
    regime_detector:
        Market regime classifier.
    pnl_tracker:
        Trade PnL tracker.
    alert_manager:
        Alert emission manager.
    model:
        Loaded model artifact (optional; ``None`` if no model found).
    pipeline:
        Feature pipeline (optional; ``None`` if not configured).
    resolution_sniper:
        Resolution sniping strategy.
    live_event_edge:
        Live event edge strategy.
    cross_market_arb:
        Cross-market arbitrage strategy.
    market_maker:
        Market making strategy.
    """

    def __init__(
        self,
        config: AppConfig,
        rest_client: KalshiRestClient,
        store: DataStore,
        recorder: MarketDataRecorder,
        fee_calculator: KalshiFeeCalculator,
        edge_calculator: EdgeCalculator,
        signal_filter: SignalFilter,
        drift_detector: DriftDetector,
        calibration_monitor: CalibrationMonitor,
        regime_detector: RegimeDetector,
        pnl_tracker: PnLTracker,
        alert_manager: AlertManager,
        model: Any | None = None,
        pipeline: Any | None = None,
        execution_engine: ExecutionEngine | None = None,
        resolution_sniper: ResolutionSniper | None = None,
        live_event_edge: LiveEventEdge | None = None,
        cross_market_arb: CrossMarketArbitrage | None = None,
        market_maker: MarketMaker | None = None,
        async_closeables: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.rest_client = rest_client
        self.store = store
        self.recorder = recorder
        self.fee_calculator = fee_calculator
        self.edge_calculator = edge_calculator
        self.signal_filter = signal_filter
        self.drift_detector = drift_detector
        self.calibration_monitor = calibration_monitor
        self.regime_detector = regime_detector
        self.pnl_tracker = pnl_tracker
        self.alert_manager = alert_manager
        self.model = model
        self.pipeline = pipeline
        self.execution_engine = execution_engine
        self.resolution_sniper = resolution_sniper
        self.live_event_edge = live_event_edge
        self.cross_market_arb = cross_market_arb
        self.market_maker = market_maker
        self._async_closeables = async_closeables or []

        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all async services.

        Initializes the REST client, starts the data recorder, and
        launches background monitoring tasks.
        """
        if self._running:
            log.warning("app.already_running")
            return

        log.info("app.starting")

        # Ensure REST client is ready
        await self.rest_client._ensure_client()  # noqa: SLF001

        # Start data recorder
        await self.recorder.start()

        # Launch background monitoring
        self._tasks.append(
            asyncio.create_task(
                self._monitoring_loop(), name="monitoring_loop"
            )
        )

        if self.execution_engine:
            await self.execution_engine.start()
            log.info("app.execution_engine_started")

        # Start strategies
        if self.resolution_sniper:
            await self.resolution_sniper.start()
            log.info("app.strategy_started", strategy="resolution_sniper")
        if self.live_event_edge:
            await self.live_event_edge.start()
            log.info("app.strategy_started", strategy="live_event_edge")
        if self.cross_market_arb:
            await self.cross_market_arb.start()
            log.info("app.strategy_started", strategy="cross_market_arb")
        if self.market_maker:
            await self.market_maker.start()
            log.info("app.strategy_started", strategy="market_maker")

        self._running = True
        log.info(
            "app.started",
            demo_mode=self.config.exchange.demo_mode,
            model_loaded=self.model is not None,
            strategies_active=[
                s for s, v in [
                    ("resolution_sniper", self.resolution_sniper),
                    ("live_event_edge", self.live_event_edge),
                    ("cross_market_arb", self.cross_market_arb),
                    ("market_maker", self.market_maker),
                ] if v is not None
            ],
        )

    async def stop(self) -> None:
        """Gracefully shut down all services."""
        if not self._running:
            return

        log.info("app.stopping")

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self.execution_engine:
            try:
                await self.execution_engine.stop()
                log.info("app.execution_engine_stopped")
            except Exception:
                log.exception("app.execution_engine_stop_error")

        # Stop strategies
        for name, strategy in [
            ("resolution_sniper", self.resolution_sniper),
            ("live_event_edge", self.live_event_edge),
            ("cross_market_arb", self.cross_market_arb),
            ("market_maker", self.market_maker),
        ]:
            if strategy:
                try:
                    await strategy.stop()
                    log.info("app.strategy_stopped", strategy=name)
                except Exception:
                    log.exception("app.strategy_stop_error", strategy=name)

        # Stop recorder (flushes remaining data)
        await self.recorder.stop()

        # Close clients
        await self.rest_client.close()
        await self.alert_manager.close()
        for closeable in self._async_closeables:
            close = getattr(closeable, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("app.async_closeable_error", closeable=type(closeable).__name__)

        # Close data store
        self.store.close()

        self._running = False
        log.info("app.stopped")

    async def run(self) -> None:
        """Start the application and block until a shutdown signal.

        Handles SIGINT and SIGTERM for graceful shutdown.
        """
        await self.start()

        shutdown = asyncio.Event()

        def _signal_handler() -> None:
            log.info("app.shutdown_signal")
            shutdown.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        try:
            await shutdown.wait()
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _monitoring_loop(self) -> None:
        """Periodically check drift, calibration, and regime status."""
        interval = 60.0  # Check every minute
        while True:
            try:
                # Auto-seed drift reference from first window of predictions
                if not self.drift_detector._reference_seeded:
                    recent = list(self.drift_detector._recent)
                    if len(recent) >= self.drift_detector._window_size:
                        self.drift_detector.set_reference(np.array(recent))

                # Check drift
                drift_result = self.drift_detector.check_drift()
                if drift_result.is_drifted:
                    await self.alert_manager.alert_drift_detected(
                        drift_result.metric_name,
                        drift_result.metric_value,
                        drift_result.severity,
                    )

                # Check calibration
                if self.calibration_monitor.is_degraded():
                    metrics = self.calibration_monitor.get_rolling_metrics()
                    await self.alert_manager.alert_calibration_degraded(
                        metrics.ece,
                        self.config.monitoring.ece_threshold,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("app.monitoring_error")

            await asyncio.sleep(interval)


def build_app(config: AppConfig) -> Application:
    """Construct all application components in dependency order.

    Parameters
    ----------
    config:
        Fully resolved application configuration.

    Returns
    -------
    Application
        Wired application instance ready for :meth:`Application.start`.
    """
    # 1. Logging (already set up by caller, but ensure it is done)
    setup_logging(config.log_level)
    log.info("build_app.start")

    # 2. REST client (auth is handled internally)
    rest_client = KalshiRestClient(config.exchange)

    # 3. Data store
    db_path = Path(config.data_dir) / "moneygone.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = DataStore(db_path)
    store.initialize_schema()
    log.info("build_app.store_ready", path=str(db_path))

    # 4. Data recorder
    recorder = MarketDataRecorder(store)

    # 5. Model + feature pipeline
    pipeline: Any | None = None
    model: Any | None = None
    execution_engine: ExecutionEngine | None = None
    async_closeables: list[Any] = []
    sports_only_mode = bool(config.sportsbook.enabled and config.sportsbook.leagues)

    # 7. Signal generation
    fee_calculator = KalshiFeeCalculator()
    edge_calculator = EdgeCalculator(
        fee_calculator=fee_calculator,
        min_edge_threshold=config.execution.min_edge_threshold,
        max_edge_sanity=config.execution.max_edge_sanity,
    )
    signal_filter = SignalFilter(
        risk_config=config.risk,
        execution_config=config.execution,
    )
    log.info("build_app.signals_ready")

    # 8. Monitoring
    #    Drift detector starts with an empty reference; it will be seeded
    #    from actual model predictions once enough data is collected, rather
    #    than using random noise which would cause false positive drift alerts.
    drift_detector = DriftDetector(
        reference_distribution=np.array([]),
        window_size=config.monitoring.drift_window,
        psi_critical=config.monitoring.psi_threshold,
    )

    calibration_monitor = CalibrationMonitor(
        ece_threshold=config.monitoring.ece_threshold,
    )

    regime_detector = RegimeDetector()

    pnl_tracker = PnLTracker()

    alert_manager = AlertManager(config=config.monitoring)

    log.info("build_app.monitoring_ready")

    # 9. Execution components
    order_manager = OrderManager(rest_client)
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
    kelly_sizer = KellySizer(
        kelly_fraction=config.risk.kelly_fraction,
        max_position_pct=config.risk.max_total_exposure_pct,
    )
    log.info("build_app.execution_ready")

    # 10. Data feeds
    espn_feed = None
    weather_feed = None
    crypto_feed = None
    if not sports_only_mode:
        espn_feed = ESPNLiveFeed()
        weather_feed = LiveWeatherFeed()
        crypto_feed = CryptoDataFeed(config.crypto) if config.crypto.enabled else None
        async_closeables.extend([espn_feed, weather_feed])
        if crypto_feed is not None:
            async_closeables.append(crypto_feed)

    # 11. Strategies
    resolution_sniper = None
    live_event_edge = None
    cross_market_arb = None
    market_maker = None

    if not sports_only_mode:
        resolution_sniper = ResolutionSniper(
            rest_client=rest_client,
            order_manager=order_manager,
            fee_calculator=fee_calculator,
            contract_mappings=[],  # auto-discovered at start()
            config=SnipeConfig(),
        )

        live_event_edge = LiveEventEdge(
            rest_client=rest_client,
            espn_feed=espn_feed,
            weather_feed=weather_feed,
            fee_calculator=fee_calculator,
            edge_calculator=edge_calculator,
            sizer=kelly_sizer,
            risk_manager=risk_manager,
            order_manager=order_manager,
            config=LiveEdgeConfig(),
        )

        cross_market_arb = CrossMarketArbitrage(
            rest_client=rest_client,
            fee_calculator=fee_calculator,
            order_manager=order_manager,
            config=ArbConfig(),
        )

        market_maker = MarketMaker(
            rest_client=rest_client,
            order_manager=order_manager,
            fee_calculator=fee_calculator,
            config=MMConfig(),
        )

    if sports_only_mode:
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
        ws_client = KalshiWebSocket(config.exchange)
        fill_tracker = FillTracker(store=store)
        sports_provider = StoreBackedSportsSnapshotProvider(
            store,
            leagues=config.sportsbook.leagues,
            rest_client=rest_client,
            max_line_age=recommended_max_line_age(
                config.sportsbook.fetch_interval_minutes,
            ),
        )
        async_closeables.append(sports_provider)
        inner_strategy = (
            PassiveStrategy(timeout_seconds=30.0)
            if config.execution.prefer_maker
            else AggressiveStrategy()
        )
        strategy = (
            DryRunStrategy(inner_strategy) if config.exchange.demo_mode
            else inner_strategy
        )

        # Weather category provider — NWS + NOAA + ECMWF
        category_providers: dict[MarketCategory, CategoryProvider] = {}
        if config.weather.enabled:
            weather_locations = config.weather.locations or [
                {"name": "New York", "lat": 40.7128, "lon": -74.0060},
                {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
                {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
                {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
                {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
                {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
            ]
            nws_fetcher = NWSFetcher()
            noaa_fetcher = NOAAEnsembleFetcher()
            ecmwf_fetcher = ECMWFOpenDataFetcher()
            weather_provider = WeatherDataProvider(
                noaa_fetcher=noaa_fetcher,
                ecmwf_fetcher=ecmwf_fetcher,
                locations=weather_locations,
                nws_fetcher=nws_fetcher,
            )
            weather_pipeline = FeaturePipeline(
                [
                    EnsembleMean(),
                    EnsembleSpread(),
                    EnsembleExceedanceProb(),
                    ForecastRevisionMagnitude(),
                    ForecastRevisionDirection(),
                    ModelDisagreement(),
                    ForecastHorizon(),
                    ClimatologicalAnomaly(),
                ],
                store=store,
            )
            weather_model = WeatherEnsembleModel()
            category_providers[MarketCategory.WEATHER] = CategoryProvider(
                category=MarketCategory.WEATHER,
                model=weather_model,
                pipeline=weather_pipeline,
                get_context_data=weather_provider.get_context,
            )
            async_closeables.append(weather_provider)
            log.info(
                "build_app.weather_ready",
                locations=[loc["name"] for loc in weather_locations],
            )

        # Register market-baseline model for all categories that lack a
        # specialised data feed (politics, economics, financials, companies,
        # crypto-without-feed, unknown).  The baseline model uses only
        # orderbook microstructure features and needs no external data.
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
        _baseline_cats = [
            MarketCategory.ECONOMICS,
            MarketCategory.POLITICS,
            MarketCategory.FINANCIALS,
            MarketCategory.COMPANIES,
            MarketCategory.CRYPTO,
            MarketCategory.UNKNOWN,
        ]
        for cat in _baseline_cats:
            if cat not in category_providers:
                category_providers[cat] = CategoryProvider(
                    category=cat,
                    model=baseline_model,
                    pipeline=baseline_pipeline,
                    get_context_data=None,  # No external data needed
                )
        log.info(
            "build_app.baseline_categories_ready",
            categories=[c.value for c in _baseline_cats if c in category_providers and category_providers[c].model is baseline_model],
        )

        # Load sportsbook data from parquet (written by collector worker)
        data_dir = Path(config.data_dir)
        sportsbook_parquet = data_dir / "sportsbook_lines.parquet"
        if sportsbook_parquet.exists():
            try:
                count = store.load_parquet_into_table(
                    "sportsbook_game_lines", sportsbook_parquet,
                )
                log.info("build_app.sportsbook_parquet_loaded", rows=count)
            except Exception:
                log.warning("build_app.sportsbook_parquet_failed", exc_info=True)
        else:
            log.info(
                "build_app.no_sportsbook_parquet",
                msg="Sports data will load when collector exports parquet",
            )

        execution_engine = ExecutionEngine(
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
            recorder=recorder,
            category_providers=category_providers,
            sportsbook_parquet_path=sportsbook_parquet,
        )
        log.info(
            "build_app.sports_execution_ready",
            leagues=config.sportsbook.leagues,
            model=model.name,
        )
    else:
        model = _load_latest_model(config)

    log.info("build_app.strategies_ready")

    # 12. Build application
    app = Application(
        config=config,
        rest_client=rest_client,
        store=store,
        recorder=recorder,
        fee_calculator=fee_calculator,
        edge_calculator=edge_calculator,
        signal_filter=signal_filter,
        drift_detector=drift_detector,
        calibration_monitor=calibration_monitor,
        regime_detector=regime_detector,
        pnl_tracker=pnl_tracker,
        alert_manager=alert_manager,
        model=model,
        pipeline=pipeline,
        execution_engine=execution_engine,
        resolution_sniper=resolution_sniper,
        live_event_edge=live_event_edge,
        cross_market_arb=cross_market_arb,
        market_maker=market_maker,
        async_closeables=async_closeables,
    )

    log.info(
        "build_app.complete",
        demo_mode=config.exchange.demo_mode,
        model_loaded=model is not None,
    )

    return app


def _load_latest_model(config: AppConfig) -> Any | None:
    """Attempt to load the most recently trained model from disk.

    Returns ``None`` if no model files are found or loading fails.
    """
    model_dir = Path(config.model.model_dir)
    if not model_dir.exists():
        log.info("build_app.no_model_dir", path=str(model_dir))
        return None

    # Find the newest pickled model artifact anywhere under the model dir.
    pkl_files = sorted(model_dir.rglob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not pkl_files:
        log.info("build_app.no_models_found", path=str(model_dir))
        return None

    for latest in reversed(pkl_files):
        try:
            with open(latest, "rb") as f:
                artifact = pickle.load(f)  # noqa: S301
        except Exception:
            log.warning(
                "build_app.model_load_failed",
                path=str(latest),
                exc_info=True,
            )
            continue

        feature_names = artifact.get("feature_names", []) if isinstance(artifact, dict) else []
        if _artifact_has_label_leakage(feature_names):
            log.warning(
                "build_app.model_rejected",
                path=str(latest),
                reason="label_leakage_features",
            )
            continue

        log.info(
            "build_app.model_loaded",
            path=str(latest),
            model_type=artifact.get("type", "unknown") if isinstance(artifact, dict) else type(artifact).__name__,
        )
        return artifact

    log.warning("build_app.no_safe_models_found", path=str(model_dir))
    return None


def _artifact_has_label_leakage(feature_names: list[str]) -> bool:
    leaked = {"settlement_value", "has_settlement_value"}
    return any(name in leaked for name in feature_names)
