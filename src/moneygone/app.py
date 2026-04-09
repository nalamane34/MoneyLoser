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
from moneygone.data.sports.espn import ESPNLiveFeed
from moneygone.data.sports.live_weather import LiveWeatherFeed
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.execution.order_manager import OrderManager
from moneygone.monitoring.alerts import AlertManager
from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker
from moneygone.monitoring.regime_detector import RegimeDetector
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
        resolution_sniper: ResolutionSniper | None = None,
        live_event_edge: LiveEventEdge | None = None,
        cross_market_arb: CrossMarketArbitrage | None = None,
        market_maker: MarketMaker | None = None,
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
        self.resolution_sniper = resolution_sniper
        self.live_event_edge = live_event_edge
        self.cross_market_arb = cross_market_arb
        self.market_maker = market_maker

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

    # 5. Feature pipeline (load features if available)
    pipeline = None
    try:
        from moneygone.features.pipeline import FeaturePipeline
        # Pipeline requires registered features; will be initialized
        # when model and features are configured.
        log.debug("build_app.pipeline_available")
    except ImportError:
        log.debug("build_app.pipeline_not_available")

    # 6. Model loading
    model = _load_latest_model(config)

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
    #    Drift detector needs a reference distribution; use uniform as default
    reference_dist = np.random.default_rng(42).uniform(0, 1, size=1000)
    drift_detector = DriftDetector(
        reference_distribution=reference_dist,
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
        limits=risk_limits,
        portfolio=portfolio_tracker,
        drawdown=drawdown_monitor,
        exposure=exposure_calculator,
    )
    kelly_sizer = KellySizer(
        fraction=config.risk.kelly_fraction,
        max_position_pct=config.risk.max_total_exposure_pct,
    )
    log.info("build_app.execution_ready")

    # 10. Data feeds
    espn_feed = ESPNLiveFeed()
    weather_feed = LiveWeatherFeed()
    crypto_feed = CryptoDataFeed(config.crypto) if config.crypto.enabled else None

    # 11. Strategies
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
        resolution_sniper=resolution_sniper,
        live_event_edge=live_event_edge,
        cross_market_arb=cross_market_arb,
        market_maker=market_maker,
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

    # Find the newest .pkl file
    pkl_files = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not pkl_files:
        log.info("build_app.no_models_found", path=str(model_dir))
        return None

    latest = pkl_files[-1]
    try:
        with open(latest, "rb") as f:
            artifact = pickle.load(f)  # noqa: S301
        log.info(
            "build_app.model_loaded",
            path=str(latest),
            model_type=artifact.get("type", "unknown"),
        )
        return artifact
    except Exception:
        log.warning(
            "build_app.model_load_failed",
            path=str(latest),
            exc_info=True,
        )
        return None
