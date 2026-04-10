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
import signal
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
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
from moneygone.execution.artifact_runtime import build_universal_artifact_fallbacks
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
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.execution")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Execution engine worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    args = parser.parse_args()

    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config.model.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir

    # Own database for writes (features, predictions, fills)
    store = DataStore(data_dir / "execution.duckdb")
    store.initialize_schema(EXECUTION_TABLES)

    collector_db = data_dir / "collector.duckdb"
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
            from moneygone.data.weather.noaa import NOAAEnsembleFetcher
            from moneygone.data.weather.ecmwf import ECMWFOpenDataFetcher

            noaa_fetcher = NOAAEnsembleFetcher(
                api_key=config.weather.open_meteo_api_key,
            )
            ecmwf_fetcher = ECMWFOpenDataFetcher()
            weather_locations = [
                {"name": loc["name"], "lat": loc["lat"], "lon": loc["lon"]}
                for loc in config.weather.locations
            ]
            weather_provider = WeatherDataProvider(noaa_fetcher, ecmwf_fetcher, weather_locations)
            weather_model = WeatherEnsembleModel()
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

    fallback_categories = [
        MarketCategory.ECONOMICS,
        MarketCategory.POLITICS,
        MarketCategory.FINANCIALS,
        MarketCategory.COMPANIES,
        MarketCategory.CRYPTO,
        MarketCategory.WEATHER,
        MarketCategory.ENTERTAINMENT,
        MarketCategory.UNKNOWN,
    ]
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
    )

    try:
        await engine.start()
        await shutdown.wait()
    finally:
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
