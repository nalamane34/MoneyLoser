#!/usr/bin/env python3
"""Backtest runner: replays historical data through the full decision pipeline.

Uses the BacktestEngine to replay candlestick-derived market states and
orderbook snapshots from backfilled DuckDB databases.  Each category
(sports, weather, crypto) uses its own model and feature pipeline —
the same ones used in live trading.

Usage::

    # Run all categories (auto-discovers DBs in data/)
    python scripts/run_backtest.py

    # Single category
    python scripts/run_backtest.py --category sports --db data/backtest_sports.duckdb

    # Custom date range & bankroll
    python scripts/run_backtest.py --category weather --start 2026-01-01 --end 2026-01-08 --bankroll 5000

    # Use a specific fill model
    python scripts/run_backtest.py --fill-model realistic --slippage-bps 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.backtest.context_providers import (
    CryptoBacktestContextProvider,
    SportsBacktestContextProvider,
    WeatherBacktestContextProvider,
)
from moneygone.backtest.data_loader import HistoricalDataLoader
from moneygone.backtest.engine import BacktestEngine
from moneygone.backtest.guards import LeakageGuard
from moneygone.backtest.results import BacktestResult
from moneygone.config import BacktestConfig, RiskConfig, load_config
from moneygone.data.store import DataStore
from moneygone.execution.simulator import FillSimulator
from moneygone.features.pipeline import FeaturePipeline
from moneygone.models.base import ProbabilityModel
from moneygone.signals.edge import EdgeCalculator
from moneygone.signals.fees import KalshiFeeCalculator
from moneygone.sizing.kelly import KellySizer
from moneygone.sizing.risk_limits import RiskLimits
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("run_backtest")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Category-specific pipeline builders
# ---------------------------------------------------------------------------


def build_sports_pipeline(store: DataStore) -> tuple[ProbabilityModel, FeaturePipeline]:
    """Build the SharpSportsbook model + feature pipeline."""
    from moneygone.features import (
        HomeFieldAdvantage,
        KalshiVsSportsbookEdge,
        MoneylineMovement,
        PinnacleVsMarketEdge,
        PinnacleWinProbability,
        PowerRatingEdge,
        SpreadImpliedWinProb,
        SportsbookWinProbability,
        TeamInjuryImpact,
    )
    from moneygone.models.sharp_sportsbook import SharpSportsbookModel

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
    return model, pipeline


def build_weather_pipeline(store: DataStore) -> tuple[ProbabilityModel, FeaturePipeline]:
    """Build the WeatherEnsemble model + feature pipeline."""
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
    from moneygone.models.weather_ensemble import WeatherEnsembleModel

    model = WeatherEnsembleModel()
    pipeline = FeaturePipeline(
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
    return model, pipeline


def build_crypto_pipeline(store: DataStore) -> tuple[ProbabilityModel, FeaturePipeline]:
    """Build the CryptoVol model + feature pipeline."""
    from moneygone.features.crypto_features import (
        ATR14,
        ATR24,
        BRTIDistanceToThreshold,
        BRTIPrice,
        BasisSpread,
        CryptoOrderbookImbalance,
        FundingRateSignal,
        FundingRateZScore,
        HoursToExpiry,
        ImpliedVolatility,
        OpenInterestChange,
        RealizedVol7d,
        RealizedVol24h,
        RealizedVol30d,
        ThresholdPrice,
        TrendRegime,
        TrendStrength,
        VolSpread,
        VolatilityRegime,
        WhaleFlowIndicator,
    )
    from moneygone.models.crypto_vol import CryptoVolModel

    model = CryptoVolModel()
    pipeline = FeaturePipeline(
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
            ThresholdPrice(),
            HoursToExpiry(),
        ],
        store=store,
    )
    return model, pipeline


def build_baseline_pipeline(store: DataStore) -> tuple[ProbabilityModel, FeaturePipeline]:
    """Build the MarketBaseline model (demo/stress-test only)."""
    from moneygone.features import (
        BidAskSpread,
        DepthRatio,
        MidPrice,
        OrderbookImbalance,
        TimeToExpiry,
        WeightedMidPrice,
    )
    from moneygone.models.market_baseline import MarketBaselineModel

    model = MarketBaselineModel()
    pipeline = FeaturePipeline(
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
    return model, pipeline


PIPELINE_BUILDERS = {
    "sports": build_sports_pipeline,
    "weather": build_weather_pipeline,
    "crypto": build_crypto_pipeline,
    "baseline": build_baseline_pipeline,
}


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def run_category_backtest(
    category: str,
    db_path: Path,
    config: Any,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    bankroll: float = 10000.0,
    fill_model: str = "realistic",
    slippage_bps: float = 0.0,
    model_override: str | None = None,
) -> BacktestResult | None:
    """Run a backtest for a single category.

    Returns the BacktestResult, or None if no events were found.
    """
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST: {category.upper()}")
    print(f"  Database: {db_path}")
    print(f"{'=' * 60}")

    if not db_path.exists():
        print(f"  ERROR: Database not found at {db_path}")
        return None

    store = DataStore(db_path)

    try:
        # Check DB contents
        for table in ["market_states", "orderbook_snapshots", "trades", "settlements_log"]:
            try:
                count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table:25s} {count:>8} rows")
            except Exception:
                print(f"  {table:25s} (not found)")

        # Auto-detect date range from data if not specified
        if start_date is None or end_date is None:
            try:
                result = store._conn.execute(
                    "SELECT MIN(snapshot_time), MAX(snapshot_time) FROM orderbook_snapshots"
                ).fetchone()
                if result and result[0]:
                    data_start = result[0]
                    data_end = result[1]
                    if not isinstance(data_start, datetime):
                        data_start = datetime.fromisoformat(str(data_start))
                    if not isinstance(data_end, datetime):
                        data_end = datetime.fromisoformat(str(data_end))

                    if data_start.tzinfo is None:
                        data_start = data_start.replace(tzinfo=timezone.utc)
                    if data_end.tzinfo is None:
                        data_end = data_end.replace(tzinfo=timezone.utc)

                    start_date = start_date or data_start
                    end_date = end_date or data_end
                    print(f"\n  Auto-detected date range: {start_date.date()} → {end_date.date()}")
                else:
                    print("  ERROR: No orderbook data found in DB")
                    return None
            except Exception as e:
                print(f"  ERROR: Could not detect date range: {e}")
                return None

        # Build pipeline for this category (or use override model)
        model_key = model_override or category
        builder = PIPELINE_BUILDERS.get(model_key)
        if builder is None:
            print(f"  ERROR: Unknown model/category '{model_key}'. Available: {list(PIPELINE_BUILDERS.keys())}")
            return None

        model, pipeline = builder(store)
        print(f"\n  Model:    {model.name} v{model.version}")
        print(f"  Features: {len(pipeline._ordered)}")
        print(f"  Bankroll: ${bankroll:,.2f}")
        print(f"  Fill:     {fill_model} (slippage={slippage_bps}bps)")

        # Build backtest components
        fee_calc = KalshiFeeCalculator()
        edge_calc = EdgeCalculator(
            fee_calculator=fee_calc,
            min_edge_threshold=getattr(config.execution, "min_edge_threshold", 0.02),
            max_edge_sanity=getattr(config.execution, "max_edge_sanity", 0.30),
        )
        sizer = KellySizer(
            kelly_fraction=getattr(config.risk, "kelly_fraction", 0.25),
            max_position_pct=getattr(config.risk, "max_total_exposure_pct", 0.50),
        )
        risk_limits = RiskLimits(config.risk)
        fill_sim = FillSimulator(
            model=fill_model,
            fee_calculator=fee_calc,
            slippage_bps=slippage_bps,
        )
        leakage_guard = LeakageGuard({})
        data_loader = HistoricalDataLoader(store)

        backtest_config = BacktestConfig(
            initial_bankroll=bankroll,
            fill_model=fill_model,
            slippage_bps=slippage_bps,
        )

        # Build context provider for categories that need external data
        context_provider = None
        effective_category = model_key if model_override else category
        if effective_category == "crypto":
            # Check if crypto_context table exists
            try:
                ctx_count = store._conn.execute(
                    "SELECT COUNT(*) FROM crypto_context"
                ).fetchone()[0]
                if ctx_count > 0:
                    context_provider = CryptoBacktestContextProvider(store)
                    print(f"  Context:  CryptoBacktestContextProvider ({ctx_count} snapshots)")
                else:
                    print("  Context:  NONE (crypto_context table empty)")
            except Exception:
                print("  Context:  NONE (crypto_context table not found)")
                print("           Run: python scripts/backfill_crypto_context.py")
        elif effective_category == "sports":
            try:
                ctx_count = store._conn.execute(
                    "SELECT COUNT(*) FROM sports_context"
                ).fetchone()[0]
                if ctx_count > 0:
                    context_provider = SportsBacktestContextProvider(store)
                    print(f"  Context:  SportsBacktestContextProvider ({ctx_count} snapshots)")
            except Exception:
                print("  Context:  NONE (sports_context table not found)")
        elif effective_category == "weather":
            try:
                ctx_count = store._conn.execute(
                    "SELECT COUNT(*) FROM weather_context"
                ).fetchone()[0]
                if ctx_count > 0:
                    context_provider = WeatherBacktestContextProvider(store)
                    print(f"  Context:  WeatherBacktestContextProvider ({ctx_count} snapshots)")
            except Exception:
                print("  Context:  NONE (weather_context table not found)")

        engine = BacktestEngine(
            data_loader=data_loader,
            feature_pipeline=pipeline,
            model=model,
            edge_calculator=edge_calc,
            sizer=sizer,
            risk_limits=risk_limits,
            fill_simulator=fill_sim,
            leakage_guard=leakage_guard,
            config=backtest_config,
            risk_config=config.risk,
            fee_calculator=fee_calc,
            evaluation_on_orderbook=True,
            progress_interval=500,
            context_provider=context_provider,
            data_store=store,
        )

        # Run backtest
        print(f"\n  Running backtest {start_date.date()} → {end_date.date()}...")
        t0 = time.time()

        result = engine.run(
            start_date=start_date,
            end_date=end_date,
            initial_bankroll=Decimal(str(bankroll)),
        )

        elapsed = time.time() - t0

        # Print results
        print(f"\n  {'─' * 50}")
        print(f"  RESULTS ({elapsed:.1f}s)")
        print(f"  {'─' * 50}")
        print(f"  Trades:           {result.num_trades:>8}")
        print(f"  Net PnL:          ${float(result.net_pnl):>+10.2f}")
        print(f"  Total Fees:       ${float(result.total_fees):>10.2f}")
        print(f"  Win Rate:         {result.win_rate:>9.1%}")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:>9.2f}")
        print(f"  Max Drawdown:     {result.max_drawdown:>9.2%}")
        print(f"  Avg Edge (pred):  {result.avg_edge_predicted:>9.4f}")
        print(f"  Avg Edge (real):  {result.avg_edge_realized:>9.4f}")
        print(f"  Brier Score:      {result.brier_score:>9.4f}")

        final_equity = float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else bankroll
        ret_pct = (final_equity - bankroll) / bankroll * 100
        print(f"  Final Equity:     ${final_equity:>10.2f}")
        print(f"  Return:           {ret_pct:>+9.2f}%")
        print(f"  {'─' * 50}")

        return result

    except Exception as e:
        log.error("backtest.failed", category=category, error=str(e), exc_info=True)
        print(f"\n  ERROR: {e}")
        return None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests against historical data")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--overlay", default=None, help="Overlay config (optional)")
    parser.add_argument("--category", default="", help="Category to backtest (sports, weather, crypto, or blank for all)")
    parser.add_argument("--model", default="", help="Override model (baseline, sports, weather, crypto)")
    parser.add_argument("--db", default="", help="DuckDB path (auto-detected if not specified)")
    parser.add_argument("--start", default="", help="Start date YYYY-MM-DD (auto-detected from data)")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD (auto-detected from data)")
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Initial bankroll (default: 10000)")
    parser.add_argument("--fill-model", default="realistic", choices=["instant", "queue", "realistic"],
                        help="Fill simulation model (default: realistic)")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Additional slippage in basis points")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    overlay_path = Path(args.overlay) if args.overlay else None
    config = load_config(config_path, overlay_path)
    setup_logging("INFO")

    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start else None
    )
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else None
    )

    # Determine which categories to run
    if args.category:
        categories = [c.strip().lower() for c in args.category.split(",")]
    else:
        # Auto-discover from available DB files
        categories = []
        for name in ["sports", "weather", "crypto"]:
            if (DATA_DIR / f"backtest_{name}.duckdb").exists():
                categories.append(name)
        if not categories:
            # Fall back to single DB
            if (DATA_DIR / "backtest.duckdb").exists():
                categories = ["baseline"]

    if not categories:
        print("ERROR: No backtest databases found in data/")
        print("  Run backfill first: python scripts/backfill_historical.py --help")
        sys.exit(1)

    print("=" * 60)
    print("  BACKTEST SUITE")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Bankroll:   ${args.bankroll:,.2f}")
    print(f"  Fill model: {args.fill_model}")
    print("=" * 60)

    results: dict[str, BacktestResult] = {}
    all_start = time.time()

    for category in categories:
        # Determine DB path
        if args.db:
            db_path = Path(args.db)
        else:
            db_path = DATA_DIR / f"backtest_{category}.duckdb"

        result = run_category_backtest(
            category=category,
            db_path=db_path,
            config=config,
            start_date=start_date,
            end_date=end_date,
            bankroll=args.bankroll,
            fill_model=args.fill_model,
            slippage_bps=args.slippage_bps,
            model_override=args.model or None,
        )

        if result is not None:
            results[category] = result

    # Summary across all categories
    total_elapsed = time.time() - all_start

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  COMBINED SUMMARY")
        print(f"{'=' * 60}")
        total_trades = sum(r.num_trades for r in results.values())
        total_pnl = sum(float(r.net_pnl) for r in results.values())
        total_fees = sum(float(r.total_fees) for r in results.values())

        print(f"  {'Category':<12} {'Trades':>8} {'Net PnL':>12} {'Win%':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'─' * 58}")
        for cat, r in results.items():
            print(
                f"  {cat:<12} {r.num_trades:>8} "
                f"${float(r.net_pnl):>+10.2f} "
                f"{r.win_rate:>7.1%} "
                f"{r.sharpe_ratio:>7.2f} "
                f"{r.max_drawdown:>7.2%}"
            )
        print(f"  {'─' * 58}")
        print(
            f"  {'TOTAL':<12} {total_trades:>8} "
            f"${total_pnl:>+10.2f} "
            f"{'':>7s} "
            f"{'':>7s} "
            f"{'':>7s}"
        )
        print(f"  Total fees: ${total_fees:.2f}")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Save results
    if args.save:
        results_dir = DATA_DIR / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for cat, result in results.items():
            out_path = results_dir / f"backtest_{cat}_{timestamp}.json"
            summary = {
                "category": cat,
                "num_trades": result.num_trades,
                "net_pnl": float(result.net_pnl),
                "total_fees": float(result.total_fees),
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "brier_score": result.brier_score,
                "avg_edge_predicted": result.avg_edge_predicted,
                "avg_edge_realized": result.avg_edge_realized,
                "trades": result.trades,
            }
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
