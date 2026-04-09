#!/usr/bin/env python3
"""Backtest runner CLI.

Loads a trained model from the registry (or from file), runs the
BacktestEngine over a specified date range, prints a summary, and
optionally saves results.

Usage::

    python scripts/run_backtest.py --model models/gbm_20240101_120000.pkl \\
        --start 2024-07-01 --end 2024-12-31
    python scripts/run_backtest.py --model models/gbm_20240101_120000.pkl \\
        --bankroll 5000
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.store import DataStore
from moneygone.research.report import ReportGenerator
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _load_model(model_path: str) -> dict[str, Any]:
    """Load a pickled model artifact from disk."""
    path = Path(model_path)
    if not path.exists():
        log.error("backtest.model_not_found", path=model_path)
        sys.exit(1)

    with open(path, "rb") as f:
        artifact = pickle.load(f)  # noqa: S301

    log.info(
        "backtest.model_loaded",
        path=model_path,
        model_type=artifact.get("type", "unknown"),
        features=len(artifact.get("feature_names", [])),
    )
    return artifact


class _BacktestModel:
    """Wrapper that adapts a sklearn model artifact to the predict interface."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self._model = artifact["model"]
        self._scaler = artifact.get("scaler")
        self._feature_names = artifact.get("feature_names", [])

    def predict(self, features: dict[str, float]) -> float:
        """Return P(YES) for a feature dict."""
        import pandas as pd

        row = {name: features.get(name, 0.0) for name in self._feature_names}
        X = pd.DataFrame([row])

        if self._scaler is not None:
            X = self._scaler.transform(X)

        probas = self._model.predict_proba(X)[:, 1]
        return float(probas[0])


def _run_backtest(
    store: DataStore,
    model: _BacktestModel,
    start: datetime,
    end: datetime,
    initial_bankroll: float,
    config: Any,
) -> dict[str, Any]:
    """Run a simple backtest over the specified date range.

    This implements a basic backtest loop.  When the full BacktestEngine
    is available, it should be used instead.
    """
    from moneygone.signals.edge import EdgeCalculator, EdgeResult
    from moneygone.signals.fees import KalshiFeeCalculator

    fee_calc = KalshiFeeCalculator()
    edge_calc = EdgeCalculator(
        fee_calculator=fee_calc,
        min_edge_threshold=config.execution.min_edge_threshold,
        max_edge_sanity=config.execution.max_edge_sanity,
    )

    # Query all tickers that have both predictions and settlements
    conn = store._conn  # noqa: SLF001
    ticker_query = """
        SELECT DISTINCT p.ticker
        FROM predictions p
        JOIN settlements_log s ON p.ticker = s.ticker
        WHERE p.prediction_time >= ? AND p.prediction_time <= ?
    """
    tickers_raw = conn.execute(ticker_query, [start, end]).fetchall()
    tickers = [row[0] for row in tickers_raw]

    log.info("backtest.tickers_found", count=len(tickers))

    bankroll = initial_bankroll
    trades: list[dict[str, Any]] = []
    daily_pnl: dict[str, float] = {}

    for ticker in tickers:
        features = store.get_features_at(ticker, end)
        if not features:
            continue

        # Get settlement for outcome
        settlements = store.get_settlements(ticker)
        if not settlements:
            continue
        settlement = settlements[-1]

        try:
            prob = model.predict(features)
        except Exception:
            log.debug("backtest.predict_failed", ticker=ticker, exc_info=True)
            continue

        # Get last known orderbook
        ob_data = store.get_orderbook_at(ticker, end)
        if ob_data is None:
            continue

        from decimal import Decimal

        from moneygone.exchange.types import OrderbookLevel, OrderbookSnapshot
        from moneygone.utils.time import parse_iso

        yes_levels = tuple(
            OrderbookLevel(
                price=Decimal(str(lv[0])),
                contracts=Decimal(str(lv[1])),
            )
            for lv in ob_data.get("yes_levels", [])
        )
        no_levels = tuple(
            OrderbookLevel(
                price=Decimal(str(lv[0])),
                contracts=Decimal(str(lv[1])),
            )
            for lv in ob_data.get("no_levels", [])
        )

        snapshot_ts = ob_data.get("snapshot_time")
        if isinstance(snapshot_ts, str):
            snapshot_ts = parse_iso(snapshot_ts)
        elif not isinstance(snapshot_ts, datetime):
            snapshot_ts = datetime.now(timezone.utc)

        ob = OrderbookSnapshot(
            ticker=ticker,
            yes_levels=yes_levels,
            no_levels=no_levels,
            seq=ob_data.get("seq", 0),
            timestamp=snapshot_ts,
        )

        edge = edge_calc.compute_edge(prob, ob)

        if not edge.is_actionable:
            continue

        # Position sizing: fractional Kelly
        kelly_fraction = config.risk.kelly_fraction
        price = float(edge.target_price)
        if price <= 0 or price >= 1:
            continue

        # Kelly: f* = (p*b - q) / b where b = (1-price)/price, p = model_prob
        b = (1.0 - price) / price
        p = prob if edge.side == "yes" else (1.0 - prob)
        q = 1.0 - p
        kelly_full = (p * b - q) / b if b > 0 else 0
        bet_fraction = max(0, kelly_full * kelly_fraction)
        bet_amount = bankroll * bet_fraction
        contracts = max(1, int(bet_amount / price))
        contracts = min(contracts, config.risk.max_position_per_market)

        # Compute PnL
        result = settlement["market_result"]
        if edge.side == "yes":
            if result in ("yes", "all_yes"):
                pnl = (1.0 - price) * contracts
            else:
                pnl = -price * contracts
        else:
            if result in ("no", "all_no"):
                pnl = (1.0 - (1.0 - price)) * contracts
            else:
                pnl = -(1.0 - price) * contracts

        # Deduct fees
        fee = float(fee_calc.taker_fee(contracts, Decimal(str(price))))
        net_pnl = pnl - fee

        bankroll += net_pnl

        trade_record = {
            "ticker": ticker,
            "side": edge.side,
            "price": price,
            "contracts": contracts,
            "edge": edge.fee_adjusted_edge,
            "pnl": round(net_pnl, 4),
            "result": result,
        }
        trades.append(trade_record)

        # Daily PnL tracking
        day_key = settlement.get("settled_time", "unknown")
        if isinstance(day_key, datetime):
            day_key = day_key.strftime("%Y-%m-%d")
        elif isinstance(day_key, str):
            day_key = day_key[:10]
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + net_pnl

    # Compile results
    pnls = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]

    total_pnl = sum(pnls)
    max_drawdown = 0.0
    peak = initial_bankroll
    running = initial_bankroll
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, dd)

    result = {
        "total_pnl": round(total_pnl, 2),
        "num_trades": len(trades),
        "win_rate": len(winners) / len(trades) if trades else 0,
        "sharpe_ratio": (
            float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
            if len(pnls) > 1 and np.std(pnls) > 0
            else 0.0
        ),
        "max_drawdown": round(max_drawdown, 4),
        "final_bankroll": round(bankroll, 2),
        "return_pct": round((bankroll - initial_bankroll) / initial_bankroll * 100, 2),
        "trades": trades,
        "daily_pnl": daily_pnl,
        "config": {
            "initial_bankroll": initial_bankroll,
            "kelly_fraction": config.risk.kelly_fraction,
            "min_edge": config.execution.min_edge_threshold,
        },
    }

    return result


def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    config = load_config(base_path=Path(args.config))
    setup_logging(config.log_level)

    log.info(
        "run_backtest.starting",
        model=args.model,
        start=args.start,
        end=args.end,
        bankroll=args.bankroll,
    )

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    # Load model
    artifact = _load_model(args.model)
    model = _BacktestModel(artifact)

    # Override bankroll if specified
    if args.bankroll is not None:
        initial_bankroll = args.bankroll
    else:
        initial_bankroll = config.backtest.initial_bankroll

    # Open data store
    db_path = Path(config.data_dir) / "moneygone.duckdb"
    if not db_path.exists():
        log.error("backtest.db_not_found", path=str(db_path))
        sys.exit(1)

    store = DataStore(db_path)

    try:
        result = _run_backtest(
            store, model, start, end, initial_bankroll, config
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Period:          {args.start} to {args.end}")
        print(f"  Initial Bankroll: ${initial_bankroll:,.2f}")
        print(f"  Final Bankroll:   ${result['final_bankroll']:,.2f}")
        print(f"  Total PnL:        ${result['total_pnl']:,.2f}")
        print(f"  Return:           {result['return_pct']:.2f}%")
        print(f"  Trades:           {result['num_trades']}")
        print(f"  Win Rate:         {result['win_rate']:.1%}")
        print(f"  Sharpe Ratio:     {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {result['max_drawdown']:.2%}")
        print("=" * 60)

        # Generate report
        report = ReportGenerator.generate_backtest_report(result)

        # Save results
        results_dir = Path(config.data_dir) / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"backtest_{timestamp}.json"
        report_path = results_dir / f"backtest_{timestamp}_report.md"

        # Save JSON results (without non-serializable objects)
        serializable = {
            k: v
            for k, v in result.items()
            if k != "trades" or isinstance(v, (list, dict))
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        with open(report_path, "w") as f:
            f.write(report)

        log.info(
            "run_backtest.complete",
            results_path=str(results_path),
            report_path=str(report_path),
            total_pnl=result["total_pnl"],
        )

    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest over historical data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model pickle file",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-07-01",
        help="Backtest start date (YYYY-MM-DD, default: 2024-07-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="Backtest end date (YYYY-MM-DD, default: 2024-12-31)",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Initial bankroll (overrides config, default: from config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
