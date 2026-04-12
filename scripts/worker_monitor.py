#!/usr/bin/env python3
"""Worker: Read-only monitoring and alerting.

Periodically reads from all databases, checks drift/calibration/PnL,
and sends alerts. Makes no writes and has no Kalshi connection.

Usage::

    python scripts/worker_monitor.py --config config/default.yaml --overlay config/paper-soak.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.store import DataStore
from moneygone.exchange.types import MarketResult, Settlement
from moneygone.monitoring.alerts import AlertManager
from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker
from moneygone.utils.env import load_repo_env
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.monitor")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _coerce_timestamp(value: Any) -> datetime:
    """Normalize DB timestamps to timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp value: {value!r}")


def _consume_incremental_rows(
    rows: list[tuple[Any, ...]],
    *,
    time_index: int,
    key_fn,
    cursor_time: datetime | None,
    cursor_keys: set[str],
) -> tuple[list[tuple[Any, ...]], datetime | None, set[str]]:
    """Filter rows to only those newer than the cursor, preserving ties safely."""
    new_rows: list[tuple[Any, ...]] = []
    latest_time = cursor_time
    latest_keys = set(cursor_keys) if cursor_time is not None else set()

    for row in rows:
        row_time = _coerce_timestamp(row[time_index])
        row_key = key_fn(row)
        if cursor_time is not None:
            if row_time < cursor_time:
                continue
            if row_time == cursor_time and row_key in cursor_keys:
                continue
        new_rows.append(row)
        if latest_time is None or row_time > latest_time:
            latest_time = row_time
            latest_keys = {row_key}
        elif row_time == latest_time:
            latest_keys.add(row_key)

    return new_rows, latest_time, latest_keys


def _hydrate_from_execution_store(
    *,
    exec_store: DataStore,
    drift_detector: DriftDetector,
    calibration_monitor: CalibrationMonitor,
    pnl_tracker: PnLTracker,
    prediction_cursor_time: datetime | None,
    prediction_cursor_keys: set[str],
    fill_cursor_time: datetime | None,
    fill_cursor_keys: set[str],
    settlement_cursor_time: datetime | None,
    settlement_cursor_keys: set[str],
) -> tuple[
    datetime | None,
    set[str],
    datetime | None,
    set[str],
    datetime | None,
    set[str],
]:
    """Consume new predictions, fills, and settlements from execution DB."""
    prediction_rows = exec_store.query(
        """
        SELECT ticker, model_name, probability, prediction_time, ingested_at
        FROM predictions
        {where_clause}
        ORDER BY ingested_at ASC, prediction_time ASC, ticker ASC, model_name ASC
        LIMIT 5000
        """.format(
            where_clause=(
                "WHERE ingested_at >= $cutoff"
                if prediction_cursor_time is not None
                else ""
            )
        ),
        (
            {"cutoff": prediction_cursor_time}
            if prediction_cursor_time is not None
            else None
        ),
    )
    (
        new_predictions,
        prediction_cursor_time,
        prediction_cursor_keys,
    ) = _consume_incremental_rows(
        prediction_rows,
        time_index=4,
        key_fn=lambda row: (
            f"{row[0]}|{row[1]}|{_coerce_timestamp(row[3]).isoformat()}"
        ),
        cursor_time=prediction_cursor_time,
        cursor_keys=prediction_cursor_keys,
    )
    for row in new_predictions:
        probability = float(row[2])
        ticker = str(row[0])
        drift_detector.add_prediction(probability)
        calibration_monitor.record_prediction(probability, ticker)

    fill_rows = exec_store.query(
        """
        SELECT trade_id, ticker, side, action, count, price, fee_paid,
               is_taker, fill_time, predicted_prob, predicted_confidence,
               raw_edge, fee_adjusted_edge, category, ingested_at
        FROM fills_log
        {where_clause}
        ORDER BY ingested_at ASC, fill_time ASC, trade_id ASC
        LIMIT 5000
        """.format(
            where_clause=(
                "WHERE ingested_at >= $cutoff"
                if fill_cursor_time is not None
                else ""
            )
        ),
        (
            {"cutoff": fill_cursor_time}
            if fill_cursor_time is not None
            else None
        ),
    )
    (
        new_fills,
        fill_cursor_time,
        fill_cursor_keys,
    ) = _consume_incremental_rows(
        fill_rows,
        time_index=14,
        key_fn=lambda row: str(row[0]),
        cursor_time=fill_cursor_time,
        cursor_keys=fill_cursor_keys,
    )
    for row in new_fills:
        pnl_tracker.record_trade_snapshot(
            trade_id=str(row[0]),
            ticker=str(row[1]),
            side=str(row[2]),
            action=str(row[3]),
            count=int(row[4]),
            price=float(row[5]),
            fee_paid=float(row[6] or 0.0),
            is_taker=bool(row[7]),
            fill_time=_coerce_timestamp(row[8]),
            predicted_prob=float(row[9] or 0.0),
            predicted_confidence=float(row[10] or 0.0),
            raw_edge=float(row[11] or 0.0),
            fee_adjusted_edge=float(row[12] or 0.0),
            category=str(row[13] or ""),
        )

    settlement_rows = exec_store.query(
        """
        SELECT ticker, market_result, revenue, settled_time, ingested_at
        FROM settlements_log
        {where_clause}
        ORDER BY ingested_at ASC, settled_time ASC, ticker ASC, market_result ASC
        LIMIT 5000
        """.format(
            where_clause=(
                "WHERE ingested_at >= $cutoff"
                if settlement_cursor_time is not None
                else ""
            )
        ),
        (
            {"cutoff": settlement_cursor_time}
            if settlement_cursor_time is not None
            else None
        ),
    )
    (
        new_settlements,
        settlement_cursor_time,
        settlement_cursor_keys,
    ) = _consume_incremental_rows(
        settlement_rows,
        time_index=4,
        key_fn=lambda row: (
            f"{row[0]}|{row[1]}|{_coerce_timestamp(row[3]).isoformat()}"
        ),
        cursor_time=settlement_cursor_time,
        cursor_keys=settlement_cursor_keys,
    )
    for row in new_settlements:
        settlement = Settlement(
            ticker=str(row[0]),
            market_result=MarketResult(str(row[1])),
            revenue=Decimal(str(row[2])),
            settled_time=_coerce_timestamp(row[3]),
        )
        pnl_tracker.record_settlement(settlement)
        if settlement.market_result in {
            MarketResult.YES,
            MarketResult.ALL_YES,
        }:
            calibration_monitor.record_outcome(settlement.ticker, True)
        elif settlement.market_result in {
            MarketResult.NO,
            MarketResult.ALL_NO,
        }:
            calibration_monitor.record_outcome(settlement.ticker, False)

    return (
        prediction_cursor_time,
        prediction_cursor_keys,
        fill_cursor_time,
        fill_cursor_keys,
        settlement_cursor_time,
        settlement_cursor_keys,
    )


async def _monitoring_loop(
    config,
    drift_detector: DriftDetector,
    calibration_monitor: CalibrationMonitor,
    pnl_tracker: PnLTracker,
    alert_manager: AlertManager,
    exec_db_path: Path,
) -> None:
    """Periodic monitoring checks."""
    interval = 60.0
    prediction_cursor_time: datetime | None = None
    prediction_cursor_keys: set[str] = set()
    fill_cursor_time: datetime | None = None
    fill_cursor_keys: set[str] = set()
    settlement_cursor_time: datetime | None = None
    settlement_cursor_keys: set[str] = set()

    while True:
        try:
            # Auto-seed drift reference from first window of predictions
            if not drift_detector._reference_seeded:
                recent = list(drift_detector._recent)
                if len(recent) >= drift_detector._window_size:
                    drift_detector.set_reference(np.array(recent))

            # Load recent predictions from execution DB
            # Open briefly and close — don't hold a lock that blocks the writer
            if exec_db_path.exists():
                try:
                    exec_store = DataStore(exec_db_path, read_only=True)
                    try:
                        (
                            prediction_cursor_time,
                            prediction_cursor_keys,
                            fill_cursor_time,
                            fill_cursor_keys,
                            settlement_cursor_time,
                            settlement_cursor_keys,
                        ) = _hydrate_from_execution_store(
                            exec_store=exec_store,
                            drift_detector=drift_detector,
                            calibration_monitor=calibration_monitor,
                            pnl_tracker=pnl_tracker,
                            prediction_cursor_time=prediction_cursor_time,
                            prediction_cursor_keys=prediction_cursor_keys,
                            fill_cursor_time=fill_cursor_time,
                            fill_cursor_keys=fill_cursor_keys,
                            settlement_cursor_time=settlement_cursor_time,
                            settlement_cursor_keys=settlement_cursor_keys,
                        )
                    finally:
                        exec_store.close()
                except Exception:
                    log.debug("monitor.execution_db_load_failed", exc_info=True)

            # Check drift
            drift_result = drift_detector.check_drift()
            if drift_result.is_drifted:
                await alert_manager.alert_drift_detected(
                    drift_result.metric_name,
                    drift_result.metric_value,
                    drift_result.severity,
                )

            # Check calibration
            if calibration_monitor.is_degraded():
                metrics = calibration_monitor.get_rolling_metrics()
                await alert_manager.alert_calibration_degraded(
                    metrics.ece,
                    config.monitoring.ece_threshold,
                )

            # Log PnL summary
            pnl_summary = pnl_tracker.get_summary()
            if pnl_summary.num_trades > 0:
                log.info(
                    "monitor.pnl_summary",
                    net_pnl=round(pnl_summary.net_pnl, 4),
                    trades=pnl_summary.num_trades,
                    win_rate=round(pnl_summary.win_rate, 4) if pnl_summary.win_rate else 0,
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("monitor.check_error")

        await asyncio.sleep(interval)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring worker")
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
        log.info("worker_monitor.repo_env_loaded", keys=sorted(loaded_env))

    data_dir = Path(config.data_dir)
    exec_db_path = data_dir / "execution.duckdb"

    # Build monitoring components
    drift_detector = DriftDetector(
        reference_distribution=np.array([]),
        window_size=config.monitoring.drift_window,
        psi_critical=config.monitoring.psi_threshold,
    )
    calibration_monitor = CalibrationMonitor(
        ece_threshold=config.monitoring.ece_threshold,
    )
    pnl_tracker = PnLTracker()
    alert_manager = AlertManager(config=config.monitoring)

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("monitor.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    log.info("monitor.started")

    task = asyncio.create_task(
        _monitoring_loop(config, drift_detector, calibration_monitor, pnl_tracker, alert_manager, exec_db_path)
    )

    try:
        await shutdown.wait()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await alert_manager.close()
        log.info("monitor.stopped")


if __name__ == "__main__":
    asyncio.run(main())
