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
import signal
import sys
from pathlib import Path

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.store import DataStore
from moneygone.monitoring.alerts import AlertManager
from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.monitor")


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
                        recent_preds = exec_store.query(
                            "SELECT probability FROM predictions ORDER BY prediction_time DESC LIMIT 100"
                        )
                        for row in recent_preds:
                            drift_detector.add_prediction(float(row[0]))
                    finally:
                        exec_store.close()
                except Exception:
                    log.debug("monitor.prediction_load_failed", exc_info=True)

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

    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)

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
