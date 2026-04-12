from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker


def _load_worker_monitor_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "worker_monitor.py"
    spec = importlib.util.spec_from_file_location("worker_monitor", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("worker_monitor", None)
    spec.loader.exec_module(module)
    return module


class _FakeExecStore:
    def __init__(self, predictions, fills, settlements) -> None:
        self._predictions = predictions
        self._fills = fills
        self._settlements = settlements

    def query(self, sql: str, params=None):
        if "FROM predictions" in sql:
            return list(self._predictions)
        if "FROM fills_log" in sql:
            return list(self._fills)
        if "FROM settlements_log" in sql:
            return list(self._settlements)
        raise AssertionError(f"Unexpected SQL: {sql}")


def test_consume_incremental_rows_skips_already_seen_same_timestamp() -> None:
    module = _load_worker_monitor_module()
    t1 = datetime(2026, 4, 11, 3, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 11, 3, 1, tzinfo=timezone.utc)
    rows = [
        ("A", t1),
        ("B", t1),
        ("C", t2),
    ]

    first_rows, cursor_time, cursor_keys = module._consume_incremental_rows(
        rows,
        time_index=1,
        key_fn=lambda row: str(row[0]),
        cursor_time=None,
        cursor_keys=set(),
    )
    second_rows, _, _ = module._consume_incremental_rows(
        rows,
        time_index=1,
        key_fn=lambda row: str(row[0]),
        cursor_time=cursor_time,
        cursor_keys=cursor_keys,
    )

    assert [row[0] for row in first_rows] == ["A", "B", "C"]
    assert [row[0] for row in second_rows] == []


def test_hydrate_from_execution_store_is_incremental() -> None:
    module = _load_worker_monitor_module()
    now = datetime(2026, 4, 11, 3, 5, tzinfo=timezone.utc)
    exec_store = _FakeExecStore(
        predictions=[
            ("KXTEST", "model-a", 0.73, now, now),
        ],
        fills=[
            (
                "fill-1",
                "KXTEST",
                "yes",
                "buy",
                2,
                0.71,
                0.01,
                True,
                now,
                0.73,
                0.84,
                0.05,
                0.04,
                "sports",
                now,
            ),
        ],
        settlements=[
            ("KXTEST", "yes", 0.58, now, now),
        ],
    )

    drift_detector = DriftDetector(reference_distribution=np.array([]), window_size=10)
    calibration_monitor = CalibrationMonitor()
    pnl_tracker = PnLTracker()

    cursors = module._hydrate_from_execution_store(
        exec_store=exec_store,
        drift_detector=drift_detector,
        calibration_monitor=calibration_monitor,
        pnl_tracker=pnl_tracker,
        prediction_cursor_time=None,
        prediction_cursor_keys=set(),
        fill_cursor_time=None,
        fill_cursor_keys=set(),
        settlement_cursor_time=None,
        settlement_cursor_keys=set(),
    )
    assert len(drift_detector._recent) == 1
    assert calibration_monitor.get_rolling_metrics().n_resolved == 1
    assert pnl_tracker.get_summary().num_trades == 1

    cursors = module._hydrate_from_execution_store(
        exec_store=exec_store,
        drift_detector=drift_detector,
        calibration_monitor=calibration_monitor,
        pnl_tracker=pnl_tracker,
        prediction_cursor_time=cursors[0],
        prediction_cursor_keys=cursors[1],
        fill_cursor_time=cursors[2],
        fill_cursor_keys=cursors[3],
        settlement_cursor_time=cursors[4],
        settlement_cursor_keys=cursors[5],
    )
    assert len(drift_detector._recent) == 1
    assert calibration_monitor.get_rolling_metrics().n_resolved == 1
    assert pnl_tracker.get_summary().num_trades == 1
