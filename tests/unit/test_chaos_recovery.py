"""Tests for the worker chaos-recovery drill."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_chaos_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "chaos_recovery.py"
    spec = importlib.util.spec_from_file_location("chaos_recovery", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("chaos_recovery", None)
    sys.modules["chaos_recovery"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_event_line_supports_prefixed_worker_logs() -> None:
    module = _load_chaos_module()

    line = (
        '[execution] {"event":"engine.orders_reconciled","open_orders":2,'
        '"timestamp":"2026-04-10T00:00:00Z"}'
    )

    parsed = module.parse_event_line(line)

    assert parsed is not None
    assert parsed["event"] == "engine.orders_reconciled"
    assert parsed["open_orders"] == 2


def test_required_health_events_for_execution_include_reconcile() -> None:
    module = _load_chaos_module()

    events = module.required_health_events("execution")

    assert "execution.starting" in events
    assert "engine.orders_reconciled" in events


def test_match_worker_process_uses_expected_script_names() -> None:
    module = _load_chaos_module()

    assert module.match_worker_process(
        "/home/ubuntu/MoneyGone/scripts/worker_market_data.py --config x",
        "market_data",
    )
    assert not module.match_worker_process(
        "/home/ubuntu/MoneyGone/scripts/worker_execution.py --config x",
        "market_data",
    )
