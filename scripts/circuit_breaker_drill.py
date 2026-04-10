#!/usr/bin/env python3
"""Operational drill for execution circuit-breaker halt behavior.

Boots an isolated execution worker against a copied discovery cache with:

- sportsbook disabled,
- weather disabled,
- crypto disabled,
- max_drawdown_pct forced to 0.0.

That configuration should halt every candidate before any order is placed.
The running worker stack is untouched; this drill uses a temporary data dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution circuit-breaker drill")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/live.yaml")
    parser.add_argument(
        "--duration",
        type=int,
        default=35,
        help="How many seconds to let the isolated worker run",
    )
    return parser.parse_args()


def _extract_event(payload: str) -> dict[str, object] | None:
    json_start = payload.find("{")
    if json_start == -1:
        return None
    try:
        return json.loads(payload[json_start:])
    except json.JSONDecodeError:
        return None


def _build_overlay(base_overlay: Path, *, data_dir: Path) -> dict[str, object]:
    base = {}
    if base_overlay.exists():
        with base_overlay.open() as handle:
            base = yaml.safe_load(handle) or {}

    drill_overlay = {
        "data_dir": str(data_dir),
        "sportsbook": {"enabled": False},
        "weather": {"enabled": False},
        "crypto": {"enabled": False},
        "execution": {
            "evaluation_interval_seconds": 5.0,
        },
        "risk": {
            "max_drawdown_pct": 0.0,
        },
        "log_level": "INFO",
    }
    base.update(drill_overlay)
    return base


def _parse_results(log_path: Path) -> tuple[int, int, int]:
    circuit_breaker_rejects = 0
    selected = 0
    order_events = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            event = _extract_event(line)
            if event is None:
                continue
            name = str(event.get("event", ""))
            if name == "engine.candidate":
                status = str(event.get("status", ""))
                reason = str(event.get("reject_reason", ""))
                if status == "selected":
                    selected += 1
                if reason == "risk_rejected:circuit_breaker":
                    circuit_breaker_rejects += 1
            elif name in {"engine.order_executed", "passive.order_placed", "aggressive.order_placed"}:
                order_events += 1

    return circuit_breaker_rejects, selected, order_events


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / args.config
    overlay_config_path = repo_root / args.overlay
    config = load_config(base_config_path, overlay_config_path)
    source_data_dir = repo_root / config.data_dir
    cache_path = source_data_dir / "discovered_markets.json"
    if not cache_path.exists():
        print(f"discovery cache missing: {cache_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="moneygone-circuit-drill-") as tmpdir:
        tmp_root = Path(tmpdir)
        tmp_data_dir = tmp_root / "data"
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, tmp_data_dir / "discovered_markets.json")

        overlay_path = tmp_root / "circuit-breaker.yaml"
        with overlay_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                _build_overlay(overlay_config_path, data_dir=tmp_data_dir),
                handle,
                sort_keys=False,
            )

        log_path = tmp_root / "worker_execution.log"
        cmd = [
            "timeout",
            str(args.duration),
            sys.executable,
            str(repo_root / "scripts" / "worker_execution.py"),
            "--config",
            str(base_config_path),
            "--overlay",
            str(overlay_path),
        ]

        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )

        if proc.returncode not in (0, 124):
            print(f"worker_execution exited with unexpected code {proc.returncode}", flush=True)
            print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
            return 1

        circuit_breaker_rejects, selected, order_events = _parse_results(log_path)
        print(
            "circuit_breaker_rejects="
            f"{circuit_breaker_rejects} selected={selected} order_events={order_events} "
            f"duration_seconds={args.duration}",
            flush=True,
        )

        if circuit_breaker_rejects <= 0:
            print("FAIL: no circuit-breaker rejections observed", flush=True)
            return 1
        if selected > 0:
            print("FAIL: selected trades slipped through the circuit breaker", flush=True)
            return 1
        if order_events > 0:
            print("FAIL: order placement occurred during circuit-breaker drill", flush=True)
            return 1

        print("PASS: circuit breaker halted all trading activity", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
