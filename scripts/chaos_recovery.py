#!/usr/bin/env python3
"""Chaos recovery drill for the MoneyGone worker stack.

Kills one or more worker processes under the supervisor and verifies that:
  1. the supervisor notices the exit,
  2. the worker is restarted with a new PID, and
  3. the restarted worker emits a category-appropriate health event.

This is designed to run on the host where ``scripts/run_workers.py`` is
already running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


WORKER_SCRIPTS = {
    "collector": "worker_collector.py",
    "market_data": "worker_market_data.py",
    "execution": "worker_execution.py",
    "monitor": "worker_monitor.py",
}

DEFAULT_HEALTH_EVENTS = {
    "collector": ["collector.started"],
    "market_data": ["market_data.started"],
    "execution": ["execution.starting", "engine.orders_reconciled"],
    "monitor": ["monitor.started"],
}


@dataclass
class DrillResult:
    worker: str
    old_pid: int
    new_pid: int | None = None
    success: bool = False
    seen_events: list[str] = field(default_factory=list)
    error: str = ""


def parse_event_line(line: str) -> dict[str, object] | None:
    """Extract a JSON event from a supervisor log line."""
    json_start = line.find("{")
    if json_start == -1:
        return None
    try:
        return json.loads(line[json_start:])
    except json.JSONDecodeError:
        return None


def required_health_events(worker: str) -> list[str]:
    """Return the required post-restart health signals for a worker."""
    return list(DEFAULT_HEALTH_EVENTS.get(worker, []))


def match_worker_process(args: str, worker: str) -> bool:
    """Return True when a ps command line belongs to the named worker."""
    script = WORKER_SCRIPTS[worker]
    return script in args


def list_worker_pids(worker: str) -> list[int]:
    """Return live PIDs for a worker script on the current host."""
    output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    pids: list[int] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if match_worker_process(args, worker):
            pids.append(int(pid_text))
    return pids


def pid_exists(pid: int) -> bool:
    """Return True if the PID still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid_exit(pid: int, timeout: float) -> bool:
    """Wait for a PID to disappear."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_exists(pid):
            return True
        time.sleep(0.2)
    return not pid_exists(pid)


def _record_seen(seen: list[str], event: str) -> None:
    if event not in seen:
        seen.append(event)


def run_worker_drill(
    worker: str,
    *,
    log_path: Path,
    timeout: float,
    terminate_signal: signal.Signals,
    cool_down: float,
) -> DrillResult:
    """Kill a worker and verify supervisor restart + health signal."""
    pids = list_worker_pids(worker)
    if len(pids) != 1:
        raise RuntimeError(
            f"expected exactly one PID for {worker}, found {pids or 'none'}"
        )

    old_pid = pids[0]
    result = DrillResult(worker=worker, old_pid=old_pid)
    log_offset = log_path.stat().st_size if log_path.exists() else 0

    os.kill(old_pid, terminate_signal)
    wait_for_pid_exit(old_pid, timeout=min(timeout, 15.0))

    required_supervisor_events = {
        "supervisor.worker_exited",
        "supervisor.restarting_worker",
        "supervisor.worker_started",
    }
    required_worker_events = set(required_health_events(worker))
    deadline = time.time() + timeout

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(log_offset)
        while time.time() < deadline:
            line = handle.readline()
            if not line:
                time.sleep(0.2)
                continue

            event = parse_event_line(line)
            if event is None:
                continue

            name = str(event.get("event", ""))
            if name == "supervisor.worker_exited" and event.get("worker") == worker:
                _record_seen(result.seen_events, name)
            elif name == "supervisor.restarting_worker" and event.get("worker") == worker:
                _record_seen(result.seen_events, name)
            elif name == "supervisor.worker_started" and event.get("worker") == worker:
                new_pid = int(event.get("pid", 0))
                if new_pid and new_pid != old_pid:
                    result.new_pid = new_pid
                    _record_seen(result.seen_events, name)
            elif name in required_worker_events:
                _record_seen(result.seen_events, name)

            if required_supervisor_events.issubset(result.seen_events) and required_worker_events.issubset(result.seen_events):
                break

    missing = sorted(
        required_supervisor_events.union(required_worker_events)
        - set(result.seen_events)
    )
    if missing:
        result.error = f"missing events: {', '.join(missing)}"
        return result

    if result.new_pid is None or not pid_exists(result.new_pid):
        result.error = "restarted worker PID not alive"
        return result

    time.sleep(cool_down)
    result.success = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker chaos recovery drill")
    parser.add_argument(
        "--workers",
        default="execution,market_data,collector,monitor",
        help="Comma-separated worker list",
    )
    parser.add_argument(
        "--log",
        default="logs/supervisor.log",
        help="Supervisor log path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-worker timeout in seconds",
    )
    parser.add_argument(
        "--cool-down",
        type=float,
        default=2.0,
        help="Pause between drills so the stack can settle",
    )
    parser.add_argument(
        "--signal",
        default="TERM",
        choices=("TERM", "KILL"),
        help="Signal used to kill each worker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    invalid = [w for w in workers if w not in WORKER_SCRIPTS]
    if invalid:
        print(f"unknown workers: {', '.join(invalid)}", file=sys.stderr)
        return 2

    terminate_signal = getattr(signal, f"SIG{args.signal}")
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"log file not found: {log_path}", file=sys.stderr)
        return 2

    results: list[DrillResult] = []
    for worker in workers:
        print(f"== drill: {worker} ==", flush=True)
        try:
            result = run_worker_drill(
                worker,
                log_path=log_path,
                timeout=args.timeout,
                terminate_signal=terminate_signal,
                cool_down=args.cool_down,
            )
        except Exception as exc:  # pragma: no cover - surfaced to user directly
            print(f"{worker}: FAIL ({exc})", flush=True)
            return 1
        results.append(result)
        if result.success:
            print(
                f"{worker}: PASS old_pid={result.old_pid} new_pid={result.new_pid} "
                f"events={','.join(result.seen_events)}",
                flush=True,
            )
        else:
            print(
                f"{worker}: FAIL old_pid={result.old_pid} "
                f"events={','.join(result.seen_events)} error={result.error}",
                flush=True,
            )
            return 1

    print(f"completed {len(results)} worker drills successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
