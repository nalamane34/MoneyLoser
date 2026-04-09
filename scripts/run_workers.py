#!/usr/bin/env python3
"""Supervisor: launches all worker processes and manages their lifecycle.

Starts writer processes first (collector, market_data), waits for DB
initialization, then starts readers (execution, monitor).

Handles SIGINT/SIGTERM by forwarding to all children for graceful shutdown.
Automatically restarts crashed workers up to a configurable limit.

Usage::

    # Start all workers
    python scripts/run_workers.py

    # Start specific workers
    python scripts/run_workers.py --workers collector,market_data

    # Custom config
    python scripts/run_workers.py --config config/default.yaml --overlay config/paper-soak.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure structlog is available
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("supervisor")

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# Worker definitions: (name, script, is_writer)
# Writers start first so their DBs exist before readers try to ATTACH.
WORKER_DEFS = [
    ("collector", "worker_collector.py", True),
    ("market_data", "worker_market_data.py", True),
    ("execution", "worker_execution.py", True),   # writes to execution.duckdb
    ("monitor", "worker_monitor.py", False),       # read-only
]

MAX_RESTARTS = 5
RESTART_WINDOW_SECONDS = 300  # 5 minutes


class WorkerProcess:
    """Manages a single worker subprocess."""

    def __init__(self, name: str, cmd: list[str], is_writer: bool) -> None:
        self.name = name
        self.cmd = cmd
        self.is_writer = is_writer
        self.process: asyncio.subprocess.Process | None = None
        self.restart_times: list[datetime] = []
        self.intentional_stop = False

    async def start(self) -> None:
        """Launch the subprocess."""
        log.info("supervisor.starting_worker", worker=self.name, cmd=" ".join(self.cmd))
        self.process = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Launch log reader
        asyncio.create_task(self._read_output(), name=f"log_{self.name}")
        log.info("supervisor.worker_started", worker=self.name, pid=self.process.pid)

    async def stop(self, timeout: float = 10.0) -> None:
        """Send SIGTERM and wait for exit, SIGKILL if timeout."""
        if self.process is None or self.process.returncode is not None:
            return
        self.intentional_stop = True
        log.info("supervisor.stopping_worker", worker=self.name, pid=self.process.pid)
        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("supervisor.killing_worker", worker=self.name, pid=self.process.pid)
            self.process.kill()
            await self.process.wait()

    def can_restart(self) -> bool:
        """Check if we haven't exceeded restart limits."""
        now = datetime.now(timezone.utc)
        # Prune old restart times
        cutoff = now.timestamp() - RESTART_WINDOW_SECONDS
        self.restart_times = [t for t in self.restart_times if t.timestamp() > cutoff]
        return len(self.restart_times) < MAX_RESTARTS

    def record_restart(self) -> None:
        self.restart_times.append(datetime.now(timezone.utc))

    async def _read_output(self) -> None:
        """Read stdout/stderr and log with worker name prefix."""
        if self.process is None or self.process.stdout is None:
            return
        try:
            async for line in self.process.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    # Print directly with worker prefix for clean log output
                    print(f"[{self.name}] {text}", flush=True)
        except Exception:
            pass


async def run_supervisor(
    workers_to_run: list[str],
    config_path: str,
    overlay_path: str,
) -> None:
    """Main supervisor loop."""
    # Build worker processes
    workers: list[WorkerProcess] = []
    for name, script, is_writer in WORKER_DEFS:
        if name not in workers_to_run:
            continue
        cmd = [
            PYTHON, str(SCRIPTS_DIR / script),
            "--config", config_path,
            "--overlay", overlay_path,
        ]
        workers.append(WorkerProcess(name, cmd, is_writer))

    if not workers:
        log.error("supervisor.no_workers_selected")
        return

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("supervisor.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Phase 1: Start writers first
    writers = [w for w in workers if w.is_writer]
    readers = [w for w in workers if not w.is_writer]

    for w in writers:
        await w.start()

    if writers and readers:
        log.info("supervisor.waiting_for_writer_init", seconds=3)
        await asyncio.sleep(3)

    # Phase 2: Start readers
    for w in readers:
        await w.start()

    log.info(
        "supervisor.all_started",
        workers=[w.name for w in workers],
        pids={w.name: w.process.pid for w in workers if w.process},
    )

    # Phase 3: Monitor and restart
    try:
        while not shutdown.is_set():
            for w in workers:
                if w.process is None or w.process.returncode is None:
                    continue

                # Worker has exited
                rc = w.process.returncode
                if w.intentional_stop:
                    continue

                log.warning(
                    "supervisor.worker_exited",
                    worker=w.name,
                    return_code=rc,
                )

                if w.can_restart():
                    w.record_restart()
                    log.info("supervisor.restarting_worker", worker=w.name)
                    await asyncio.sleep(2)
                    await w.start()
                else:
                    log.error(
                        "supervisor.restart_limit_exceeded",
                        worker=w.name,
                        max_restarts=MAX_RESTARTS,
                        window_seconds=RESTART_WINDOW_SECONDS,
                    )

            await asyncio.sleep(2)
    except asyncio.CancelledError:
        pass

    # Shutdown all workers
    log.info("supervisor.shutting_down_workers")
    await asyncio.gather(*(w.stop() for w in workers))
    log.info("supervisor.all_stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="MoneyGone worker supervisor")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    parser.add_argument(
        "--workers",
        default="collector,market_data,execution,monitor",
        help="Comma-separated list of workers to run (default: all)",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    workers_to_run = [w.strip() for w in args.workers.split(",")]
    valid_names = {name for name, _, _ in WORKER_DEFS}
    invalid = set(workers_to_run) - valid_names
    if invalid:
        print(f"Unknown workers: {invalid}. Valid: {valid_names}", file=sys.stderr)
        sys.exit(1)

    log.info(
        "supervisor.starting",
        workers=workers_to_run,
        config=args.config,
        overlay=args.overlay,
    )

    asyncio.run(run_supervisor(workers_to_run, args.config, args.overlay))


if __name__ == "__main__":
    main()
