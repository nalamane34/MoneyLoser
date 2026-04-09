#!/usr/bin/env python3
"""Live / paper trading runner.

Wires all system components together and starts the trading pipeline.
Supports both paper trading (demo API) and live trading (real API)
via config overlays.

Usage::

    # Paper trading (default)
    python scripts/run_live.py

    # Paper trading with explicit overlay
    python scripts/run_live.py --overlay config/paper.yaml

    # Live trading
    python scripts/run_live.py --overlay config/live.yaml

    # Custom config
    python scripts/run_live.py --config config/custom.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.app import Application, build_app
from moneygone.config import load_config
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the live trading system."""
    # Load configuration with optional overlay
    overlay = Path(args.overlay) if args.overlay else None
    config = load_config(
        base_path=Path(args.config),
        overlay_path=overlay,
    )

    # Setup logging first
    setup_logging(config.log_level)

    log.info(
        "run_live.starting",
        config_path=args.config,
        overlay_path=args.overlay,
        demo_mode=config.exchange.demo_mode,
        log_level=config.log_level,
    )

    # Log configuration summary (sensitive fields redacted)
    log.info(
        "run_live.config",
        base_url=config.exchange.base_url,
        demo_mode=config.exchange.demo_mode,
        min_edge=config.execution.min_edge_threshold,
        kelly_fraction=config.risk.kelly_fraction,
        max_position=config.risk.max_position_per_market,
        daily_loss_limit=config.risk.daily_loss_limit_pct,
        max_drawdown=config.risk.max_drawdown_pct,
        data_dir=str(config.data_dir),
    )

    if not config.exchange.demo_mode:
        log.warning(
            "run_live.LIVE_MODE",
            message="Running with REAL MONEY. Double-check your configuration.",
        )

    # Build the application
    app = build_app(config)

    # Setup signal handling for graceful shutdown
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        log.info("run_live.shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    # Run the application
    try:
        await app.start()
        log.info("run_live.running", message="System is live. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await shutdown_event.wait()
    except Exception:
        log.exception("run_live.fatal_error")
    finally:
        log.info("run_live.shutting_down")
        await app.stop()
        log.info("run_live.stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MoneyGone live/paper trading system."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to base config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default=None,
        help=(
            "Path to config overlay YAML, e.g. config/paper.yaml or "
            "config/live.yaml (default: none)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
