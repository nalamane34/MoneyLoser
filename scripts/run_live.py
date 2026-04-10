#!/usr/bin/env python3
"""Live / paper trading runner.

Wires all system components together and starts the trading pipeline.
Supports both paper trading (demo API) and live trading (real API)
via config overlays.

Usage::

    # Paper trading soak (default)
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
from moneygone.utils.env import load_repo_env
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)

DEFAULT_SOAK_OVERLAY = Path("config/paper-soak.yaml")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_overlay_path(args: argparse.Namespace) -> Path:
    """Return the requested overlay or the conservative soak default."""
    return Path(args.overlay) if args.overlay else DEFAULT_SOAK_OVERLAY


def _is_paper_soak_overlay(overlay: Path) -> bool:
    """Identify the conservative soak overlay, regardless of path spelling."""
    return overlay.resolve() == DEFAULT_SOAK_OVERLAY.resolve()


def _validate_paper_soak_config(config) -> None:
    """Guardrails for the conservative sports soak profile."""
    if not config.exchange.demo_mode:
        raise ValueError("paper soak must run in demo mode")
    if not config.sportsbook.enabled:
        raise ValueError("paper soak must enable sportsbook polling")
    if config.crypto.enabled:
        raise ValueError("paper soak must keep crypto feeds disabled")
    if config.weather.enabled:
        raise ValueError("paper soak must keep weather feeds disabled")

    required_bookmakers = {"pinnacle"}
    expected_markets = {"h2h"}
    if not required_bookmakers.issubset(set(config.sportsbook.bookmakers)):
        raise ValueError("paper soak must include pinnacle")
    if set(config.sportsbook.markets) != expected_markets:
        raise ValueError("paper soak must use h2h only")
    if config.sportsbook.fetch_interval_minutes < 15:
        raise ValueError("paper soak polling is too aggressive")
    if config.sportsbook.min_requests_remaining < 200:
        raise ValueError("paper soak reserve is too low")
    if config.execution.evaluation_interval_seconds < 10.0:
        raise ValueError("paper soak evaluation loop is too aggressive")


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the live trading system."""
    loaded_env = load_repo_env(REPO_ROOT)

    # Load configuration with optional overlay
    overlay = _resolve_overlay_path(args)
    config = load_config(
        base_path=Path(args.config),
        overlay_path=overlay,
    )

    # Setup logging first
    setup_logging(config.log_level)
    if loaded_env:
        log.info(
            "run_live.repo_env_loaded",
            path=str(REPO_ROOT / ".env"),
            keys=sorted(loaded_env),
        )

    if _is_paper_soak_overlay(overlay):
        _validate_paper_soak_config(config)

    log.info(
        "run_live.starting",
        config_path=args.config,
        overlay_path=str(overlay),
        demo_mode=config.exchange.demo_mode,
        paper_soak=_is_paper_soak_overlay(overlay),
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
        sportsbook_enabled=config.sportsbook.enabled,
        sportsbook_poll_interval_minutes=config.sportsbook.fetch_interval_minutes,
        sportsbook_quota_reserve=config.sportsbook.min_requests_remaining,
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
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            log.debug("run_live.signal_handler_unsupported", signal=str(sig))

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
        default=str(DEFAULT_SOAK_OVERLAY),
        help=(
            "Path to config overlay YAML, e.g. config/paper.yaml or "
            "config/live.yaml (default: config/paper-soak.yaml)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
