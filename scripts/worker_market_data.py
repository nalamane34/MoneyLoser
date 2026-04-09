#!/usr/bin/env python3
"""Worker: Kalshi market data recorder.

Connects to Kalshi REST API, polls market data on a short interval,
and writes snapshots to ``market_data.duckdb``.

Usage::

    python scripts/worker_market_data.py --config config/default.yaml --overlay config/paper-soak.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.schemas import MARKET_DATA_TABLES
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.market_data")


async def _poll_loop(
    rest_client: KalshiRestClient,
    recorder: MarketDataRecorder,
    poll_interval: float = 5.0,
) -> None:
    """Poll all open markets and orderbooks, write to DuckDB."""
    log.info("market_data.poll_started", interval=poll_interval)

    while True:
        try:
            markets = await rest_client.get_all_markets(status="open", limit=1000)
            if markets:
                for m in markets:
                    row = {
                        "ticker": m.ticker,
                        "event_ticker": m.event_ticker,
                        "title": m.title,
                        "status": m.status.value,
                        "yes_bid": float(m.yes_bid),
                        "yes_ask": float(m.yes_ask),
                        "last_price": float(m.last_price),
                        "volume": m.volume,
                        "open_interest": m.open_interest,
                        "close_time": m.close_time.isoformat(),
                        "result": m.result.value,
                        "category": m.category,
                    }
                    await recorder.on_ticker_update({"data": row})

            # Fetch orderbooks for a subset of active markets
            tickers = [m.ticker for m in markets[:50]] if markets else []
            if tickers:
                try:
                    orderbooks = await rest_client.get_orderbooks(tickers)
                    for ticker, ob_data in orderbooks.items():
                        from moneygone.exchange.types import OrderbookSnapshot
                        ob = OrderbookSnapshot.from_api_response(ticker, ob_data)
                        await recorder.on_orderbook_update({
                            "data": {
                                "ticker": ob.ticker,
                                "yes_levels": [
                                    [float(l.price), float(l.contracts)]
                                    for l in ob.yes_bids
                                ],
                                "no_levels": [
                                    [float(l.price), float(l.contracts)]
                                    for l in ob.no_bids
                                ],
                                "seq": ob.seq,
                                "snapshot_time": ob.timestamp.isoformat(),
                            }
                        })
                except Exception:
                    log.debug("market_data.orderbook_fetch_failed", exc_info=True)

            log.debug("market_data.polled", markets=len(markets) if markets else 0)

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market_data.poll_error")

        await asyncio.sleep(poll_interval)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi market data recorder worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = DataStore(data_dir / "market_data.duckdb")
    store.initialize_schema(MARKET_DATA_TABLES)

    rest_client = KalshiRestClient(config.exchange)
    recorder = MarketDataRecorder(store)
    await recorder.start()

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("market_data.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    task = asyncio.create_task(_poll_loop(rest_client, recorder, args.poll_interval))

    try:
        await shutdown.wait()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await recorder.stop()
        await rest_client.close()
        store.close()
        log.info("market_data.stopped")


if __name__ == "__main__":
    asyncio.run(main())
