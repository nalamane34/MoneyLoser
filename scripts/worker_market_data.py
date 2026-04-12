#!/usr/bin/env python3
"""Worker: Kalshi market data recorder.

Connects to Kalshi via WebSocket for real-time orderbook/trade/ticker data,
with REST polling as a complementary sweep for markets the WS isn't subscribed to.
Writes all data to ``market_data.duckdb``.

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
from moneygone.data.market_discovery import MarketDiscoveryService
from moneygone.data.schemas import MARKET_DATA_TABLES
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.ws_client import KalshiWebSocket
from moneygone.exchange.types import WSEvent
from moneygone.utils.env import load_repo_env
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.market_data")
REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDER_FLUSH_INTERVAL_SECONDS = 1.0
RECORDER_ORDERBOOK_INTERVAL_SECONDS = 2.0


async def _ws_event_handler(
    event: WSEvent,
    recorder: MarketDataRecorder,
) -> None:
    """Route WebSocket events to the recorder."""
    data = event.data
    ticker = data.get("market_ticker", data.get("ticker", ""))

    if event.channel == "ticker":
        await recorder.on_ticker_update({"data": {
            "ticker": ticker,
            "event_ticker": data.get("event_ticker", ""),
            "title": data.get("title", ""),
            "status": data.get("status", "open"),
            "yes_bid": data.get("yes_bid_dollars", data.get("yes_bid")),
            "yes_ask": data.get("yes_ask_dollars", data.get("yes_ask")),
            "last_price": data.get("last_price_dollars", data.get("last_price")),
            "volume": data.get("volume"),
            "open_interest": data.get("open_interest"),
            "close_time": data.get("close_time", datetime.now(tz=timezone.utc).isoformat()),
            "result": data.get("result"),
            "category": data.get("category"),
        }})

    elif event.channel == "trade":
        await recorder.on_trade({"data": {
            "trade_id": data.get("trade_id", ""),
            "ticker": ticker,
            "count": int(float(str(data.get("count_fp", data.get("count", 1))))),
            "yes_price": data.get("yes_price_dollars", data.get("yes_price")),
            "taker_side": data.get("taker_side", "unknown"),
            "trade_time": data.get("created_time", datetime.now(tz=timezone.utc).isoformat()),
        }})

    elif event.channel == "orderbook_delta":
        if event.type in ("orderbook_snapshot", "orderbook_delta"):
            ob = data.get("orderbook_fp", data)
            yes_raw = ob.get("yes_dollars", data.get("yes", []))
            no_raw = ob.get("no_dollars", data.get("no", []))
            await recorder.on_orderbook_update({"data": {
                "ticker": ticker,
                "yes_levels": [[str(l[0]), str(l[1])] for l in (yes_raw or [])],
                "no_levels": [[str(l[0]), str(l[1])] for l in (no_raw or [])],
                "seq": data.get("seq", event.seq),
                "snapshot_time": datetime.now(tz=timezone.utc).isoformat(),
            }})


async def _ws_loop(
    ws_client: KalshiWebSocket,
    discovery: MarketDiscoveryService,
    recorder: MarketDataRecorder,
    refresh_interval: float = 120.0,
) -> None:
    """Maintain WebSocket subscriptions using the shared discovery cache."""
    subscribed_tickers: set[str] = set()

    while True:
        try:
            # Read from shared discovery service (already refreshed on its own timer)
            current_tickers = discovery.get_tickers()

            # Subscribe to new tickers
            new_tickers = current_tickers - subscribed_tickers
            if new_tickers:
                ticker_list = list(new_tickers)
                # Subscribe in batches of 50
                for i in range(0, len(ticker_list), 50):
                    batch = ticker_list[i : i + 50]
                    await ws_client.subscribe_orderbook(batch)
                    await ws_client.subscribe_ticker(batch)
                    await ws_client.subscribe_trades(batch)
                subscribed_tickers.update(new_tickers)
                log.info("market_data.ws_subscribed", new=len(new_tickers), total=len(subscribed_tickers))

            log.debug("market_data.ws_refresh", markets=len(current_tickers), subscribed=len(subscribed_tickers))

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market_data.ws_refresh_error")

        await asyncio.sleep(refresh_interval)


async def _rest_sweep_loop(
    rest_client: KalshiRestClient,
    discovery: MarketDiscoveryService,
    recorder: MarketDataRecorder,
    poll_interval: float = 60.0,
) -> None:
    """Periodic REST sweep to catch any data the WebSocket might miss.

    Reads the market list from the shared discovery cache instead of
    fetching it again.  Only fetches orderbooks via REST for top markets.
    """
    log.info("market_data.rest_sweep_started", interval=poll_interval)

    while True:
        try:
            # Read from shared discovery cache (no extra API call)
            markets = discovery.get_markets()
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

            # Bulk orderbook fetch for top markets by volume
            if markets:
                top = sorted(markets, key=lambda m: m.volume, reverse=True)[:50]
                tickers = [m.ticker for m in top]
                try:
                    orderbooks = await rest_client.get_orderbooks(tickers)
                    for ob in orderbooks:
                        await recorder.on_orderbook_update({"data": {
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
                        }})
                except Exception:
                    log.debug("market_data.orderbook_sweep_failed", exc_info=True)

            log.debug("market_data.rest_sweep", markets=len(markets) if markets else 0)

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market_data.rest_sweep_error")

        await asyncio.sleep(poll_interval)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi market data recorder worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket, use REST-only polling")
    args = parser.parse_args()

    loaded_env = load_repo_env(REPO_ROOT)
    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)
    if loaded_env:
        log.info("worker_market_data.repo_env_loaded", keys=sorted(loaded_env))

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = DataStore(data_dir / "market_data.duckdb")
    store.initialize_schema(MARKET_DATA_TABLES)

    rest_client = KalshiRestClient(config.exchange)
    recorder = MarketDataRecorder(
        store,
        flush_interval_seconds=RECORDER_FLUSH_INTERVAL_SECONDS,
        orderbook_snapshot_interval=RECORDER_ORDERBOOK_INTERVAL_SECONDS,
    )
    await recorder.start()
    log.info(
        "market_data.recorder_configured",
        flush_interval_seconds=RECORDER_FLUSH_INTERVAL_SECONDS,
        orderbook_snapshot_interval=RECORDER_ORDERBOOK_INTERVAL_SECONDS,
    )

    # Shared market discovery — fetches all markets once and caches to JSON
    cache_path = data_dir / "discovered_markets.json"
    discovery = MarketDiscoveryService(
        rest_client=rest_client,
        cache_path=cache_path,
        refresh_interval=120.0,  # re-discover every 2 min
    )
    await discovery.start()
    log.info("market_data.discovery_started", cache_path=str(cache_path))

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("market_data.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    tasks: list[asyncio.Task] = []
    ws_client: KalshiWebSocket | None = None

    if not args.no_ws:
        # Start WebSocket for real-time data
        ws_client = KalshiWebSocket(
            config.exchange,
            on_event=lambda evt: _ws_event_handler(evt, recorder),
        )
        try:
            await ws_client.connect()
            await ws_client.subscribe_fills()
            log.info("market_data.ws_connected")

            # WS subscription manager — reads tickers from discovery cache
            tasks.append(asyncio.create_task(
                _ws_loop(ws_client, discovery, recorder),
                name="ws_loop",
            ))
        except Exception:
            log.warning("market_data.ws_connect_failed, falling back to REST-only", exc_info=True)
            ws_client = None

    # REST sweep — reads market list from discovery cache, only fetches orderbooks
    sweep_interval = args.poll_interval if ws_client is None else max(args.poll_interval, 60.0)
    tasks.append(asyncio.create_task(
        _rest_sweep_loop(rest_client, discovery, recorder, sweep_interval),
        name="rest_sweep",
    ))

    log.info("market_data.started", ws_active=ws_client is not None, rest_interval=sweep_interval)

    try:
        await shutdown.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await discovery.stop()
        if ws_client:
            await ws_client.disconnect()
        await recorder.stop()
        await rest_client.close()
        store.close()
        log.info("market_data.stopped")


if __name__ == "__main__":
    asyncio.run(main())
