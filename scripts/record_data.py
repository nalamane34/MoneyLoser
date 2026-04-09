#!/usr/bin/env python3
"""Data recording CLI.

Connects to the Kalshi WebSocket feed and records market ticks, orderbook
snapshots, and trades to the DuckDB DataStore.  Optionally fetches weather
and crypto data on a schedule.

Usage::

    python scripts/record_data.py --config config/default.yaml
    python scripts/record_data.py --markets TICKER1,TICKER2
    python scripts/record_data.py --markets all
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import structlog

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)


async def _fetch_weather_loop(config, store: DataStore) -> None:
    """Periodically fetch weather data (placeholder for external provider)."""
    interval = config.weather.fetch_interval_minutes * 60
    log.info("weather_fetch.started", interval_minutes=config.weather.fetch_interval_minutes)

    while True:
        try:
            # Placeholder: actual weather fetching would go here (NOAA, ECMWF)
            log.debug("weather_fetch.tick", locations=len(config.weather.locations))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("weather_fetch.error")
        await asyncio.sleep(interval)


async def _fetch_crypto_loop(config, store: DataStore) -> None:
    """Periodically fetch crypto data via ccxt."""
    interval = config.crypto.fetch_interval_seconds
    log.info(
        "crypto_fetch.started",
        interval_seconds=interval,
        symbols=config.crypto.symbols,
    )

    while True:
        try:
            import ccxt.async_support as ccxt_async

            for exchange_name in config.crypto.exchanges:
                exchange_cls = getattr(ccxt_async, exchange_name, None)
                if exchange_cls is None:
                    log.warning("crypto_fetch.unknown_exchange", name=exchange_name)
                    continue

                exchange = exchange_cls()
                try:
                    for symbol in config.crypto.symbols:
                        try:
                            funding = await exchange.fetch_funding_rate(symbol)
                            if funding:
                                from datetime import datetime, timezone

                                store.insert_funding_rates([{
                                    "exchange": exchange_name,
                                    "symbol": symbol,
                                    "rate": float(funding.get("fundingRate", 0.0)),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }])
                        except Exception:
                            log.debug(
                                "crypto_fetch.symbol_error",
                                exchange=exchange_name,
                                symbol=symbol,
                                exc_info=True,
                            )
                finally:
                    await exchange.close()

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("crypto_fetch.error")
        await asyncio.sleep(interval)


async def _poll_markets_loop(
    rest_client: KalshiRestClient,
    recorder: MarketDataRecorder,
    tickers: list[str] | None,
    poll_interval: float = 5.0,
) -> None:
    """Poll market data via REST and feed it to the recorder.

    If the WebSocket client is not yet implemented, this provides a
    fallback data collection mechanism.
    """
    log.info("market_poll.started", tickers=tickers or "all")

    while True:
        try:
            if tickers:
                for ticker in tickers:
                    try:
                        market = await rest_client.get_market(ticker)
                        await recorder.on_ticker_update({
                            "data": {
                                "ticker": market.ticker,
                                "event_ticker": market.event_ticker,
                                "title": market.title,
                                "status": market.status.value,
                                "yes_bid": float(market.yes_bid),
                                "yes_ask": float(market.yes_ask),
                                "last_price": float(market.last_price),
                                "volume": market.volume,
                                "open_interest": market.open_interest,
                                "close_time": market.close_time.isoformat(),
                                "result": market.result.value if market.result else None,
                                "category": market.category,
                            }
                        })

                        ob = await rest_client.get_orderbook(ticker)
                        await recorder.on_orderbook_update({
                            "data": {
                                "ticker": ob.ticker,
                                "yes_levels": [
                                    [float(lv.price), float(lv.contracts)]
                                    for lv in ob.yes_levels
                                ],
                                "no_levels": [
                                    [float(lv.price), float(lv.contracts)]
                                    for lv in ob.no_levels
                                ],
                                "seq": ob.seq,
                                "snapshot_time": ob.timestamp.isoformat(),
                            }
                        })

                        trades = await rest_client.get_trades(ticker, limit=20)
                        for trade in trades:
                            await recorder.on_trade({
                                "data": {
                                    "trade_id": trade.trade_id,
                                    "ticker": trade.ticker,
                                    "count": trade.count,
                                    "yes_price": float(trade.yes_price),
                                    "taker_side": trade.taker_side.value,
                                    "trade_time": trade.created_time.isoformat(),
                                }
                            })
                    except Exception:
                        log.warning(
                            "market_poll.ticker_error",
                            ticker=ticker,
                            exc_info=True,
                        )
            else:
                # Fetch all open markets
                markets = await rest_client.get_markets(status="open", limit=100)
                for market in markets:
                    await recorder.on_ticker_update({
                        "data": {
                            "ticker": market.ticker,
                            "event_ticker": market.event_ticker,
                            "title": market.title,
                            "status": market.status.value,
                            "yes_bid": float(market.yes_bid),
                            "yes_ask": float(market.yes_ask),
                            "last_price": float(market.last_price),
                            "volume": market.volume,
                            "open_interest": market.open_interest,
                            "close_time": market.close_time.isoformat(),
                            "result": market.result.value if market.result else None,
                            "category": market.category,
                        }
                    })

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market_poll.error")
        await asyncio.sleep(poll_interval)


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the data recorder."""
    config = load_config(
        base_path=Path(args.config),
    )

    setup_logging(config.log_level)

    log.info(
        "record_data.starting",
        config_path=args.config,
        markets=args.markets,
    )

    # Initialize store
    db_path = Path(config.data_dir) / "moneygone.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = DataStore(db_path)
    store.initialize_schema()

    # Initialize recorder
    recorder = MarketDataRecorder(store)
    await recorder.start()

    # Initialize REST client
    rest_client = KalshiRestClient(config.exchange)

    # Parse tickers
    tickers: list[str] | None = None
    if args.markets and args.markets.lower() != "all":
        tickers = [t.strip() for t in args.markets.split(",")]

    # Build task list
    tasks: list[asyncio.Task[None]] = []

    # Market data polling
    tasks.append(
        asyncio.create_task(
            _poll_markets_loop(rest_client, recorder, tickers),
            name="market_poll",
        )
    )

    # Optional weather fetching
    if config.weather.enabled:
        tasks.append(
            asyncio.create_task(
                _fetch_weather_loop(config, store),
                name="weather_fetch",
            )
        )

    # Optional crypto fetching
    if config.crypto.enabled:
        tasks.append(
            asyncio.create_task(
                _fetch_crypto_loop(config, store),
                name="crypto_fetch",
            )
        )

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        log.info("record_data.shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    log.info("record_data.running", task_count=len(tasks))

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Cancel all tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup
    await recorder.stop()
    await rest_client.close()
    store.close()

    log.info("record_data.stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Kalshi market data to DuckDB."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to base config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of market tickers to record, "
            "or 'all' for all open markets (default: all)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
