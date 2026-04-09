#!/usr/bin/env python3
"""Data recording CLI.

Connects to the Kalshi WebSocket feed and records market ticks, orderbook
snapshots, and trades to the DuckDB DataStore. Optionally fetches weather,
crypto, and sportsbook data on a schedule.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.sports.odds import OddsAPIFeed
from moneygone.data.sports.stats import PlayerStatsFeed
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)

_SPORTBOOK_LEAGUE_TO_SPORT = {
    "nba": "basketball",
    "ncaab": "basketball",
    "nfl": "football",
    "ncaaf": "football",
    "nhl": "hockey",
    "mlb": "baseball",
}


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


def _scoreboard_event_start(event: dict) -> datetime | None:
    """Extract an ESPN scoreboard event start time."""
    raw = event.get("date")
    if raw is None:
        competitions = event.get("competitions", [])
        if competitions:
            raw = competitions[0].get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


async def _league_has_games_within_window(
    stats_feed: PlayerStatsFeed,
    league: str,
    lookahead_hours: int,
    *,
    reference_time: datetime | None = None,
) -> bool:
    """Use free ESPN scoreboards to decide whether a paid poll is worthwhile."""
    sport = _SPORTBOOK_LEAGUE_TO_SPORT.get(league.lower())
    if sport is None:
        log.warning("sportsbook_fetch.unsupported_league", league=league)
        return False

    start = reference_time or datetime.now(timezone.utc)
    end = start + timedelta(hours=max(0, lookahead_hours))
    day_count = (end.date() - start.date()).days + 1

    for day_offset in range(day_count):
        day = start.date() + timedelta(days=day_offset)
        payload = await stats_feed.get_scoreboard(
            sport,
            league,
            date=day.strftime("%Y%m%d"),
        )
        if not payload:
            continue
        for event in payload.get("events", []):
            start_time = _scoreboard_event_start(event)
            if start_time is None:
                continue
            if start <= start_time <= end:
                return True

    return False


async def _fetch_sportsbook_loop(config, store: DataStore) -> None:
    """Periodically record sharp-book line history with tight quota controls."""
    interval = max(1, config.sportsbook.fetch_interval_minutes) * 60
    leagues = [league.lower() for league in config.sportsbook.leagues]
    odds_feed = OddsAPIFeed()
    stats_feed = PlayerStatsFeed()

    if not leagues:
        log.info("sportsbook_fetch.disabled_no_leagues")
        await odds_feed.close()
        await stats_feed.close()
        return

    if not odds_feed.has_api_key:
        log.warning("sportsbook_fetch.no_api_key")
        await odds_feed.close()
        await stats_feed.close()
        return

    log.info(
        "sportsbook_fetch.started",
        interval_minutes=config.sportsbook.fetch_interval_minutes,
        leagues=leagues,
        bookmakers=config.sportsbook.bookmakers,
        markets=config.sportsbook.markets,
        lookahead_hours=config.sportsbook.lookahead_hours,
        min_requests_remaining=config.sportsbook.min_requests_remaining,
    )

    try:
        while True:
            try:
                remaining = odds_feed.requests_remaining
                reserve = config.sportsbook.min_requests_remaining
                if remaining is not None and remaining <= reserve:
                    log.warning(
                        "sportsbook_fetch.quota_reserve_reached",
                        remaining=remaining,
                        reserve=reserve,
                    )
                    await asyncio.sleep(interval)
                    continue

                now = datetime.now(timezone.utc)
                for league in leagues:
                    remaining = odds_feed.requests_remaining
                    if remaining is not None and remaining <= reserve:
                        log.warning(
                            "sportsbook_fetch.stop_for_quota",
                            league=league,
                            remaining=remaining,
                            reserve=reserve,
                        )
                        break

                    has_nearby_games = await _league_has_games_within_window(
                        stats_feed,
                        league,
                        config.sportsbook.lookahead_hours,
                        reference_time=now,
                    )
                    if not has_nearby_games:
                        log.debug(
                            "sportsbook_fetch.skip_no_games",
                            league=league,
                            lookahead_hours=config.sportsbook.lookahead_hours,
                        )
                        continue

                    games = await odds_feed.get_upcoming_games(
                        league,
                        bookmakers=list(config.sportsbook.bookmakers),
                        markets=list(config.sportsbook.markets),
                    )
                    rows = odds_feed.build_line_history_rows(
                        league,
                        games,
                        captured_at=now,
                    )
                    store.insert_sportsbook_game_lines(rows)
                    log.info(
                        "sportsbook_fetch.recorded",
                        league=league,
                        games=len(games),
                        rows=len(rows),
                        remaining=odds_feed.requests_remaining,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("sportsbook_fetch.error")

            await asyncio.sleep(interval)
    finally:
        await odds_feed.close()
        await stats_feed.close()


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

                        trades, _ = await rest_client.get_trades(ticker, limit=20)
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

    # Optional sportsbook line-history fetching
    if config.sportsbook.enabled:
        tasks.append(
            asyncio.create_task(
                _fetch_sportsbook_loop(config, store),
                name="sportsbook_fetch",
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
