#!/usr/bin/env python3
"""Worker: Sportsbook + ESPN data collector.

Polls the Odds API for sportsbook lines and ESPN for injuries/ratings
on a configurable interval. Writes to ``collector.duckdb``.

No Kalshi connection required.

Usage::

    python scripts/worker_collector.py --config config/default.yaml --overlay config/paper-soak.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.schemas import COLLECTOR_TABLES
from moneygone.data.sports.odds import OddsAPIFeed
from moneygone.data.sports.stats import PlayerStatsFeed
from moneygone.data.store import DataStore
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.collector")


async def _league_has_games_within_window(
    stats_feed: PlayerStatsFeed,
    league: str,
    lookahead_hours: int,
    *,
    reference_time: datetime | None = None,
) -> bool:
    """Check if there are games within the lookahead window via ESPN scoreboard."""
    try:
        from moneygone.data.sports.stats import _SPORT_MAP
        sport = _SPORT_MAP.get(league.lower(), {}).get("sport", "")
        espn_league = _SPORT_MAP.get(league.lower(), {}).get("league", "")
        if not sport or not espn_league:
            return True  # Default to fetching if we can't check

        now = reference_time or datetime.now(timezone.utc)
        from datetime import timedelta
        start = now - timedelta(hours=2)
        end = now + timedelta(hours=lookahead_hours)

        client = await stats_feed._get_client()
        for day_offset in range(max(1, (lookahead_hours // 24) + 2)):
            day = now + timedelta(days=day_offset)
            url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{espn_league}/scoreboard"
            r = await client.get(url, params={"dates": day.strftime("%Y%m%d")})
            if r.status_code != 200:
                continue
            for event in r.json().get("events", []):
                date_str = event.get("date", "")
                if not date_str:
                    continue
                try:
                    event_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if start <= event_time <= end:
                    return True
        return False
    except Exception:
        log.debug("league_check.failed", league=league, exc_info=True)
        return True  # Default to fetching on failure


async def collector_loop(config, store: DataStore, data_dir: Path) -> None:
    """Main collection loop — fetches sportsbook lines on interval."""
    interval = max(1, config.sportsbook.fetch_interval_minutes) * 60
    leagues = [league.lower() for league in config.sportsbook.leagues]
    odds_feed = OddsAPIFeed()
    stats_feed = PlayerStatsFeed()

    if not leagues:
        log.info("collector.disabled_no_leagues")
        return

    if not odds_feed.has_api_key:
        log.warning("collector.no_api_key")
        return

    log.info(
        "collector.started",
        interval_minutes=config.sportsbook.fetch_interval_minutes,
        leagues=leagues,
        bookmakers=config.sportsbook.bookmakers,
    )

    try:
        while True:
            try:
                remaining = odds_feed.requests_remaining
                reserve = config.sportsbook.min_requests_remaining
                if remaining is not None and remaining <= reserve:
                    log.warning("collector.quota_reserve", remaining=remaining, reserve=reserve)
                    await asyncio.sleep(interval)
                    continue

                now = datetime.now(timezone.utc)
                for league in leagues:
                    remaining = odds_feed.requests_remaining
                    if remaining is not None and remaining <= reserve:
                        log.warning("collector.stop_for_quota", league=league, remaining=remaining)
                        break

                    has_games = await _league_has_games_within_window(
                        stats_feed, league, config.sportsbook.lookahead_hours, reference_time=now,
                    )
                    if not has_games:
                        log.debug("collector.skip_no_games", league=league)
                        continue

                    games = await odds_feed.get_upcoming_games(
                        league,
                        bookmakers=list(config.sportsbook.bookmakers),
                        markets=list(config.sportsbook.markets),
                    )
                    rows = odds_feed.build_line_history_rows(league, games, captured_at=now)
                    store.insert_sportsbook_game_lines(rows)
                    log.info(
                        "collector.recorded",
                        league=league,
                        games=len(games),
                        rows=len(rows),
                        remaining=odds_feed.requests_remaining,
                    )

                # Export sportsbook data to parquet for cross-process access
                try:
                    parquet_path = data_dir / "sportsbook_lines.parquet"
                    store.export_table_to_parquet("sportsbook_game_lines", parquet_path)
                    log.debug("collector.parquet_exported", path=str(parquet_path))
                except Exception:
                    log.debug("collector.parquet_export_failed", exc_info=True)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("collector.error")

            await asyncio.sleep(interval)
    finally:
        await odds_feed.close()
        await stats_feed.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sportsbook data collector worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    args = parser.parse_args()

    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = DataStore(data_dir / "collector.duckdb")
    store.initialize_schema(COLLECTOR_TABLES)

    shutdown = asyncio.Event()

    def _signal_handler() -> None:
        log.info("collector.shutdown_signal")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    task = asyncio.create_task(collector_loop(config, store, data_dir))

    try:
        await shutdown.wait()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        store.close()
        log.info("collector.stopped")


if __name__ == "__main__":
    asyncio.run(main())
