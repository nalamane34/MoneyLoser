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
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.schemas import COLLECTOR_TABLES
from moneygone.data.sports.odds import OddsAPIFeed, TENNIS_TOURNAMENT_KEYS
from moneygone.data.sports.esports_odds import EsportsOddsFeed
from moneygone.data.sports.sportsgameodds import SportsGameOddsFeed
from moneygone.data.sports.stats import PlayerStatsFeed
from moneygone.data.store import DataStore
from moneygone.utils.env import load_repo_env
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("worker.collector")
REPO_ROOT = Path(__file__).resolve().parent.parent


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


async def collect_once(
    config,
    store: DataStore,
    data_dir: Path,
    *,
    odds_feed: OddsAPIFeed,
    sgo_feed: SportsGameOddsFeed | None = None,
    stats_feed: PlayerStatsFeed,
    reference_time: datetime | None = None,
) -> None:
    """Run a single sportsbook collection pass.

    Uses SportsGameOdds as primary feed when available, falling back to
    The Odds API for leagues that SportsGameOdds doesn't cover (e.g.
    soccer, UFC on the free tier).
    """
    leagues = [league.lower() for league in config.sportsbook.leagues]
    reserve = config.sportsbook.min_requests_remaining

    now = reference_time or datetime.now(timezone.utc)

    # Expand tennis shorthands into active tournament keys.
    expanded_leagues: list[str] = []
    active_tennis_keys: list[str] | None = None
    for league in leagues:
        if league in TENNIS_TOURNAMENT_KEYS:
            if active_tennis_keys is None:
                active_tennis_keys = await odds_feed.get_active_tennis_keys()
                log.info("collector.active_tennis", keys=active_tennis_keys)
            prefix = league.replace("tennis_", "tennis_")  # e.g. "tennis_atp" or "tennis_wta"
            matching = [k for k in active_tennis_keys if k.startswith(prefix)]
            if matching:
                expanded_leagues.extend(matching)
            else:
                log.info("collector.no_active_tennis", league=league)
        else:
            expanded_leagues.append(league)

    for league in expanded_leagues:
        # Skip ESPN schedule check for tennis tournaments and MMA — those
        # don't have standard ESPN schedule pages.
        is_special = league.startswith("tennis_") or league in ("ufc", "mma")
        if not is_special:
            try:
                has_games = await asyncio.wait_for(
                    _league_has_games_within_window(
                        stats_feed,
                        league,
                        config.sportsbook.lookahead_hours,
                        reference_time=now,
                    ),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                log.warning("collector.espn_check_timeout", league=league)
                has_games = True

            if not has_games:
                log.info("collector.skip_no_games", league=league)
                continue

        # Try SportsGameOdds first, fall back to Odds API if it fails or
        # returns no data (e.g. league not on free tier).
        games = []
        used_feed = "odds_api"

        if sgo_feed is not None and sgo_feed.has_api_key:
            try:
                games = await sgo_feed.get_upcoming_games(
                    league,
                    bookmakers=list(config.sportsbook.bookmakers),
                    markets=list(config.sportsbook.markets),
                )
                if games:
                    used_feed = "sportsgameodds"
            except Exception:
                log.debug("collector.sgo_failed", league=league, exc_info=True)

        if not games and odds_feed.has_api_key:
            remaining = odds_feed.requests_remaining
            if remaining is not None and remaining <= reserve:
                log.warning("collector.stop_for_quota", league=league, remaining=remaining)
                continue
            games = await odds_feed.get_upcoming_games(
                league,
                bookmakers=list(config.sportsbook.bookmakers),
                markets=list(config.sportsbook.markets),
            )
            used_feed = "odds_api"

        # Map tennis tournament keys back to generic sport for storage.
        storage_league = league
        if league.startswith("tennis_atp_"):
            storage_league = "tennis_atp"
        elif league.startswith("tennis_wta_"):
            storage_league = "tennis_wta"

        feed_obj = sgo_feed if used_feed == "sportsgameodds" and sgo_feed else odds_feed
        rows = feed_obj.build_line_history_rows(storage_league, games, captured_at=now)
        store.insert_sportsbook_game_lines(rows)
        log.info(
            "collector.recorded",
            league=storage_league,
            api_key=league,
            games=len(games),
            rows=len(rows),
            feed=used_feed,
            remaining=odds_feed.requests_remaining if used_feed == "odds_api" else None,
        )

    # Collect esports odds from OddsPapi (if key available).
    # Budget: 250 req/month free tier.  With 3 sports × 4 calls each = 12 per run.
    # Run esports collection only every 6th cycle (every 3 hours with 30-min interval).
    esports_feed = EsportsOddsFeed()
    if esports_feed.has_api_key:
        # Check if we should run this cycle — use a simple file-based counter
        esports_counter_file = data_dir / ".esports_counter"
        run_esports = False
        try:
            if esports_counter_file.exists():
                counter = int(esports_counter_file.read_text().strip())
            else:
                counter = 0
            counter += 1
            esports_counter_file.write_text(str(counter))
            # Run every 6th cycle (3 hours at 30-min interval)
            run_esports = counter % 6 == 1
        except Exception:
            run_esports = True  # fallback: always run

        if run_esports:
            try:
                # Focus on CS2, LoL, Valorant — highest Kalshi volume
                all_esports = await esports_feed.get_all_esports_odds(
                    max_fixtures_per_sport=5,
                    sports=["esports_cs2", "esports_lol", "esports_valorant"],
                )
                for sport, lines in all_esports.items():
                    rows = esports_feed.build_line_history_rows(lines, captured_at=now)
                    store.insert_sportsbook_game_lines(rows)
                    log.info(
                        "collector.esports_recorded",
                        sport=sport,
                        lines=len(lines),
                        rows=len(rows),
                        api_calls=esports_feed.requests_used,
                    )
            except Exception:
                log.warning("collector.esports_error", exc_info=True)
            finally:
                await esports_feed.close()
        else:
            log.debug("collector.esports_skipped_not_this_cycle", counter=counter)
    else:
        log.debug("collector.esports_skipped_no_key")

    try:
        parquet_path = data_dir / "sportsbook_lines.parquet"
        store.export_table_to_parquet("sportsbook_game_lines", parquet_path)
        log.debug("collector.parquet_exported", path=str(parquet_path))
    except Exception:
        log.debug("collector.parquet_export_failed", exc_info=True)


async def collector_loop(config, store: DataStore, data_dir: Path) -> None:
    """Main collection loop — fetches sportsbook lines on interval.

    Uses SportsGameOdds as primary for leagues it supports (NBA, NHL, MLB),
    and falls back to The Odds API for leagues SportsGameOdds blocks on the
    free tier (EPL, La Liga, Bundesliga, Serie A, UFC, tennis).
    """
    interval = max(1, config.sportsbook.fetch_interval_minutes) * 60
    leagues = [league.lower() for league in config.sportsbook.leagues]

    # Initialize both feeds.
    odds_api_feed = OddsAPIFeed()
    sgo_key = (
        config.sportsbook.sportsgameodds_api_key
        or os.environ.get("SPORTSGAMEODDS_API_KEY", "")
    )
    sgo_feed = SportsGameOddsFeed(api_key=sgo_key) if sgo_key else None

    has_sgo = sgo_feed is not None and sgo_feed.has_api_key
    has_odds_api = odds_api_feed.has_api_key

    if has_sgo:
        log.info("collector.sportsgameodds_available")
    if has_odds_api:
        log.info("collector.odds_api_available")

    stats_feed = PlayerStatsFeed()

    if not leagues:
        log.info("collector.disabled_no_leagues")
        return

    if not has_sgo and not has_odds_api:
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
                await collect_once(
                    config,
                    store,
                    data_dir,
                    odds_feed=odds_api_feed,
                    sgo_feed=sgo_feed,
                    stats_feed=stats_feed,
                )

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("collector.error")

            await asyncio.sleep(interval)
    finally:
        await odds_api_feed.close()
        if sgo_feed:
            await sgo_feed.close()
        await stats_feed.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sportsbook data collector worker")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/paper-soak.yaml")
    args = parser.parse_args()

    loaded_env = load_repo_env(REPO_ROOT)
    config = load_config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay),
    )
    setup_logging(config.log_level)
    if loaded_env:
        log.info("worker_collector.repo_env_loaded", keys=sorted(loaded_env))

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
