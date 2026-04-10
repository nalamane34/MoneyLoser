#!/usr/bin/env python3
"""Operational drill for stale sportsbook fail-closed behavior.

Loads the current discovery cache plus sportsbook parquet, then compares
snapshot matching under:

1. the normal freshness window derived from config, and
2. an intentionally impossible freshness window (0 seconds).

Pass condition:
  - fresh matching finds at least one sports market, and
  - stale matching finds zero sports markets.

This exercises the exact ``StoreBackedSportsSnapshotProvider`` used by the
live execution worker without touching the running worker stack.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.market_discovery import MarketCategory, MarketDiscoveryService
from moneygone.data.schemas import EXECUTION_TABLES
from moneygone.data.sports.live_snapshots import (
    StoreBackedSportsSnapshotProvider,
    _sport_from_ticker,
    recommended_max_line_age,
)
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("drill.stale_sports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stale sportsbook fail-closed drill")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/live.yaml")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Maximum number of fresh matched example tickers to print",
    )
    return parser.parse_args()


def _sample_sports_markets(
    cache_path: Path,
    *,
    leagues: list[str],
) -> list[Market]:
    classified, refreshed_at = MarketDiscoveryService.load_cache(cache_path)
    if not classified or refreshed_at is None:
        raise RuntimeError(f"discovery cache unavailable or empty: {cache_path}")

    allowed_leagues = {league.lower() for league in leagues}
    sports = [
        market
        for market, category in classified
        if category == MarketCategory.SPORTS
        and _sport_from_ticker(market.ticker) in allowed_leagues
        and market.volume >= 10
        and market.yes_bid > 0
        and market.yes_ask > 0
    ]
    if not sports:
        raise RuntimeError("no configured-league sports markets found in discovery cache")
    return sports


async def _run_probe(
    *,
    store: DataStore,
    rest_client: KalshiRestClient,
    leagues: list[str],
    markets: list[Market],
    max_line_age: timedelta,
) -> list[Market]:
    provider = StoreBackedSportsSnapshotProvider(
        store,
        leagues=leagues,
        rest_client=rest_client,
        max_line_age=max_line_age,
    )
    try:
        return await provider.refresh(markets)
    finally:
        await provider.close()


async def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / args.config, repo_root / args.overlay)
    setup_logging(config.log_level)

    data_dir = repo_root / config.data_dir
    cache_path = data_dir / "discovered_markets.json"
    parquet_path = data_dir / "sportsbook_lines.parquet"
    if not parquet_path.exists():
        raise RuntimeError(f"sportsbook parquet missing: {parquet_path}")

    fresh_window = recommended_max_line_age(config.sportsbook.fetch_interval_minutes)

    with tempfile.TemporaryDirectory(prefix="moneygone-stale-drill-") as tmpdir:
        store = DataStore(Path(tmpdir) / "probe.duckdb")
        store.initialize_schema(EXECUTION_TABLES)
        loaded = store.load_parquet_into_table("sportsbook_game_lines", parquet_path)
        if loaded <= 0:
            raise RuntimeError(f"no sportsbook rows loaded from {parquet_path}")

        rest_client = KalshiRestClient(config.exchange)
        try:
            sampler = StoreBackedSportsSnapshotProvider(
                store,
                leagues=config.sportsbook.leagues,
                rest_client=rest_client,
                max_line_age=fresh_window,
            )
            try:
                markets = [
                    market
                    for market in _sample_sports_markets(
                        cache_path,
                        leagues=config.sportsbook.leagues,
                    )
                    if sampler._looks_like_game_winner_market(market)
                ]
            finally:
                await sampler.close()

            if not markets:
                raise RuntimeError("no game-winner sportsbook markets found in sampled universe")

            fresh = await _run_probe(
                store=store,
                rest_client=rest_client,
                leagues=config.sportsbook.leagues,
                markets=markets,
                max_line_age=fresh_window,
            )
            stale = await _run_probe(
                store=store,
                rest_client=rest_client,
                leagues=config.sportsbook.leagues,
                markets=markets,
                max_line_age=timedelta(seconds=0),
            )
        finally:
            await rest_client.close()
            store.close()

    print(
        "fresh_matched="
        f"{len(fresh)} stale_matched={len(stale)} probed_markets={len(markets)} "
        f"fresh_window_minutes={fresh_window.total_seconds() / 60:.0f}",
        flush=True,
    )
    if fresh:
        print(
            "fresh_examples=" + ",".join(m.ticker for m in fresh[: max(1, args.sample_size)]),
            flush=True,
        )

    if not fresh:
        print("FAIL: no fresh sports matches found", flush=True)
        return 1
    if stale:
        print(
            "FAIL: stale lines still matched tickers="
            + ",".join(m.ticker for m in stale[:10]),
            flush=True,
        )
        return 1

    print("PASS: stale sportsbook lines fail closed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
