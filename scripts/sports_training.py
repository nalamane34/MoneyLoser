#!/usr/bin/env python3
"""Sports model training pipeline.

Builds a historical training set from settled Kalshi sports markets joined
to sportsbook lines, then trains a residual model over Pinnacle.

Commands:
  build-dataset    Fetch settled sports markets, match to sportsbook events,
                   build training data with pre-game features.
  train            Train a LightGBM residual model on the dataset.
  evaluate         Run synthetic evaluation of trained model.
  predict          Make forward predictions for active markets.

Usage::
    python scripts/sports_training.py build-dataset --days 60
    python scripts/sports_training.py train
    python scripts/sports_training.py evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.schemas import (
    CREATE_MARKET_EVENT_MAPPINGS,
    CREATE_SPORTS_OUTCOMES,
)
from moneygone.data.sports.live_snapshots import (
    _normalize_team_name,
    _sport_from_ticker,
    _team_aliases,
    _contains_alias,
    _matches_alias,
    _extract_priced_team,
)
from moneygone.data.sports.power_ratings import ESPNPowerRatings
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market, MarketResult
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("sports_training")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_WINNER_EXCLUDES = (
    "1h", "2h", "first half", "second half", "first 5", "f5",
    "spread", "total", "teamtotal", "team total", "score over",
    "points", "mention", "announcers say", "player", "prop",
    "strikeout", "home run", "touchdown", "goal scorer",
    "-tie",  # Soccer draw markets — can't price from 2-way moneyline
)

_GAME_WINNER_INCLUDES = (
    "game", "winner", "moneyline", " to win", " beat ", " defeat ", " win?",
    "match", "fight",  # Tennis matches, MMA/UFC fights
)

_SPORT_SERIES_MAP = {
    # US majors
    "nba": ["KXNBAGAME", "KXNBAGAMES"],
    "nhl": ["KXNHLGAME"],
    "mlb": ["KXMLBGAME"],
    "nfl": ["KXNFLGAME"],
    # Soccer
    "soccer_epl": ["KXEPLGAME"],
    "soccer_spain_la_liga": ["KXLALIGAGAME"],
    "soccer_germany_bundesliga": ["KXBUNDESLIGAGAME"],
    "soccer_italy_serie_a": ["KXSERIEAGAME"],
    "soccer_france_ligue_one": ["KXLIGUE1GAME"],
    "soccer_usa_mls": ["KXMLSGAME"],
    "soccer_ucl": ["KXUCLGAME"],
    "soccer_uel": ["KXUELGAME"],
    "soccer_brazil": ["KXBRASILEIROGAME"],
    "soccer_liga_mx": ["KXLIGAMXGAME"],
    "soccer_eredivisie": ["KXEREDIVISIEGAME"],
    "soccer_portugal": ["KXLIGAPORTUGALGAME"],
    "soccer_turkey": ["KXSUPERLIGGAME"],
    # Tennis
    "tennis_atp": ["KXATPMATCH"],
    "tennis_wta": ["KXWTAMATCH"],
    # Other
    "kbo": ["KXKBOGAME"],
    "npb": ["KXNPBGAME"],
    "cricket_ipl": ["KXIPLGAME"],
    "ahl": ["KXAHLGAME"],
    "afl": ["KXAFLGAME"],
    "rugby_nrl": ["KXRUGBYNRLMATCH"],
    # Esports
    "esports_cs2": ["KXCS2GAME"],
    "esports_lol": ["KXLOLGAME"],
    "esports_valorant": ["KXVALORANTGAME"],
    "esports_dota2": ["KXDOTA2GAME"],
    "esports_overwatch": ["KXOWGAME"],
    "esports_r6": ["KXR6GAME"],
    "esports_rl": ["KXRLGAME"],
    # MMA / UFC
    "ufc": ["KXUFCFIGHT"],
    # Boxing
    "boxing": ["KXBOXING"],
}


def _is_game_winner_market(market: Market) -> bool:
    """Check if a market is a game-winner/moneyline market."""
    text = " ".join(
        (v or "").lower()
        for v in [
            market.ticker, market.event_ticker, market.series_ticker,
            market.title, market.subtitle, market.yes_sub_title,
        ]
    )
    if any(ex in text for ex in _GAME_WINNER_EXCLUDES):
        return False
    return any(inc in text for inc in _GAME_WINNER_INCLUDES)


def _odds_to_prob(price: float) -> float:
    """Convert decimal odds to implied probability."""
    if price <= 1.0:
        return 1.0
    return 1.0 / price


# Historical draw rates by league — FALLBACK only when no draw_price available.
# Pinnacle/SBR provide actual 3-way odds (1/X/2) for soccer; use those instead.
_DRAW_RATES_FALLBACK: dict[str, float] = {
    "soccer_epl": 0.25,
    "soccer_spain_la_liga": 0.24,
    "soccer_germany_bundesliga": 0.23,
    "soccer_italy_serie_a": 0.25,
    "soccer_france_ligue_one": 0.24,
    "soccer_usa_mls": 0.22,
    "soccer_ucl": 0.22,
    "soccer_uel": 0.23,
    "soccer_brazil": 0.24,
    "soccer_liga_mx": 0.24,
    "soccer_eredivisie": 0.23,
    "soccer_portugal": 0.24,
    "soccer_turkey": 0.24,
    "soccer_scotland": 0.22,
    "soccer_wc": 0.22,
    "soccer_concacaf": 0.22,
}
_DEFAULT_DRAW_RATE_FALLBACK = 0.24


def _normalize_away_prob(
    home_price: float, away_price: float, sport: str = "",
    draw_price: float | None = None,
) -> float:
    """Get the away team's win probability (soccer 3-way aware).

    If draw_price is available (from Pinnacle/SBR 3-way line), use it
    for exact vig-adjusted probability. Otherwise fall back to league avg.
    """
    raw_home = _odds_to_prob(home_price)
    raw_away = _odds_to_prob(away_price)

    if draw_price is not None and draw_price > 1.0:
        # 3-way: use actual draw odds for exact probability
        raw_draw = _odds_to_prob(draw_price)
        total = raw_home + raw_away + raw_draw
        if total < 0.01:
            return 0.5
        return raw_away / total

    total = raw_home + raw_away
    if total < 0.01:
        return 0.5
    normalized = raw_away / total
    if sport.startswith("soccer"):
        draw_rate = _DRAW_RATES_FALLBACK.get(sport, _DEFAULT_DRAW_RATE_FALLBACK)
        return normalized * (1.0 - draw_rate)
    return normalized


def _normalize_home_prob(
    home_price: float, away_price: float, sport: str = "",
    draw_price: float | None = None,
) -> float:
    """Normalize odds to get implied home win probability.

    For soccer with draw_price: use exact 3-way vig removal.
      P(home) = raw_home / (raw_home + raw_draw + raw_away)

    For soccer without draw_price: fallback to league-average draw rate.
      P(home) = (raw_home / (raw_home + raw_away)) × (1 - draw_rate)

    For non-soccer: normalize home+away to sum=1.0.
    """
    raw_home = _odds_to_prob(home_price)
    raw_away = _odds_to_prob(away_price)

    if draw_price is not None and draw_price > 1.0:
        # 3-way: use actual draw odds for exact probability
        raw_draw = _odds_to_prob(draw_price)
        total = raw_home + raw_away + raw_draw
        if total < 0.01:
            return 0.5
        return raw_home / total

    total = raw_home + raw_away
    if total < 0.01:
        return 0.5

    normalized = raw_home / total

    # Soccer without draw price: fallback to historical rates
    if sport.startswith("soccer"):
        draw_rate = _DRAW_RATES_FALLBACK.get(sport, _DEFAULT_DRAW_RATE_FALLBACK)
        return normalized * (1.0 - draw_rate)

    return normalized


# ---------------------------------------------------------------------------
# Step 1: Build mapping + training dataset
# ---------------------------------------------------------------------------

@dataclass
class TrainingRow:
    """One training example: a settled game-winner market with sportsbook features."""
    kalshi_ticker: str
    event_id: str
    sport: str
    home_team: str
    away_team: str
    priced_team: str
    is_home_team: bool
    market_result: str  # "yes" or "no"
    # Pinnacle lines
    pinnacle_home_price: float | None = None
    pinnacle_away_price: float | None = None
    pinnacle_draw_price: float | None = None  # 3-way (soccer)
    pinnacle_home_prob: float | None = None
    # Consensus (all books averaged)
    consensus_home_price: float | None = None
    consensus_away_price: float | None = None
    consensus_draw_price: float | None = None  # 3-way (soccer)
    consensus_home_prob: float | None = None
    # Opening lines
    opening_home_price: float | None = None
    opening_away_price: float | None = None
    # Spreads / totals
    spread_home: float | None = None
    total: float | None = None
    # Power ratings
    home_rating: float | None = None
    away_rating: float | None = None
    # Kalshi pricing
    kalshi_last_price: float | None = None
    kalshi_volume: int | None = None
    close_time: str | None = None
    commence_time: str | None = None
    # Data quality
    line_age_hours: float | None = None
    is_consensus_fallback: bool = False
    match_score: int = 0


async def build_dataset(
    days: int = 60,
    leagues: list[str] | None = None,
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
) -> list[TrainingRow]:
    """Build training dataset from settled Kalshi sports markets + sportsbook lines.

    1. Fetch settled game-winner markets from Kalshi API
    2. For each, find matching sportsbook event in DuckDB
    3. Extract pre-game Pinnacle/consensus lines, spreads, ratings
    4. Record outcome (yes/no) with all features
    """
    if leagues is None:
        leagues = ["nba", "nhl", "mlb"]

    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )
    client = KalshiRestClient(config.exchange)

    # Set up DuckDB for sportsbook line lookups
    data_dir = Path(config.data_dir)
    collector_db = data_dir / "collector.duckdb"
    store = DataStore(data_dir / "training.duckdb")
    store.initialize_schema([CREATE_MARKET_EVENT_MAPPINGS, CREATE_SPORTS_OUTCOMES])
    has_sportsbook_data = False

    if collector_db.exists():
        try:
            store.attach_readonly("collector", collector_db)
            store.create_attached_views({"collector": ["sportsbook_game_lines"]})
            has_sportsbook_data = True
            log.info("training.sportsbook_attached", path=str(collector_db))
        except Exception:
            log.warning("training.sportsbook_attach_failed", exc_info=True)

    # Check if sportsbook_game_lines table already has data (e.g., from SBR backfill)
    if not has_sportsbook_data:
        try:
            from moneygone.data.schemas import CREATE_SPORTSBOOK_GAME_LINES
            store.initialize_schema([CREATE_SPORTSBOOK_GAME_LINES])
            existing = store.query("SELECT COUNT(*) FROM sportsbook_game_lines")
            existing_count = existing[0][0] if existing else 0
            if existing_count > 0:
                has_sportsbook_data = True
                log.info("training.sportsbook_existing", rows=existing_count)
        except Exception:
            pass

    # Also check for parquet (only if no data yet — avoids wiping SBR backfill)
    sportsbook_parquet = data_dir / "sportsbook_lines.parquet"
    if not has_sportsbook_data and sportsbook_parquet.exists():
        try:
            from moneygone.data.schemas import CREATE_SPORTSBOOK_GAME_LINES
            store.initialize_schema([CREATE_SPORTSBOOK_GAME_LINES])
            count = store.load_parquet_into_table("sportsbook_game_lines", sportsbook_parquet)
            has_sportsbook_data = True
            log.info("training.sportsbook_loaded_parquet", rows=count)
        except Exception:
            log.warning("training.parquet_load_failed", exc_info=True)

    # Fetch settled game-winner markets
    cutoff = datetime.now(timezone.utc) - timedelta(days=days + 1)
    settled_markets: list[Market] = []
    seen_tickers: set[str] = set()

    for league in leagues:
        prefixes = _SPORT_SERIES_MAP.get(league, [])
        for prefix in prefixes:
            try:
                batch = await client.get_all_markets(
                    series_ticker=prefix,
                    status="settled",
                    max_pages=20,
                )
            except Exception:
                log.warning("training.fetch_failed", prefix=prefix, exc_info=True)
                continue
            for m in batch:
                if m.close_time and m.close_time < cutoff:
                    continue
                if m.result in (MarketResult.NOT_SETTLED, MarketResult.VOIDED):
                    continue
                if not _is_game_winner_market(m):
                    continue
                if m.ticker not in seen_tickers:
                    seen_tickers.add(m.ticker)
                    settled_markets.append(m)

    log.info("training.settled_markets", count=len(settled_markets))

    # Power ratings (for enrichment)
    ratings_provider = ESPNPowerRatings()
    all_ratings: dict[str, dict] = {}
    for league in leagues:
        try:
            all_ratings[league] = await ratings_provider.get_ratings(league)
        except Exception:
            log.warning("training.ratings_failed", league=league)

    rows: list[TrainingRow] = []

    for market in settled_markets:
        sport = _sport_from_ticker(market.ticker)
        if not sport:
            continue

        # Extract priced team
        priced_team = _extract_priced_team(
            market.title, yes_sub_title=market.yes_sub_title,
        )
        if not priced_team:
            continue

        actual = "yes" if market.result == MarketResult.YES else "no"

        # Try to match to sportsbook event
        matched_event = None
        match_score = 0
        if has_sportsbook_data and store is not None:
            matched_event, match_score = await _match_to_sportsbook(
                market, sport, store,
            )

        if matched_event is not None:
            home_team = matched_event["home_team"]
            away_team = matched_event["away_team"]
            event_id = matched_event.get("event_id", "")

            # Determine orientation: is priced team = home team?
            priced_norm = _normalize_team_name(priced_team)
            home_aliases = _team_aliases(home_team)
            away_aliases = _team_aliases(away_team)
            is_home_team = _matches_alias(priced_norm, home_aliases)

            # Get Pinnacle and consensus lines
            pinnacle_row = await _get_book_line(
                store, event_id, "pinnacle", market.close_time,
            )
            # Fallback: use sharpest available book if no Pinnacle
            if pinnacle_row is None:
                for sharp_book in ["draftkings", "fanduel", "bet365"]:
                    pinnacle_row = await _get_book_line(
                        store, event_id, sharp_book, market.close_time,
                    )
                    if pinnacle_row is not None:
                        break
            consensus_rows = await _get_all_lines(
                store, event_id, market.close_time,
            )

            pin_home = pinnacle_row["home_price"] if pinnacle_row else None
            pin_away = pinnacle_row["away_price"] if pinnacle_row else None
            pin_draw = pinnacle_row.get("draw_price") if pinnacle_row else None
            pin_prob = _normalize_home_prob(pin_home, pin_away, sport, pin_draw) if pin_home and pin_away else None

            # Consensus = average across books
            cons_home = None
            cons_away = None
            cons_draw = None
            cons_prob = None
            if consensus_rows:
                cons_home = sum(r["home_price"] for r in consensus_rows) / len(consensus_rows)
                cons_away = sum(r["away_price"] for r in consensus_rows) / len(consensus_rows)
                # Average draw prices from books that have them
                draw_prices = [r["draw_price"] for r in consensus_rows if r.get("draw_price")]
                cons_draw = sum(draw_prices) / len(draw_prices) if draw_prices else None
                cons_prob = _normalize_home_prob(cons_home, cons_away, sport, cons_draw)

            # Opening lines (earliest captured for this event)
            opening = await _get_opening_line(store, event_id)
            open_home = opening["home_price"] if opening else None
            open_away = opening["away_price"] if opening else None

            # Line age
            line_age = None
            if pinnacle_row and pinnacle_row.get("captured_at"):
                try:
                    cap_time = pinnacle_row["captured_at"]
                    if isinstance(cap_time, str):
                        cap_time = datetime.fromisoformat(cap_time.replace("Z", "+00:00"))
                    if market.close_time:
                        line_age = (market.close_time - cap_time).total_seconds() / 3600
                except Exception:
                    pass

            # Power ratings
            league_ratings = all_ratings.get(sport, {})
            home_rating_obj = ratings_provider.lookup(home_team, league_ratings)
            away_rating_obj = ratings_provider.lookup(away_team, league_ratings)

            row = TrainingRow(
                kalshi_ticker=market.ticker,
                event_id=event_id,
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                priced_team=priced_team,
                is_home_team=is_home_team,
                market_result=actual,
                pinnacle_home_price=pin_home,
                pinnacle_away_price=pin_away,
                pinnacle_draw_price=pin_draw,
                pinnacle_home_prob=pin_prob,
                consensus_home_price=cons_home,
                consensus_away_price=cons_away,
                consensus_draw_price=cons_draw,
                consensus_home_prob=cons_prob,
                opening_home_price=open_home,
                opening_away_price=open_away,
                spread_home=pinnacle_row.get("spread_home") if pinnacle_row else None,
                total=pinnacle_row.get("total") if pinnacle_row else None,
                home_rating=home_rating_obj.rating if home_rating_obj else None,
                away_rating=away_rating_obj.rating if away_rating_obj else None,
                kalshi_last_price=float(market.last_price) if market.last_price else None,
                kalshi_volume=market.volume,
                close_time=market.close_time.isoformat() if market.close_time else None,
                commence_time=str(matched_event.get("commence_time", "")),
                line_age_hours=line_age,
                is_consensus_fallback=pinnacle_row is None,
                match_score=match_score,
            )
        else:
            # No sportsbook match — still record for analysis but mark as unmatched
            row = TrainingRow(
                kalshi_ticker=market.ticker,
                event_id="",
                sport=sport,
                home_team="",
                away_team="",
                priced_team=priced_team,
                is_home_team=True,
                market_result=actual,
                kalshi_last_price=float(market.last_price) if market.last_price else None,
                kalshi_volume=market.volume,
                close_time=market.close_time.isoformat() if market.close_time else None,
                match_score=0,
            )

        rows.append(row)

    await client.close()
    await ratings_provider.close()
    if store:
        store.close()

    log.info(
        "training.dataset_built",
        total=len(rows),
        matched=sum(1 for r in rows if r.event_id),
        with_pinnacle=sum(1 for r in rows if r.pinnacle_home_prob is not None),
    )
    return rows


async def _match_to_sportsbook(
    market: Market,
    sport: str,
    store: DataStore,
) -> tuple[dict | None, int]:
    """Match a Kalshi market to a sportsbook event using team name matching."""
    market_text = _normalize_team_name(
        " ".join(
            v for v in [market.title, market.subtitle, market.yes_sub_title, market.no_sub_title]
            if v
        )
    )
    priced_team_norm = _normalize_team_name(
        _extract_priced_team(market.title, yes_sub_title=market.yes_sub_title)
    )

    # Get sportsbook events near this market's close time
    window_start = market.close_time - timedelta(hours=12)
    window_end = market.close_time + timedelta(hours=6)

    try:
        rows = store.query(
            """
            SELECT DISTINCT event_id, home_team, away_team, commence_time
            FROM sportsbook_game_lines
            WHERE sport = $sport
              AND commence_time >= $start
              AND commence_time <= $end
            """,
            {"sport": sport, "start": window_start, "end": window_end},
        )
    except Exception:
        return None, 0

    if not rows:
        return None, 0

    best_row = None
    best_score = -1

    for row in rows:
        home_team = str(row[1]) if isinstance(row, (list, tuple)) else str(getattr(row, "home_team", ""))
        away_team = str(row[2]) if isinstance(row, (list, tuple)) else str(getattr(row, "away_team", ""))
        event_id = str(row[0]) if isinstance(row, (list, tuple)) else str(getattr(row, "event_id", ""))
        commence_time = row[3] if isinstance(row, (list, tuple)) else getattr(row, "commence_time", None)

        home_aliases = _team_aliases(home_team)
        away_aliases = _team_aliases(away_team)
        title_has_home = _contains_alias(market_text, home_aliases)
        title_has_away = _contains_alias(market_text, away_aliases)
        priced_matches_home = _matches_alias(priced_team_norm, home_aliases)
        priced_matches_away = _matches_alias(priced_team_norm, away_aliases)

        score = 0
        if title_has_home and title_has_away and (priced_matches_home or priced_matches_away):
            score = 4
        elif title_has_home and title_has_away:
            score = 3
        elif priced_matches_home or priced_matches_away:
            score = 2
        elif title_has_home or title_has_away:
            score = 1

        if score > best_score:
            best_score = score
            best_row = {
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
            }

    if best_score < 3:
        return None, best_score
    return best_row, best_score


async def _match_active_to_sportsbook(
    market: Market,
    sport: str,
    store: DataStore,
    now: datetime,
) -> tuple[dict | None, int]:
    """Match an active Kalshi market to a sportsbook event.

    Unlike _match_to_sportsbook (which uses close_time for settled markets),
    this searches for upcoming events within the next 48 hours by team name.
    """
    market_text = _normalize_team_name(
        " ".join(
            v for v in [market.title, market.subtitle, market.yes_sub_title, market.no_sub_title]
            if v
        )
    )
    priced_team_norm = _normalize_team_name(
        _extract_priced_team(market.title, yes_sub_title=market.yes_sub_title) or ""
    )

    # Search for upcoming sportsbook events (next 48 hours)
    window_start = now - timedelta(hours=6)
    window_end = now + timedelta(hours=48)

    try:
        rows = store.query(
            """
            SELECT DISTINCT event_id, home_team, away_team, commence_time
            FROM sportsbook_game_lines
            WHERE sport = $sport
              AND commence_time >= $start
              AND commence_time <= $end
            """,
            {"sport": sport, "start": window_start, "end": window_end},
        )
    except Exception:
        return None, 0

    if not rows:
        return None, 0

    best_row = None
    best_score = -1

    for row in rows:
        home_team = str(row[1])
        away_team = str(row[2])
        event_id = str(row[0])
        commence_time = row[3]

        home_aliases = _team_aliases(home_team)
        away_aliases = _team_aliases(away_team)
        title_has_home = _contains_alias(market_text, home_aliases)
        title_has_away = _contains_alias(market_text, away_aliases)
        priced_matches_home = _matches_alias(priced_team_norm, home_aliases)
        priced_matches_away = _matches_alias(priced_team_norm, away_aliases)

        score = 0
        if title_has_home and title_has_away and (priced_matches_home or priced_matches_away):
            score = 4
        elif title_has_home and title_has_away:
            score = 3
        elif priced_matches_home or priced_matches_away:
            score = 2
        elif title_has_home or title_has_away:
            score = 1

        if score > best_score:
            best_score = score
            best_row = {
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
            }

    if best_score < 3:
        return None, best_score
    return best_row, best_score


async def _get_book_line(
    store: DataStore, event_id: str, bookmaker: str, before: datetime,
) -> dict | None:
    """Get the latest line from a specific bookmaker before a timestamp."""
    try:
        rows = store.query(
            """
            SELECT home_price, away_price, draw_price, spread_home, total, captured_at
            FROM sportsbook_game_lines
            WHERE event_id = $eid AND bookmaker = $book AND captured_at <= $before
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            {"eid": event_id, "book": bookmaker, "before": before},
        )
    except Exception:
        return None
    if not rows:
        return None
    r = rows[0]
    if isinstance(r, (list, tuple)):
        return {"home_price": r[0], "away_price": r[1], "draw_price": r[2], "spread_home": r[3], "total": r[4], "captured_at": r[5]}
    return {"home_price": r.home_price, "away_price": r.away_price, "draw_price": getattr(r, "draw_price", None), "spread_home": r.spread_home, "total": r.total, "captured_at": r.captured_at}


async def _get_all_lines(
    store: DataStore, event_id: str, before: datetime,
) -> list[dict]:
    """Get latest line from each bookmaker for an event."""
    try:
        rows = store.query(
            """
            SELECT bookmaker, home_price, away_price, draw_price
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY bookmaker ORDER BY captured_at DESC) AS rn
                FROM sportsbook_game_lines
                WHERE event_id = $eid AND captured_at <= $before
            )
            WHERE rn = 1
            """,
            {"eid": event_id, "before": before},
        )
    except Exception:
        return []
    result = []
    for r in rows:
        if isinstance(r, (list, tuple)):
            result.append({"bookmaker": r[0], "home_price": r[1], "away_price": r[2], "draw_price": r[3]})
        else:
            result.append({"bookmaker": r.bookmaker, "home_price": r.home_price, "away_price": r.away_price, "draw_price": getattr(r, "draw_price", None)})
    return result


async def _get_opening_line(store: DataStore, event_id: str) -> dict | None:
    """Get the earliest captured line for an event (opening line)."""
    try:
        rows = store.query(
            """
            SELECT home_price, away_price
            FROM sportsbook_game_lines
            WHERE event_id = $eid
            ORDER BY captured_at ASC
            LIMIT 1
            """,
            {"eid": event_id},
        )
    except Exception:
        return None
    if not rows:
        return None
    r = rows[0]
    if isinstance(r, (list, tuple)):
        return {"home_price": r[0], "away_price": r[1]}
    return {"home_price": r.home_price, "away_price": r.away_price}


# ---------------------------------------------------------------------------
# Step 2: Feature engineering for training
# ---------------------------------------------------------------------------

def extract_features(row: TrainingRow) -> dict[str, float]:
    """Convert a TrainingRow into model features.

    Features are oriented to the PRICED TEAM's perspective (the team
    that YES corresponds to on Kalshi).
    """
    features: dict[str, float] = {}

    # Pinnacle anchor (the most important feature)
    # For soccer with 3-way draw odds: use exact probabilities
    # For soccer without draw odds: fall back to league-average draw rate
    is_soccer = row.sport.startswith("soccer")
    if row.pinnacle_home_prob is not None:
        if row.is_home_team:
            pin_prob = row.pinnacle_home_prob
        elif is_soccer and row.pinnacle_home_price and row.pinnacle_away_price:
            pin_prob = _normalize_away_prob(
                row.pinnacle_home_price, row.pinnacle_away_price, row.sport,
                row.pinnacle_draw_price,
            )
        else:
            pin_prob = 1.0 - row.pinnacle_home_prob
        features["pinnacle_win_prob"] = pin_prob
    elif row.consensus_home_prob is not None:
        if row.is_home_team:
            pin_prob = row.consensus_home_prob
        elif is_soccer and row.consensus_home_price and row.consensus_away_price:
            pin_prob = _normalize_away_prob(
                row.consensus_home_price, row.consensus_away_price, row.sport,
                row.consensus_draw_price,
            )
        else:
            pin_prob = 1.0 - row.consensus_home_prob
        features["pinnacle_win_prob"] = pin_prob
        features["is_consensus_fallback"] = 1.0
    else:
        return {}  # Can't do anything without sportsbook pricing

    # Line movement (use draw prices if available for soccer accuracy)
    if row.opening_home_price and row.opening_away_price and row.pinnacle_home_price and row.pinnacle_away_price:
        open_prob = _normalize_home_prob(row.opening_home_price, row.opening_away_price)
        current_prob = _normalize_home_prob(
            row.pinnacle_home_price, row.pinnacle_away_price, row.sport,
            row.pinnacle_draw_price,
        )
        movement = current_prob - open_prob
        if not row.is_home_team:
            movement = -movement
        features["moneyline_movement"] = movement

    # NOTE: kalshi_vs_sportsbook_edge is EXCLUDED from training because
    # kalshi_last_price at settlement converges to 0/1 and leaks the outcome.
    # In live prediction, this feature uses the pre-game Kalshi price instead.

    # Power rating edge
    if row.home_rating is not None and row.away_rating is not None:
        rating_diff = row.home_rating - row.away_rating
        if not row.is_home_team:
            rating_diff = -rating_diff
        features["power_rating_edge"] = rating_diff

    # Home field advantage
    features["home_field_advantage"] = 1.0 if row.is_home_team else -1.0

    # Spread-implied win prob (using normal approximation)
    if row.spread_home is not None:
        from scipy import stats
        sport_sigma = {"nba": 12.0, "nfl": 13.5, "mlb": 4.5, "nhl": 2.5}.get(row.sport, 10.0)
        spread = row.spread_home if row.is_home_team else -row.spread_home
        spread_prob = float(stats.norm.cdf(-spread / sport_sigma))
        features["spread_implied_win_prob"] = spread_prob

    # Data quality features (#9)
    features["line_age_hours"] = min(row.line_age_hours or 0.0, 48.0)
    features["is_consensus_fallback"] = features.get("is_consensus_fallback", 0.0)
    features["match_quality"] = float(row.match_score) / 4.0

    return features


# ---------------------------------------------------------------------------
# Step 3: Train residual model
# ---------------------------------------------------------------------------

def train_residual_model(
    rows: list[TrainingRow],
    model_dir: Path,
) -> dict[str, Any]:
    """Train a LightGBM model to predict the residual over Pinnacle.

    Target: P(YES) - pinnacle_implied_prob
    The model learns systematic biases in Pinnacle pricing that can be
    exploited on Kalshi.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    # Build feature matrix
    X_rows = []
    y_rows = []
    pinnacle_probs = []
    tickers = []

    for row in rows:
        features = extract_features(row)
        if not features or "pinnacle_win_prob" not in features:
            continue

        pin_prob = features["pinnacle_win_prob"]
        actual = 1.0 if row.market_result == "yes" else 0.0
        residual = actual - pin_prob  # What Pinnacle got wrong

        X_rows.append(features)
        y_rows.append(residual)
        pinnacle_probs.append(pin_prob)
        tickers.append(row.kalshi_ticker)

    if len(X_rows) < 50:
        log.warning("training.insufficient_data", count=len(X_rows))
        return {"error": "insufficient_data", "count": len(X_rows)}

    # Feature columns (consistent order)
    feature_cols = sorted(set().union(*(f.keys() for f in X_rows)))
    X = np.array([[f.get(c, 0.0) for c in feature_cols] for f in X_rows])
    y = np.array(y_rows)
    pin = np.array(pinnacle_probs)

    log.info("training.data_prepared", n_samples=len(X), n_features=len(feature_cols))

    # Temporal split (80/20 — preserve time ordering)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    pin_train, pin_test = pin[:split_idx], pin[split_idx:]
    tickers_test = tickers[split_idx:]

    # Further split train into train/val for early stopping
    val_size = max(int(len(X_train) * 0.15), 10)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    # LightGBM training
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Evaluate on test set
    residual_pred = model.predict(X_test)
    adjusted_probs = np.clip(pin_test + residual_pred, 0.01, 0.99)
    actuals = (y_test + pin_test)  # Recover actual outcome (0 or 1)

    # Brier scores
    brier_pinnacle = np.mean((pin_test - actuals) ** 2)
    brier_model = np.mean((adjusted_probs - actuals) ** 2)
    brier_market = None

    # Accuracy
    pin_correct = np.mean((pin_test > 0.5) == (actuals > 0.5))
    model_correct = np.mean((adjusted_probs > 0.5) == (actuals > 0.5))

    # Edge analysis
    edge_calls = []
    for i in range(len(X_test)):
        model_edge = abs(adjusted_probs[i] - pin_test[i])
        if model_edge > 0.03:  # Model disagrees with Pinnacle by 3%+
            model_side = adjusted_probs[i] > 0.5
            actual_side = actuals[i] > 0.5
            edge_calls.append(model_side == actual_side)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance_sorted = sorted(importance.items(), key=lambda x: -x[1])

    results = {
        "n_train": len(X_tr),
        "n_val": val_size,
        "n_test": len(X_test),
        "brier_pinnacle": float(brier_pinnacle),
        "brier_model": float(brier_model),
        "brier_improvement": float(brier_pinnacle - brier_model),
        "accuracy_pinnacle": float(pin_correct),
        "accuracy_model": float(model_correct),
        "edge_calls_correct": sum(edge_calls) if edge_calls else 0,
        "edge_calls_total": len(edge_calls),
        "edge_call_rate": sum(edge_calls) / len(edge_calls) if edge_calls else 0,
        "feature_importance": importance_sorted,
        "feature_cols": feature_cols,
        "best_iteration": model.best_iteration,
    }

    # Print results
    print(f"\n{'='*70}")
    print(f"SPORTS RESIDUAL MODEL TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"  Training:   {len(X_tr)} samples")
    print(f"  Validation: {val_size} samples")
    print(f"  Test:       {len(X_test)} samples")
    print(f"\n  BRIER SCORES (lower = better):")
    print(f"    Pinnacle alone: {brier_pinnacle:.4f}")
    print(f"    Model adjusted: {brier_model:.4f}")
    print(f"    Improvement:    {brier_pinnacle - brier_model:+.4f}")
    print(f"\n  ACCURACY:")
    print(f"    Pinnacle: {pin_correct*100:.1f}%")
    print(f"    Model:    {model_correct*100:.1f}%")
    if edge_calls:
        print(f"\n  EDGE CALLS (model disagrees with Pinnacle by >3%):")
        print(f"    Correct: {sum(edge_calls)}/{len(edge_calls)} ({100*sum(edge_calls)/len(edge_calls):.1f}%)")
    print(f"\n  FEATURE IMPORTANCE:")
    for name, imp in importance_sorted[:10]:
        print(f"    {name:30s} {imp:.1f}")

    # Save model only if improvement is meaningful (>0.005 Brier)
    # Smaller improvements are likely noise on this sample size
    MIN_BRIER_IMPROVEMENT = 0.005
    model_dir.mkdir(parents=True, exist_ok=True)
    improvement = brier_pinnacle - brier_model
    if improvement > MIN_BRIER_IMPROVEMENT:
        model_path = model_dir / "sports_residual.lgb"
        model.save_model(str(model_path))
        meta = {
            "feature_cols": feature_cols,
            "brier_improvement": float(brier_pinnacle - brier_model),
            "n_train": len(X_tr),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(model_dir / "sports_residual_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Model BEATS Pinnacle — saved to {model_path}")
    else:
        print(f"\n  Model does NOT beat Pinnacle — NOT saving")
        print(f"  Will fall back to Pinnacle-only in live trading")

    return results


# ---------------------------------------------------------------------------
# Step 4: Calibrate confidence by league and edge bucket
# ---------------------------------------------------------------------------

def calibrate_confidence(rows: list[TrainingRow]) -> dict[str, dict[str, float]]:
    """Compute empirical accuracy by sport and edge bucket.

    Returns a lookup: sport → edge_bucket → historical_accuracy
    This replaces the "count available features" confidence heuristic.
    """
    buckets: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        features = extract_features(row)
        if "pinnacle_win_prob" not in features:
            continue

        pin_prob = features["pinnacle_win_prob"]
        actual_yes = row.market_result == "yes"
        pin_correct = (pin_prob > 0.5) == actual_yes

        # Edge bucket
        edge = abs(pin_prob - 0.5)
        if edge < 0.05:
            bucket = "toss-up"
        elif edge < 0.15:
            bucket = "lean"
        elif edge < 0.30:
            bucket = "solid"
        else:
            bucket = "blowout"

        buckets[row.sport][bucket].append(pin_correct)
        buckets["_all"][bucket].append(pin_correct)

    # Convert to accuracy rates
    result: dict[str, dict[str, float]] = {}
    for sport, sport_buckets in buckets.items():
        result[sport] = {}
        for bucket, outcomes in sport_buckets.items():
            if outcomes:
                result[sport][bucket] = sum(outcomes) / len(outcomes)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_dataset(rows: list[TrainingRow], path: Path) -> None:
    """Save training dataset to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {k: v for k, v in asdict(r).items()}
        for r in rows
    ]
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  Dataset saved to {path} ({len(records)} rows)")


def load_dataset(path: Path) -> list[TrainingRow]:
    """Load training dataset from JSON."""
    with open(path) as f:
        records = json.load(f)
    rows = []
    for rec in records:
        rows.append(TrainingRow(**{
            k: v for k, v in rec.items()
            if k in TrainingRow.__dataclass_fields__
        }))
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sports model training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ds_parser = sub.add_parser("build-dataset", help="Build training dataset from settled markets")
    ds_parser.add_argument("--days", type=int, default=60, help="Days of history")
    ds_parser.add_argument("--leagues", nargs="+", default=["nba", "nhl", "mlb"])
    ds_parser.add_argument("--config", default="config/default.yaml")
    ds_parser.add_argument("--overlay", default="config/stress-test.yaml")

    tr_parser = sub.add_parser("train", help="Train residual model on dataset")
    tr_parser.add_argument("--dataset", default="data/sports_training.json")
    tr_parser.add_argument("--model-dir", default="models")

    ev_parser = sub.add_parser("evaluate", help="Evaluate current model")
    ev_parser.add_argument("--dataset", default="data/sports_training.json")
    ev_parser.add_argument("--model-dir", default="models")

    pr_parser = sub.add_parser("predict", help="Forward predictions for active markets")
    pr_parser.add_argument("--config", default="config/default.yaml")
    pr_parser.add_argument("--overlay", default="config/stress-test.yaml")

    vr_parser = sub.add_parser("verify", help="Verify forward predictions against settlements")

    args = parser.parse_args()
    setup_logging("INFO")

    if args.command == "build-dataset":
        rows = await build_dataset(
            days=args.days,
            leagues=args.leagues,
            config_base=args.config,
            config_overlay=args.overlay,
        )
        out_path = Path("data/sports_training.json")
        save_dataset(rows, out_path)

        # Print summary
        print(f"\n{'='*70}")
        print(f"DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"  Total markets:      {len(rows)}")
        print(f"  Matched to events:  {sum(1 for r in rows if r.event_id)}")
        print(f"  With Pinnacle data: {sum(1 for r in rows if r.pinnacle_home_prob is not None)}")
        print(f"  YES outcomes:       {sum(1 for r in rows if r.market_result == 'yes')}")
        print(f"  NO outcomes:        {sum(1 for r in rows if r.market_result == 'no')}")
        print(f"\n  BY SPORT:")
        for sport in sorted(set(r.sport for r in rows)):
            sport_rows = [r for r in rows if r.sport == sport]
            matched = sum(1 for r in sport_rows if r.event_id)
            print(f"    {sport:10s} {len(sport_rows):5d} total, {matched:5d} matched")

        # Calibration stats
        conf = calibrate_confidence(rows)
        if conf:
            print(f"\n  PINNACLE ACCURACY BY BUCKET:")
            for sport in sorted(conf.keys()):
                if sport.startswith("_"):
                    continue
                print(f"    {sport}:")
                for bucket, acc in sorted(conf[sport].items()):
                    print(f"      {bucket:12s} {acc*100:.1f}%")

    elif args.command == "train":
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}. Run build-dataset first.")
            return
        rows = load_dataset(dataset_path)
        print(f"Loaded {len(rows)} training rows")
        train_residual_model(rows, Path(args.model_dir))

    elif args.command == "evaluate":
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}.")
            return
        rows = load_dataset(dataset_path)

        # Quick evaluation without training
        features_list = [extract_features(r) for r in rows]
        valid = [(f, r) for f, r in zip(features_list, rows) if "pinnacle_win_prob" in f]
        print(f"\n{len(valid)} rows with Pinnacle data")

        correct = 0
        for f, r in valid:
            pin = f["pinnacle_win_prob"]
            actual_yes = r.market_result == "yes"
            if (pin > 0.5) == actual_yes:
                correct += 1
        print(f"Pinnacle accuracy: {correct}/{len(valid)} ({100*correct/len(valid):.1f}%)")

    elif args.command == "predict":
        await forward_predict(
            config_base=args.config,
            config_overlay=args.overlay,
        )

    elif args.command == "verify":
        await _forward_verify_async()


# ---------------------------------------------------------------------------
# Forward prediction + verification (sports equivalent of weather predict/verify)
# ---------------------------------------------------------------------------

@dataclass
class ForwardPrediction:
    """A forward prediction for an active sports market."""
    kalshi_ticker: str
    sport: str
    priced_team: str
    kalshi_price: float
    sharp_prob: float
    edge: float
    confidence: float
    bookmaker_source: str
    home_team: str
    away_team: str
    prediction_time: str
    close_time: str


async def forward_predict(
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
) -> None:
    """Scan active sports markets, compare to sharp lines, log predictions."""
    all_leagues = list(_SPORT_SERIES_MAP.keys())

    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )
    client = KalshiRestClient(config.exchange)

    # Load sportsbook data
    data_dir = Path(config.data_dir)
    store = DataStore(data_dir / "training.duckdb")
    from moneygone.data.schemas import CREATE_SPORTSBOOK_GAME_LINES
    store.initialize_schema([CREATE_SPORTSBOOK_GAME_LINES])

    # Check existing sportsbook data
    try:
        count = store.query("SELECT COUNT(*) FROM sportsbook_game_lines")[0][0]
        log.info("predict.sportsbook_rows", count=count)
    except Exception:
        count = 0

    if count == 0:
        print("No sportsbook data in training DB. Run sbr_backfill.py first.")
        await client.close()
        store.close()
        return

    # Fetch active game-winner markets
    active_markets = []
    seen = set()
    for league in all_leagues:
        prefixes = _SPORT_SERIES_MAP.get(league, [])
        for prefix in prefixes:
            try:
                batch = await client.get_all_markets(
                    series_ticker=prefix,
                    status="open",
                    max_pages=5,
                )
            except Exception:
                continue
            for m in batch:
                if not _is_game_winner_market(m):
                    continue
                if m.ticker not in seen:
                    seen.add(m.ticker)
                    active_markets.append(m)

    log.info("predict.active_markets", count=len(active_markets))

    predictions: list[ForwardPrediction] = []
    now = datetime.now(timezone.utc)

    for market in active_markets:
        sport = _sport_from_ticker(market.ticker)
        if not sport:
            continue

        priced_team = _extract_priced_team(
            market.title, yes_sub_title=market.yes_sub_title,
        )
        if not priced_team:
            continue

        kalshi_price = float(market.yes_bid or market.last_price or 0.5)
        if kalshi_price <= 0.01 or kalshi_price >= 0.99:
            continue

        # Match to sportsbook using team names + upcoming events
        # For active markets, close_time is often far in the future (market expiry),
        # so we search for upcoming sportsbook events by team name instead
        matched, match_score = await _match_active_to_sportsbook(
            market, sport, store, now,
        )
        if matched is None or match_score < 3:
            continue

        event_id = matched["event_id"]
        home_team = matched["home_team"]
        away_team = matched["away_team"]

        # Get sharp line
        sharp_row = None
        bookmaker_source = "consensus"
        for book in ["pinnacle", "draftkings", "fanduel", "bet365"]:
            sharp_row = await _get_book_line(store, event_id, book, now)
            if sharp_row:
                bookmaker_source = book
                break

        if sharp_row is None:
            consensus = await _get_all_lines(store, event_id, now)
            if not consensus:
                continue
            avg_home = sum(r["home_price"] for r in consensus) / len(consensus)
            avg_away = sum(r["away_price"] for r in consensus) / len(consensus)
            draw_prices = [r["draw_price"] for r in consensus if r.get("draw_price")]
            avg_draw = sum(draw_prices) / len(draw_prices) if draw_prices else None
            sharp_prob_home = _normalize_home_prob(avg_home, avg_away, sport, avg_draw)
        else:
            sharp_draw = sharp_row.get("draw_price")
            avg_home = avg_away = avg_draw = None  # not used in this branch
            sharp_prob_home = _normalize_home_prob(
                sharp_row["home_price"], sharp_row["away_price"], sport, sharp_draw,
            )

        # Orient to priced team
        priced_norm = _normalize_team_name(priced_team)
        home_aliases = _team_aliases(home_team)
        is_home = _matches_alias(priced_norm, home_aliases)

        if sport.startswith("soccer"):
            # For soccer, compute away prob separately (not 1 - home)
            # because draw absorbs probability
            if is_home:
                sharp_prob = sharp_prob_home
            else:
                # Compute away probability with actual draw odds
                if sharp_row is not None:
                    sharp_prob = _normalize_away_prob(
                        sharp_row["home_price"], sharp_row["away_price"], sport,
                        sharp_row.get("draw_price"),
                    )
                else:
                    sharp_prob = _normalize_away_prob(avg_home, avg_away, sport, avg_draw)
        else:
            sharp_prob = sharp_prob_home if is_home else (1.0 - sharp_prob_home)

        edge = sharp_prob - kalshi_price
        confidence = 0.65
        if bookmaker_source == "consensus":
            confidence -= 0.10
        if match_score < 4:
            confidence -= 0.05

        predictions.append(ForwardPrediction(
            kalshi_ticker=market.ticker,
            sport=sport,
            priced_team=priced_team,
            kalshi_price=kalshi_price,
            sharp_prob=sharp_prob,
            edge=edge,
            confidence=confidence,
            bookmaker_source=bookmaker_source,
            home_team=home_team,
            away_team=away_team,
            prediction_time=now.isoformat(),
            close_time=market.close_time.isoformat() if market.close_time else "",
        ))

    await client.close()
    store.close()

    # Save predictions
    pred_dir = Path("data/forward_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / f"sports_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(pred_file, "w") as f:
        json.dump([asdict(p) for p in predictions], f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SPORTS FORWARD PREDICTIONS")
    print(f"{'='*70}")
    print(f"  Active markets scanned: {len(active_markets)}")
    print(f"  Matched to sportsbooks: {len(predictions)}")

    # Sort by absolute edge
    predictions.sort(key=lambda p: abs(p.edge), reverse=True)

    tradeable = [p for p in predictions if abs(p.edge) >= 0.08]
    print(f"  Tradeable (>=8% edge):  {len(tradeable)}")
    print(f"\n  Saved to: {pred_file}")

    if tradeable:
        print(f"\n  TOP EDGE OPPORTUNITIES:")
        for p in tradeable[:20]:
            side = "BUY YES" if p.edge > 0 else "BUY NO"
            print(
                f"    {p.kalshi_ticker:40s} {p.sport:12s} "
                f"sharp={p.sharp_prob:.2f} kalshi={p.kalshi_price:.2f} "
                f"edge={p.edge:+.2f} → {side} ({p.bookmaker_source})"
            )

    if predictions:
        print(f"\n  ALL PREDICTIONS BY SPORT:")
        by_sport = defaultdict(list)
        for p in predictions:
            by_sport[p.sport].append(p)
        for sport in sorted(by_sport):
            preds = by_sport[sport]
            avg_edge = sum(abs(p.edge) for p in preds) / len(preds)
            tradeable_count = sum(1 for p in preds if abs(p.edge) >= 0.08)
            print(f"    {sport:25s} {len(preds):3d} markets, "
                  f"avg |edge|={avg_edge:.3f}, {tradeable_count} tradeable")


def forward_verify() -> None:
    """Check settled outcomes against saved forward predictions (standalone entry point)."""
    asyncio.run(_forward_verify_async())


async def _forward_verify_async() -> None:
    pred_dir = Path("data/forward_predictions")
    if not pred_dir.exists():
        print("No forward predictions found. Run predict first.")
        return

    pred_files = sorted(pred_dir.glob("sports_*.json"))
    if not pred_files:
        print("No sports prediction files found.")
        return

    config = load_config(
        Path("config/default.yaml"),
        Path("config/stress-test.yaml"),
    )
    client = KalshiRestClient(config.exchange)

    # Collect all unique tickers across prediction files
    all_preds: list[dict] = []
    for pf in pred_files:
        with open(pf) as f:
            preds = json.load(f)
        for p in preds:
            p["_source_file"] = pf.name
        all_preds.extend(preds)

    # Deduplicate by ticker (keep latest prediction)
    by_ticker: dict[str, dict] = {}
    for p in all_preds:
        ticker = p.get("kalshi_ticker", "")
        if ticker:
            existing = by_ticker.get(ticker)
            if existing is None or p.get("prediction_time", "") > existing.get("prediction_time", ""):
                by_ticker[ticker] = p

    # Fetch current status for all predicted tickers
    settled: list[dict] = []
    unsettled: list[dict] = []
    errors = 0

    for ticker, pred in by_ticker.items():
        try:
            market = await client.get_market(ticker)
            if market is None:
                errors += 1
                continue
            if market.result in (MarketResult.YES, MarketResult.NO):
                outcome = 1.0 if market.result == MarketResult.YES else 0.0
                pred["_outcome"] = outcome
                pred["_result"] = market.result.value
                settled.append(pred)
            else:
                unsettled.append(pred)
        except Exception:
            errors += 1

    await client.close()

    # Analyze settled predictions
    print(f"\n{'='*70}")
    print(f"SPORTS FORWARD VERIFICATION")
    print(f"{'='*70}")
    print(f"  Prediction files: {len(pred_files)}")
    print(f"  Unique tickers: {len(by_ticker)}")
    print(f"  Settled: {len(settled)}")
    print(f"  Unsettled: {len(unsettled)}")
    print(f"  Errors: {errors}")

    if not settled:
        print("\n  No settled markets yet. Run verify after games complete.")
        return

    # Compute metrics
    brier_sharp = []
    brier_kalshi = []
    edge_correct = 0
    edge_total = 0
    profit_sim = 0.0  # Simulated $1 flat bets on edge signals
    results_by_sport: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "edge_correct": 0, "brier_sharp": [], "brier_kalshi": [], "profit": 0.0
    })

    for pred in settled:
        outcome = pred["_outcome"]
        sharp_prob = pred.get("sharp_prob", 0.5)
        kalshi_price = pred.get("kalshi_price", 0.5)
        edge = pred.get("edge", 0.0)
        sport = pred.get("sport", "unknown")

        # Brier score: lower is better
        brier_s = (sharp_prob - outcome) ** 2
        brier_k = (kalshi_price - outcome) ** 2
        brier_sharp.append(brier_s)
        brier_kalshi.append(brier_k)

        stats = results_by_sport[sport]
        stats["n"] += 1
        stats["brier_sharp"].append(brier_s)
        stats["brier_kalshi"].append(brier_k)

        # Edge accuracy: did we correctly identify mispricing direction?
        if abs(edge) >= 0.08:
            edge_total += 1
            stats_edge_total = stats.get("edge_total", 0) + 1
            stats["edge_total"] = stats_edge_total

            # Edge > 0 means we think YES is underpriced (buy YES)
            # Edge < 0 means we think YES is overpriced (buy NO)
            if edge > 0:
                # We'd buy YES. Profit if outcome = 1
                payout = (1.0 - kalshi_price) if outcome == 1.0 else -kalshi_price
            else:
                # We'd buy NO. Profit if outcome = 0
                payout = (1.0 - (1.0 - kalshi_price)) if outcome == 0.0 else -(1.0 - kalshi_price)

            if (edge > 0 and outcome == 1.0) or (edge < 0 and outcome == 0.0):
                edge_correct += 1
                stats["edge_correct"] += 1

            profit_sim += payout
            stats["profit"] += payout

    # Print results
    avg_brier_sharp = sum(brier_sharp) / len(brier_sharp) if brier_sharp else 0
    avg_brier_kalshi = sum(brier_kalshi) / len(brier_kalshi) if brier_kalshi else 0

    print(f"\n  --- Overall Metrics ---")
    print(f"  Settled predictions: {len(settled)}")
    print(f"  Brier (sharp model): {avg_brier_sharp:.4f}")
    print(f"  Brier (Kalshi):      {avg_brier_kalshi:.4f}")
    print(f"  Brier improvement:   {avg_brier_kalshi - avg_brier_sharp:+.4f} ({'sharp better' if avg_brier_sharp < avg_brier_kalshi else 'Kalshi better'})")
    if edge_total > 0:
        print(f"\n  Edge calls (>=8%): {edge_correct}/{edge_total} correct ({100*edge_correct/edge_total:.0f}%)")
        print(f"  Simulated P&L ($1 flat): ${profit_sim:+.2f}")

    print(f"\n  --- By Sport ---")
    for sport in sorted(results_by_sport.keys()):
        stats = results_by_sport[sport]
        n = stats["n"]
        bs = sum(stats["brier_sharp"]) / n if n else 0
        bk = sum(stats["brier_kalshi"]) / n if n else 0
        et = stats.get("edge_total", 0)
        ec = stats["edge_correct"]
        pnl = stats["profit"]
        edge_pct = f"{100*ec/et:.0f}%" if et > 0 else "N/A"
        print(f"    {sport:<30} n={n:>3}  Brier sharp={bs:.4f}  Kalshi={bk:.4f}  edge={ec}/{et} ({edge_pct})  P&L=${pnl:+.2f}")

    # Save verification results
    verify_dir = Path("data/forward_verifications")
    verify_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    verify_file = verify_dir / f"verify_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(verify_file, "w") as f:
        json.dump({
            "timestamp": now.isoformat(),
            "settled_count": len(settled),
            "unsettled_count": len(unsettled),
            "brier_sharp": avg_brier_sharp,
            "brier_kalshi": avg_brier_kalshi,
            "edge_correct": edge_correct,
            "edge_total": edge_total,
            "profit_sim": profit_sim,
            "by_sport": {
                sport: {
                    "n": s["n"],
                    "brier_sharp": sum(s["brier_sharp"]) / s["n"] if s["n"] else 0,
                    "brier_kalshi": sum(s["brier_kalshi"]) / s["n"] if s["n"] else 0,
                    "edge_correct": s["edge_correct"],
                    "edge_total": s.get("edge_total", 0),
                    "profit": s["profit"],
                }
                for sport, s in results_by_sport.items()
            },
            "settled_details": [
                {
                    "ticker": p["kalshi_ticker"],
                    "sport": p.get("sport"),
                    "sharp_prob": p.get("sharp_prob"),
                    "kalshi_price": p.get("kalshi_price"),
                    "edge": p.get("edge"),
                    "outcome": p["_outcome"],
                    "result": p["_result"],
                }
                for p in settled
            ],
        }, f, indent=2)
    print(f"\n  Verification saved: {verify_file}")


if __name__ == "__main__":
    asyncio.run(main())
