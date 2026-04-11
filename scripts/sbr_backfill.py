#!/usr/bin/env python3
"""Backfill historical sportsbook odds from SportsBookReview.

SBR embeds structured odds data in a __NEXT_DATA__ JSON blob on each page.
This script fetches historical moneyline odds for NBA/NHL/MLB and inserts
them into our sportsbook_game_lines table in DuckDB, giving us weeks of
data to train the sports residual model.

Usage::
    python scripts/sbr_backfill.py --days 14 --leagues nba nhl mlb
    python scripts/sbr_backfill.py --start 2026-03-15 --end 2026-04-10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.data.schemas import CREATE_SPORTSBOOK_GAME_LINES
from moneygone.data.store import DataStore

log = structlog.get_logger("sbr_backfill")

# ---------------------------------------------------------------------------
# SBR URL patterns
# ---------------------------------------------------------------------------

_SBR_BASE = "https://www.sportsbookreview.com/betting-odds"

_SPORT_SLUGS = {
    # US majors
    "nba": "nba-basketball",
    "nhl": "nhl-hockey",
    "mlb": "mlb-baseball",
    "nfl": "nfl-football",
    "ncaab": "ncaa-basketball",
    "ncaaf": "college-football",
    "wnba": "wnba-basketball",
    # Soccer
    "soccer_epl": "english-premier-league",
    "soccer_spain_la_liga": "la-liga",
    "soccer_germany_bundesliga": "bundesliga",
    "soccer_italy_serie_a": "serie-a",
    "soccer_france_ligue_one": "ligue-1",
    "soccer_usa_mls": "major-league-soccer",
    "soccer_ucl": "champions-league",
    "soccer_eredivisie": "eredivisie",
    "soccer_portugal": "liga-portugal",
    "soccer_turkey": "turkish-super-lig",
    "soccer_brazil": "brasileiro-serie-a",
    "soccer_liga_mx": "liga-mx",
    # Tennis
    "tennis_atp": "atp-tennis",
    "tennis_wta": "wta-tennis",
    # MMA / UFC
    "ufc": "ufc",
    # Boxing
    "boxing": "boxing",
    # Other
    "kbo": "kbo",
    "npb": "npb",
    "afl": "afl",
}

# Map SBR bookmaker names → our canonical names
_BOOK_MAP = {
    "betmgm": "betmgm",
    "fanduel": "fanduel",
    "caesars": "caesars",
    "bet365": "bet365",
    "draftkings": "draftkings",
    "fanatics": "fanatics",
    "pinnacle": "pinnacle",
    "bovada": "bovada",
    "betonline": "betonline",
}


_SOCCER_SPORTS = {
    "soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga",
    "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_usa_mls",
    "soccer_ucl", "soccer_eredivisie", "soccer_portugal", "soccer_turkey",
    "soccer_brazil", "soccer_liga_mx",
}


def _sbr_url(sport: str, date: str) -> str:
    slug = _SPORT_SLUGS.get(sport)
    if not slug:
        raise ValueError(f"Unknown sport: {sport}")
    # Soccer uses /money-line/full-game/ path; other sports use /money-line/
    if sport in _SOCCER_SPORTS:
        return f"{_SBR_BASE}/{slug}/money-line/full-game/?date={date}"
    return f"{_SBR_BASE}/{slug}/money-line/?date={date}"


# ---------------------------------------------------------------------------
# Parse __NEXT_DATA__ JSON
# ---------------------------------------------------------------------------


def _extract_next_data(html: str) -> dict | None:
    """Extract the __NEXT_DATA__ JSON from an SBR page."""
    pattern = r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return 2.0  # Even money fallback


def _parse_odds_value(val: Any) -> float | None:
    """Parse an odds value from SBR data — could be American int or string."""
    if val is None:
        return None
    try:
        american = float(val)
        if american == 0:
            return None
        return _american_to_decimal(american)
    except (ValueError, TypeError):
        return None


def _parse_games(data: dict, sport: str, date_str: str) -> list[dict]:
    """Parse game rows from __NEXT_DATA__ into our sportsbook_game_lines format.

    SBR structure::
        props.pageProps.oddsTables[0].oddsTableModel.gameRows[].{
            gameView: {homeTeam: {fullName}, awayTeam: {fullName}, startDate},
            oddsViews[]: {
                sportsbook: "betmgm",
                currentLine: {homeOdds: -220, awayOdds: 180},
                openingLine: {homeOdds: -155, awayOdds: 125},
            }
        }
    """
    rows: list[dict] = []

    props = data.get("props", {})
    page_props = props.get("pageProps", {})

    # Navigate: oddsTables[0].oddsTableModel.gameRows
    odds_tables = page_props.get("oddsTables", [])
    if not odds_tables:
        return rows
    odds_table_model = odds_tables[0].get("oddsTableModel", {})
    game_rows = odds_table_model.get("gameRows", [])

    for game in game_rows:
        game_view = game.get("gameView", {})
        odds_views = game.get("oddsViews", [])

        # Teams
        home_info = game_view.get("homeTeam", {})
        away_info = game_view.get("awayTeam", {})
        home_team = home_info.get("fullName", "")
        away_team = away_info.get("fullName", "")
        if not home_team or not away_team:
            continue

        # Game time (ISO format from SBR)
        start_time_str = game_view.get("startDate")
        commence_time = None
        if start_time_str:
            try:
                commence_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass
        if commence_time is None:
            try:
                commence_time = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

        event_id = f"sbr_{sport}_{date_str}_{_slugify(away_team)}_at_{_slugify(home_team)}"

        for odds_view in odds_views:
            if not isinstance(odds_view, dict):
                continue

            bookmaker = (odds_view.get("sportsbook") or "").lower().strip()
            # Normalize bookmaker name
            canonical = None
            for pattern, canon in _BOOK_MAP.items():
                if pattern in bookmaker:
                    canonical = canon
                    break
            if canonical is None:
                canonical = bookmaker.replace(" ", "")
            if not canonical:
                continue

            # Current line: homeOdds / awayOdds / drawOdds in American format
            current_line = odds_view.get("currentLine") or {}
            home_am = current_line.get("homeOdds")
            away_am = current_line.get("awayOdds")
            draw_am = current_line.get("drawOdds")

            home_price = _parse_odds_value(home_am)
            away_price = _parse_odds_value(away_am)
            draw_price = _parse_odds_value(draw_am)  # 3-way for soccer

            if home_price is None or away_price is None:
                continue

            # Opening line for line movement data
            opening_line = odds_view.get("openingLine") or {}
            open_home_am = opening_line.get("homeOdds")
            open_away_am = opening_line.get("awayOdds")

            # Spread/total from current line if available
            spread_home = None
            total = None
            if current_line.get("homeSpread") is not None:
                try:
                    spread_home = float(current_line["homeSpread"])
                except (ValueError, TypeError):
                    pass
            if current_line.get("total") is not None:
                try:
                    total = float(current_line["total"])
                except (ValueError, TypeError):
                    pass

            rows.append({
                "event_id": event_id,
                "sport": sport,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": canonical,
                "commence_time": commence_time,
                "home_price": home_price,
                "away_price": away_price,
                "draw_price": draw_price,
                "spread_home": spread_home,
                "total": total,
                "captured_at": commence_time - timedelta(hours=2),
                "ingested_at": datetime.now(timezone.utc),
            })

    return rows


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------


def backfill(
    leagues: list[str],
    start_date: datetime,
    end_date: datetime,
    db_path: Path,
    delay: float = 2.0,
) -> dict[str, int]:
    """Fetch historical odds from SBR and insert into DuckDB."""
    store = DataStore(db_path)
    store.initialize_schema([CREATE_SPORTSBOOK_GAME_LINES])

    stats: dict[str, int] = {"total_rows": 0, "total_pages": 0, "errors": 0}

    client = httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
        follow_redirects=True,
        timeout=30.0,
    )

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")

        for league in leagues:
            if league not in _SPORT_SLUGS:
                continue

            url = _sbr_url(league, date_str)
            log.info("sbr.fetching", url=url, league=league, date=date_str)

            try:
                resp = client.get(url)
                resp.raise_for_status()
            except Exception:
                log.warning("sbr.fetch_failed", url=url, exc_info=True)
                stats["errors"] += 1
                time.sleep(delay)
                continue

            data = _extract_next_data(resp.text)
            if data is None:
                log.warning("sbr.no_next_data", url=url)
                stats["errors"] += 1
                time.sleep(delay)
                continue

            game_rows = _parse_games(data, league, date_str)
            stats["total_pages"] += 1

            if game_rows:
                # Insert into DuckDB
                store._conn.executemany(
                    """
                    INSERT INTO sportsbook_game_lines
                        (event_id, sport, home_team, away_team, bookmaker,
                         commence_time, home_price, away_price, draw_price,
                         spread_home, total, captured_at, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            r["event_id"], r["sport"], r["home_team"],
                            r["away_team"], r["bookmaker"], r["commence_time"],
                            r["home_price"], r["away_price"], r["draw_price"],
                            r["spread_home"], r["total"], r["captured_at"],
                            r["ingested_at"],
                        )
                        for r in game_rows
                    ],
                )
                stats["total_rows"] += len(game_rows)
                log.info(
                    "sbr.inserted",
                    league=league, date=date_str,
                    games=len(set(r["event_id"] for r in game_rows)),
                    rows=len(game_rows),
                )
            else:
                log.info("sbr.no_games", league=league, date=date_str)

            time.sleep(delay)

        current += timedelta(days=1)

    store.close()
    client.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical odds from SBR")
    parser.add_argument("--days", type=int, default=14, help="Days back to fetch")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--leagues", nargs="+", default=["nba", "nhl", "mlb"])
    parser.add_argument("--db", type=str, default="data/training.duckdb")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)

    print(f"Backfilling {args.leagues} from {start.date()} to {end.date()}")
    print(f"Database: {args.db}")
    print()

    stats = backfill(args.leagues, start, end, Path(args.db), args.delay)

    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"  Pages fetched: {stats['total_pages']}")
    print(f"  Rows inserted: {stats['total_rows']}")
    print(f"  Errors:        {stats['errors']}")


if __name__ == "__main__":
    main()
