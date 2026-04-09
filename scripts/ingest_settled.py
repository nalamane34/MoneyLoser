#!/usr/bin/env python3
"""Ingest settled market data from JSON into DuckDB.

Reads data/settled_markets.json and inserts all settled market records
into the market_states table in the DuckDB DataStore.

Usage::

    python scripts/ingest_settled.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.data.store import DataStore

log = structlog.get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "settled_markets.json"
DB_PATH = DATA_DIR / "moneygone.duckdb"


def parse_dollar_str(val: str | None) -> float | None:
    """Convert a dollar-string like '0.0800' to a float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_fp_str(val: str | None) -> int | None:
    """Convert a floating-point string like '1.00' to an int, or None."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def parse_close_time(val: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp string to a datetime."""
    if val is None:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'
        ts = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        # DuckDB needs naive UTC timestamps
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def load_and_transform(json_path: Path) -> list[dict]:
    """Load settled_markets.json and map fields to the market_states schema."""
    with open(json_path) as f:
        raw_markets = json.load(f)

    rows = []
    skipped = 0
    for m in raw_markets:
        close_time = parse_close_time(m.get("close_time"))
        if close_time is None:
            skipped += 1
            continue

        row = {
            "ticker": m["ticker"],
            "event_ticker": m.get("event_ticker", ""),
            "title": m.get("title", ""),
            "status": m.get("status", "unknown"),
            "yes_bid": parse_dollar_str(m.get("yes_bid_dollars")),
            "yes_ask": parse_dollar_str(m.get("yes_ask_dollars")),
            "last_price": parse_dollar_str(m.get("last_price_dollars")),
            "volume": parse_fp_str(m.get("volume_fp")),
            "open_interest": parse_fp_str(m.get("open_interest_fp")),
            "close_time": close_time,
            "result": m.get("result"),
            "category": m.get("category"),
        }
        rows.append(row)

    log.info(
        "ingest_settled.loaded",
        total_raw=len(raw_markets),
        valid_rows=len(rows),
        skipped=skipped,
    )
    return rows


def main() -> None:
    print(f"Loading settled markets from {JSON_PATH} ...")
    rows = load_and_transform(JSON_PATH)
    if not rows:
        print("No rows to insert. Exiting.")
        return

    print(f"Opening DuckDB at {DB_PATH} ...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = DataStore(DB_PATH)
    store.initialize_schema()

    # Insert in batches of 1000
    batch_size = 1000
    total_inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        store.insert_market_states(batch)
        total_inserted += len(batch)
        print(f"  Inserted {total_inserted}/{len(rows)} rows ...")

    # Verify
    count = store._conn.execute("SELECT COUNT(*) FROM market_states").fetchone()[0]
    print(f"Done. Total rows in market_states: {count}")

    # Show some stats
    weather_count = store._conn.execute(
        "SELECT COUNT(*) FROM market_states WHERE ticker LIKE 'KXTEMPNYCH%'"
    ).fetchone()[0]
    print(f"  Weather (KXTEMPNYCH) markets: {weather_count}")

    result_counts = store._conn.execute(
        "SELECT result, COUNT(*) FROM market_states GROUP BY result ORDER BY result"
    ).fetchall()
    for result, cnt in result_counts:
        print(f"  result={result}: {cnt}")

    store.close()


if __name__ == "__main__":
    main()
