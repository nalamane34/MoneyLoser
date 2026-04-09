#!/usr/bin/env python3
"""Fetch historical candlestick data from Kalshi for settled binary markets.

For each non-KXMVE settled binary market in data/settled_markets.json, fetches
hourly candlestick data from the Kalshi API and stores it in DuckDB.

Usage::

    python scripts/ingest_historical.py [--limit N] [--series TICKER_PREFIX]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.data.store import DataStore

log = structlog.get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "settled_markets.json"
DB_PATH = DATA_DIR / "moneygone.duckdb"

# Kalshi production API
BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"

# Rate limit: 0.5s between requests
REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def load_auth():
    """Load API key and private key from environment."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ["KALSHI_API_KEY_ID"]
    pk_path = os.environ["KALSHI_PRIVATE_KEY_PATH"]
    with open(pk_path, "rb") as f:
        pk = serialization.load_pem_private_key(f.read(), password=None)
    return api_key, pk


def get_auth_headers(api_key: str, pk, method: str, path: str) -> dict[str, str]:
    """Generate signed auth headers for a request."""
    ts = str(int(time.time() * 1000))
    message = (ts + method + path).encode()
    sig = pk.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=32,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# Candlestick fetching
# ---------------------------------------------------------------------------


def extract_series_ticker(ticker: str) -> str:
    """Extract the series ticker from a market ticker.

    KXTEMPNYCH-26APR0909-T47.99 -> KXTEMPNYCH
    KXA100MON-26APR20 -> KXA100MON
    """
    return ticker.split("-")[0]


def fetch_candlesticks(
    client: httpx.Client,
    api_key: str,
    pk,
    series_ticker: str,
    market_ticker: str,
) -> list[dict]:
    """Fetch hourly candlestick data for a single market."""
    path = f"{API_PREFIX}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    now_ts = int(time.time())
    # Kalshi requires start_ts > 0; use 30 days ago as a reasonable default
    start_ts = now_ts - (30 * 24 * 3600)
    params = {
        "period_interval": 60,
        "start_ts": start_ts,
        "end_ts": now_ts,
    }

    # Build path with query string for signing
    from urllib.parse import urlencode
    sign_path = path + "?" + urlencode(params)
    headers = get_auth_headers(api_key, pk, "GET", sign_path)

    try:
        response = client.get(
            BASE_URL + path,
            params=params,
            headers=headers,
            timeout=30.0,
        )

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 5))
            log.warning("rate_limited", retry_after=retry_after, ticker=market_ticker)
            time.sleep(retry_after)
            # Retry once
            headers = get_auth_headers(api_key, pk, "GET", sign_path)
            response = client.get(
                BASE_URL + path,
                params=params,
                headers=headers,
                timeout=30.0,
            )

        if response.status_code == 404:
            log.debug("candlestick.not_found", ticker=market_ticker)
            return []

        if response.status_code != 200:
            log.warning(
                "candlestick.error",
                ticker=market_ticker,
                status=response.status_code,
                body=response.text[:200],
            )
            return []

        data = response.json()
        return data.get("candlesticks", [])

    except httpx.TransportError as exc:
        log.warning("candlestick.transport_error", ticker=market_ticker, error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def store_candlesticks(store: DataStore, ticker: str, candles: list[dict]) -> int:
    """Store candlestick data as trades in the DuckDB trades table.

    Each candlestick becomes a synthetic trade record capturing the
    OHLCV data for that period. We use the trades table since there is
    no dedicated candlestick table.
    """
    if not candles:
        return 0

    rows = []
    for i, c in enumerate(candles):
        # Candlestick fields: open_time, close_time, price (or yes_price),
        # open, high, low, close, volume
        ts = c.get("end_period_ts") or c.get("close_time") or c.get("open_time", 0)
        if isinstance(ts, (int, float)):
            trade_time = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        else:
            trade_time = datetime.fromisoformat(
                str(ts).replace("Z", "+00:00")
            ).replace(tzinfo=None)

        # Use close price as the yes_price
        price = float(c.get("price", {}).get("close", 0) if isinstance(c.get("price"), dict) else c.get("close", c.get("yes_price", 0)))
        volume = int(float(c.get("volume", 0)))

        rows.append({
            "trade_id": f"candle-{ticker}-{i}",
            "ticker": ticker,
            "count": max(volume, 1),
            "yes_price": price,
            "taker_side": "yes",
            "trade_time": trade_time,
        })

    if rows:
        store.insert_trades(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fetch historical candlestick data")
    parser.add_argument("--limit", type=int, default=0, help="Max markets to fetch (0=all)")
    parser.add_argument("--series", type=str, default="", help="Only fetch this series prefix")
    parser.add_argument("--dry-run", action="store_true", help="List markets without fetching")
    args = parser.parse_args()

    # Load markets
    with open(JSON_PATH) as f:
        raw_markets = json.load(f)

    # Filter to non-MVE binary markets
    markets = [
        m for m in raw_markets
        if not m["ticker"].startswith("KXMVE")
        and m.get("market_type") == "binary"
    ]

    if args.series:
        markets = [m for m in markets if m["ticker"].startswith(args.series)]

    print(f"Found {len(markets)} non-MVE binary markets to process")

    if args.limit > 0:
        markets = markets[: args.limit]
        print(f"  (limited to {args.limit})")

    if args.dry_run:
        for m in markets:
            series = extract_series_ticker(m["ticker"])
            print(f"  {series} / {m['ticker']}")
        return

    # Load auth
    api_key, pk = load_auth()
    print("Auth loaded successfully")

    # Open DB
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = DataStore(DB_PATH)
    store.initialize_schema()

    # Fetch candlesticks
    total_candles = 0
    errors = 0
    client = httpx.Client(http2=True)

    try:
        for i, m in enumerate(markets):
            ticker = m["ticker"]
            series = extract_series_ticker(ticker)

            candles = fetch_candlesticks(client, api_key, pk, series, ticker)
            if candles:
                n = store_candlesticks(store, ticker, candles)
                total_candles += n
                print(f"  [{i+1}/{len(markets)}] {ticker}: {n} candlesticks")
            else:
                print(f"  [{i+1}/{len(markets)}] {ticker}: no data")

            # Rate limit
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        client.close()

    print(f"\nDone. Total candlesticks stored: {total_candles}, errors: {errors}")

    # Verify
    count = store._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    print(f"Total rows in trades table: {count}")

    store.close()


if __name__ == "__main__":
    main()
