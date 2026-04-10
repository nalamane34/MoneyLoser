#!/usr/bin/env python3
"""Comprehensive historical data backfill from Kalshi API.

Pulls settled markets, candlesticks, trades, and settlement outcomes
from the Kalshi API and stores them in DuckDB for backtesting.

Candlestick bid/ask OHLC data is used to synthesize orderbook snapshots
so the backtest engine can evaluate markets and simulate fills.

The script is resumable: it tracks progress in a JSON checkpoint file
and skips already-processed markets on restart.

Usage::

    # Full backfill (all settled markets, hourly candles)
    python scripts/backfill_historical.py --config config/default.yaml --overlay config/live.yaml

    # Specific series only
    python scripts/backfill_historical.py --config config/default.yaml --overlay config/live.yaml --series KXNBA,KXEPL

    # Limit to N markets (for testing)
    python scripts/backfill_historical.py --config config/default.yaml --overlay config/live.yaml --limit 50

    # Dry run: count markets without fetching
    python scripts/backfill_historical.py --config config/default.yaml --overlay config/live.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.market_discovery import MarketCategory, classify_market
from moneygone.data.schemas import ALL_TABLES
from moneygone.data.store import DataStore
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Candlestick, Market, Trade
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("backfill")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "backtest.duckdb"
CHECKPOINT_PATH = DATA_DIR / "backfill_checkpoint.json"

# Rate limiting: conservative to stay well under API limits
REQUEST_DELAY = 0.35  # seconds between individual requests
BATCH_DELAY = 1.0     # seconds between large operations


# ---------------------------------------------------------------------------
# Checkpoint management (resume support)
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed_tickers": [], "phase": "markets", "cursor": None}


def save_checkpoint(state: dict[str, Any]) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 1: Fetch all settled markets from the API
# ---------------------------------------------------------------------------


async def fetch_all_settled_markets(client: KalshiRestClient) -> list[Market]:
    """Paginate through all historical (settled) markets."""
    all_markets: list[Market] = []
    cursor: str | None = None
    page = 0

    while True:
        page += 1
        params: dict[str, Any] = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor

        try:
            data = await client._request("GET", "/historical/markets", params=params)
        except Exception as e:
            log.error("backfill.fetch_markets_error", page=page, error=str(e))
            break

        markets_raw = data.get("markets", [])
        if not markets_raw:
            break

        for m_raw in markets_raw:
            try:
                market = client._parse_market(m_raw)
                all_markets.append(market)
            except Exception:
                pass

        cursor = data.get("cursor", "")
        log.info("backfill.markets_page", page=page, count=len(markets_raw), total=len(all_markets))

        if not cursor:
            break

        await asyncio.sleep(REQUEST_DELAY)

    return all_markets


# ---------------------------------------------------------------------------
# Phase 2: Fetch candlesticks & synthesize orderbook + market states
# ---------------------------------------------------------------------------


def candlestick_to_market_state(
    ticker: str,
    event_ticker: str,
    title: str,
    close_time: datetime,
    result: str,
    category: str,
    candle: Candlestick,
) -> dict[str, Any]:
    """Convert a candlestick into a market_states row."""
    yes_bid = float(candle.yes_bid_ohlc.close) if candle.yes_bid_ohlc else float(candle.close) - 0.01
    yes_ask = float(candle.yes_ask_ohlc.close) if candle.yes_ask_ohlc else float(candle.close) + 0.01

    # Clamp to valid range
    yes_bid = max(0.01, min(0.99, yes_bid))
    yes_ask = max(0.01, min(0.99, yes_ask))
    if yes_ask <= yes_bid:
        yes_ask = yes_bid + 0.01

    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "status": "open",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "last_price": float(candle.close),
        "volume": candle.volume,
        "open_interest": candle.open_interest,
        "close_time": close_time if isinstance(close_time, datetime) else datetime.fromisoformat(str(close_time).replace("Z", "+00:00")).replace(tzinfo=None),
        "result": result,
        "category": category,
    }


def candlestick_to_orderbook(
    ticker: str,
    candle: Candlestick,
    seq: int,
) -> dict[str, Any]:
    """Synthesize an orderbook snapshot from candlestick bid/ask OHLC.

    Creates a multi-level orderbook using the OHLC range:
    - Close price is the top-of-book level (highest liquidity)
    - Open/high/low provide additional depth levels
    - Volume is distributed across levels
    """
    # Yes bid levels (descending by price — best bid first)
    if candle.yes_bid_ohlc:
        bid_close = float(candle.yes_bid_ohlc.close)
        bid_open = float(candle.yes_bid_ohlc.open)
        bid_high = float(candle.yes_bid_ohlc.high)
        bid_low = float(candle.yes_bid_ohlc.low)
    else:
        bid_close = float(candle.close) - 0.01
        bid_open = bid_close
        bid_high = bid_close
        bid_low = bid_close

    # Yes ask levels (ascending by price — best ask first)
    if candle.yes_ask_ohlc:
        ask_close = float(candle.yes_ask_ohlc.close)
        ask_open = float(candle.yes_ask_ohlc.open)
        ask_high = float(candle.yes_ask_ohlc.high)
        ask_low = float(candle.yes_ask_ohlc.low)
    else:
        ask_close = float(candle.close) + 0.01
        ask_open = ask_close
        ask_high = ask_close
        ask_low = ask_close

    # Distribute volume across levels (primary level gets ~60%)
    total_vol = max(candle.volume, 1)
    primary_vol = max(int(total_vol * 0.6), 1)
    secondary_vol = max(int(total_vol * 0.25), 1)
    tertiary_vol = max(int(total_vol * 0.15), 1)

    # Build yes bid levels (best bid first = descending price)
    yes_bids = set()
    yes_bids.add((round(max(0.01, bid_close), 2), primary_vol))
    if abs(bid_open - bid_close) > 0.005:
        yes_bids.add((round(max(0.01, bid_open), 2), secondary_vol))
    if abs(bid_low - bid_close) > 0.005:
        yes_bids.add((round(max(0.01, bid_low), 2), tertiary_vol))

    # Build no bid levels (derived from yes ask: no_bid = 1 - yes_ask)
    no_bids = set()
    no_bid_close = round(max(0.01, 1.0 - ask_close), 2)
    no_bid_open = round(max(0.01, 1.0 - ask_open), 2)
    no_bid_low = round(max(0.01, 1.0 - ask_high), 2)  # high ask = low no bid
    no_bids.add((no_bid_close, primary_vol))
    if abs(no_bid_open - no_bid_close) > 0.005:
        no_bids.add((no_bid_open, secondary_vol))
    if abs(no_bid_low - no_bid_close) > 0.005:
        no_bids.add((no_bid_low, tertiary_vol))

    # Convert to sorted lists (descending by price for bids)
    yes_levels = sorted(yes_bids, key=lambda x: -x[0])
    no_levels = sorted(no_bids, key=lambda x: -x[0])

    ts = candle.end_period_ts
    if isinstance(ts, datetime):
        snapshot_time = ts.replace(tzinfo=None) if ts.tzinfo else ts
    else:
        snapshot_time = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)

    return {
        "ticker": ticker,
        "yes_levels": [[p, q] for p, q in yes_levels],
        "no_levels": [[p, q] for p, q in no_levels],
        "seq": seq,
        "snapshot_time": snapshot_time,
    }


def candlestick_to_trade(
    ticker: str,
    candle: Candlestick,
    idx: int,
) -> dict[str, Any] | None:
    """Convert a candlestick into a synthetic trade record."""
    if candle.volume <= 0:
        return None

    ts = candle.end_period_ts
    if isinstance(ts, datetime):
        trade_time = ts.replace(tzinfo=None) if ts.tzinfo else ts
    else:
        trade_time = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)

    return {
        "trade_id": f"synth-{ticker}-{idx}",
        "ticker": ticker,
        "count": max(candle.volume, 1),
        "yes_price": float(candle.close),
        "taker_side": "yes" if candle.close >= candle.open else "no",
        "trade_time": trade_time,
    }


async def process_market(
    client: KalshiRestClient,
    market: Market,
    store: DataStore,
    lookback_days: int = 90,
) -> dict[str, int]:
    """Fetch candlesticks for a single market and store all derived data.

    Returns counts of rows inserted.
    """
    ticker = market.ticker
    event_ticker = market.event_ticker
    title = market.title
    result_str = market.result.value if market.result else ""
    category = classify_market(market).value

    # Determine time range for candlesticks
    close_time = market.close_time
    if isinstance(close_time, str):
        close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))

    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)

    end_ts = int(close_time.timestamp())
    start_ts = end_ts - (lookback_days * 24 * 3600)

    # Fetch hourly candlesticks
    try:
        candles = await client.get_historical_candlesticks(
            ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=60,  # hourly
        )
    except Exception as e:
        log.debug("backfill.candlestick_error", ticker=ticker, error=str(e))
        candles = []

    await asyncio.sleep(REQUEST_DELAY)

    if not candles:
        # Even without candles, store the final market state and settlement
        close_time_naive = close_time.replace(tzinfo=None) if close_time.tzinfo else close_time
        final_state = {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": title,
            "status": "finalized",
            "yes_bid": float(market.yes_bid),
            "yes_ask": float(market.yes_ask),
            "last_price": float(market.last_price),
            "volume": market.volume,
            "open_interest": market.open_interest,
            "close_time": close_time_naive,
            "result": result_str,
            "category": category,
        }
        store.insert_market_states([final_state])

        # Settlement record
        if result_str in ("yes", "no", "all_yes", "all_no"):
            payout = 1.0 if result_str in ("yes", "all_yes") else 0.0
            store.insert_settlements([{
                "ticker": ticker,
                "market_result": result_str,
                "revenue": 0.0,
                "payout": payout,
                "settled_time": close_time_naive,
            }])

        return {"market_states": 1, "orderbooks": 0, "trades": 0, "settlements": 1}

    # Process candlesticks into market states, orderbooks, trades
    close_time_naive = close_time.replace(tzinfo=None) if close_time.tzinfo else close_time

    market_rows: list[dict] = []
    orderbook_rows: list[dict] = []
    trade_rows: list[dict] = []

    for i, candle in enumerate(candles):
        # Market state from each candle
        market_rows.append(candlestick_to_market_state(
            ticker, event_ticker, title, close_time_naive,
            result_str, category, candle,
        ))

        # Synthetic orderbook
        orderbook_rows.append(candlestick_to_orderbook(ticker, candle, seq=i))

        # Synthetic trade
        trade = candlestick_to_trade(ticker, candle, idx=i)
        if trade:
            trade_rows.append(trade)

    # Batch insert
    if market_rows:
        store.insert_market_states(market_rows)
    if orderbook_rows:
        store.insert_orderbook_snapshots(orderbook_rows)
    if trade_rows:
        store.insert_trades(trade_rows)

    # Settlement record
    if result_str in ("yes", "no", "all_yes", "all_no"):
        payout = 1.0 if result_str in ("yes", "all_yes") else 0.0
        store.insert_settlements([{
            "ticker": ticker,
            "market_result": result_str,
            "revenue": 0.0,
            "payout": payout,
            "settled_time": close_time_naive,
        }])

    return {
        "market_states": len(market_rows),
        "orderbooks": len(orderbook_rows),
        "trades": len(trade_rows),
        "settlements": 1 if result_str in ("yes", "no", "all_yes", "all_no") else 0,
    }


# ---------------------------------------------------------------------------
# Phase 3: Fetch real historical trades (not synthetic)
# ---------------------------------------------------------------------------


async def fetch_historical_trades(
    client: KalshiRestClient,
    tickers: list[str],
    store: DataStore,
    batch_size: int = 50,
) -> int:
    """Fetch real historical trades for settled markets."""
    total = 0

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        for ticker in batch:
            try:
                trades = await client.get_historical_trades(ticker=ticker, limit=200)
                if trades:
                    trade_rows = []
                    for t in trades:
                        trade_time = t.created_time
                        if isinstance(trade_time, datetime) and trade_time.tzinfo:
                            trade_time = trade_time.replace(tzinfo=None)
                        trade_rows.append({
                            "trade_id": t.trade_id,
                            "ticker": t.ticker,
                            "count": t.count,
                            "yes_price": float(t.yes_price),
                            "taker_side": t.taker_side,
                            "trade_time": trade_time,
                        })
                    store.insert_trades(trade_rows)
                    total += len(trade_rows)
            except Exception:
                pass

            await asyncio.sleep(REQUEST_DELAY)

        log.info("backfill.trades_batch", processed=min(i + batch_size, len(tickers)), total_tickers=len(tickers), total_trades=total)
        await asyncio.sleep(BATCH_DELAY)

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical data from Kalshi API")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/live.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Max markets to process (0=all)")
    parser.add_argument("--series", type=str, default="", help="Comma-separated series prefixes to filter")
    parser.add_argument("--category", type=str, default="", help="Comma-separated categories to filter (sports,weather,crypto,...)")
    parser.add_argument("--lookback-days", type=int, default=90, help="Days of candlestick history per market")
    parser.add_argument("--db-path", type=str, default="", help="Override DuckDB path")
    parser.add_argument("--skip-trades", action="store_true", help="Skip real trade fetching (use synthetic only)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Count markets without fetching data")
    args = parser.parse_args()

    config = load_config(Path(args.config), Path(args.overlay))
    setup_logging("INFO")

    db_path = Path(args.db_path) if args.db_path else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = KalshiRestClient(config.exchange)

    try:
        # ---- Phase 1: Fetch settled markets ----
        print("=" * 60)
        print("Phase 1: Fetching settled markets from Kalshi API...")
        print("=" * 60)

        markets = await fetch_all_settled_markets(client)
        print(f"  Fetched {len(markets)} total settled markets")

        # Filter to non-MVE binary markets
        markets = [m for m in markets if not m.ticker.startswith("KXMVE")]
        print(f"  Non-MVE markets: {len(markets)}")

        # Apply series filter
        if args.series:
            prefixes = [p.strip().upper() for p in args.series.split(",")]
            markets = [m for m in markets if any(m.ticker.startswith(p) for p in prefixes)]
            print(f"  After series filter ({args.series}): {len(markets)}")

        # Apply category filter
        if args.category:
            cat_filter = {c.strip().lower() for c in args.category.split(",")}
            markets = [
                m for m in markets
                if classify_market(m).value in cat_filter
            ]
            print(f"  After category filter ({args.category}): {len(markets)}")

        # Apply limit
        if args.limit > 0:
            markets = markets[:args.limit]
            print(f"  Limited to {args.limit} markets")

        # Show category breakdown
        cat_counts: dict[str, int] = {}
        for m in markets:
            cat = classify_market(m).value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"\n  Category breakdown:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:20s} {count:>6}")

        # Show date range
        close_times = [m.close_time for m in markets if m.close_time]
        if close_times:
            ct_values = []
            for ct in close_times:
                if isinstance(ct, datetime):
                    ct_values.append(ct)
                elif isinstance(ct, str):
                    ct_values.append(datetime.fromisoformat(ct.replace("Z", "+00:00")))
            if ct_values:
                print(f"\n  Date range: {min(ct_values).date()} → {max(ct_values).date()}")

        if args.dry_run:
            print("\n--dry-run: not fetching data.")
            return

        # ---- Phase 2: Fetch candlesticks & build backtest data ----
        print("\n" + "=" * 60)
        print("Phase 2: Fetching candlesticks & building backtest data...")
        print("=" * 60)

        store = DataStore(db_path)
        store.initialize_schema(ALL_TABLES)

        # Load checkpoint for resume
        checkpoint = load_checkpoint() if args.resume else {"processed_tickers": []}
        processed = set(checkpoint.get("processed_tickers", []))
        remaining = [m for m in markets if m.ticker not in processed]
        print(f"  Already processed: {len(processed)}, remaining: {len(remaining)}")

        totals = {"market_states": 0, "orderbooks": 0, "trades": 0, "settlements": 0}
        errors = 0
        start_time = time.time()

        for i, market in enumerate(remaining):
            try:
                counts = await process_market(
                    client, market, store,
                    lookback_days=args.lookback_days,
                )
                for k, v in counts.items():
                    totals[k] += v

                processed.add(market.ticker)

                # Progress log every 25 markets
                if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1}/{len(remaining)}] "
                        f"states={totals['market_states']} ob={totals['orderbooks']} "
                        f"trades={totals['trades']} settlements={totals['settlements']} "
                        f"({rate:.1f}/s, ETA {eta/60:.0f}m)"
                    )

                # Checkpoint every 100 markets
                if (i + 1) % 100 == 0:
                    save_checkpoint({"processed_tickers": list(processed)})

            except Exception as e:
                errors += 1
                log.warning("backfill.market_error", ticker=market.ticker, error=str(e))

        save_checkpoint({"processed_tickers": list(processed)})

        print(f"\n  Phase 2 complete:")
        print(f"    Market states:  {totals['market_states']:>8}")
        print(f"    Orderbooks:     {totals['orderbooks']:>8}")
        print(f"    Trades (synth): {totals['trades']:>8}")
        print(f"    Settlements:    {totals['settlements']:>8}")
        print(f"    Errors:         {errors:>8}")

        # ---- Phase 3: Fetch real historical trades ----
        if not args.skip_trades:
            print("\n" + "=" * 60)
            print("Phase 3: Fetching real historical trades...")
            print("=" * 60)

            tickers = [m.ticker for m in markets]
            real_trades = await fetch_historical_trades(client, tickers, store)
            print(f"  Real trades stored: {real_trades}")
        else:
            print("\n  Skipping real trade fetch (--skip-trades)")

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("Backfill complete!")
        print("=" * 60)

        # Verify DB contents
        for table in ["market_states", "orderbook_snapshots", "trades", "settlements_log"]:
            try:
                count = store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table:25s} {count:>10} rows")
            except Exception:
                print(f"  {table:25s} (table not found)")

        # Show date range in DB
        try:
            result = store._conn.execute(
                "SELECT MIN(snapshot_time), MAX(snapshot_time) FROM orderbook_snapshots"
            ).fetchone()
            if result and result[0]:
                print(f"\n  Orderbook date range: {result[0]} → {result[1]}")
        except Exception:
            pass

        try:
            result = store._conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM orderbook_snapshots"
            ).fetchone()
            if result:
                print(f"  Unique tickers with orderbooks: {result[0]}")
        except Exception:
            pass

        store.close()
        print(f"\n  Database: {db_path}")
        print(f"  To run backtest: python scripts/run_backtest.py --db {db_path}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
