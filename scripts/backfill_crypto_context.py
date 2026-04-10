#!/usr/bin/env python3
"""Backfill crypto context data for backtesting.

Fetches historical hourly OHLCV data from a US-accessible exchange (binanceus
or kraken) via CCXT, computes all crypto_snapshot fields needed by the
CryptoVolModel / crypto features pipeline, and stores them in the
``crypto_context`` table of the backtest DuckDB.

Usage::

    # Backfill the default crypto backtest DB
    python scripts/backfill_crypto_context.py

    # Specify a different DB path
    python scripts/backfill_crypto_context.py --db data/backtest_crypto.duckdb

    # Force a specific exchange
    python scripts/backfill_crypto_context.py --exchange kraken

    # Extra lookback days for volatility calculation
    python scripts/backfill_crypto_context.py --lookback-days 45
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Annualization factor: hours in a year (365.25 * 24)
HOURS_PER_YEAR = 8766.0

# Ticker prefix -> CCXT symbol mapping
# Kalshi crypto tickers use prefixes like KXBTC, KXETH, KXDOGE, etc.
TICKER_SYMBOL_MAP = {
    "KXBTC": "BTC/USDT",
    "KXBTCD": "BTC/USDT",
    "KXETH": "BTC/USDT",   # Will be overridden below
    "KXETHD": "ETH/USDT",
    "KXDOGE": "DOGE/USDT",
    "KXSHIBA": "SHIB/USDT",
    "KXSHIBAD": "SHIB/USDT",
    "KXSOL": "SOL/USDT",
    "KXSOLD": "SOL/USDT",
    "KXSOLE": "SOL/USDT",
}

# Fix ETH mapping
TICKER_SYMBOL_MAP["KXETH"] = "ETH/USDT"

# Exchange fallback order (US-accessible)
EXCHANGE_PRIORITY = ["binanceus", "kraken"]

# Kraken uses different quote currency
KRAKEN_SYMBOL_MAP = {
    "BTC/USDT": "BTC/USD",
    "ETH/USDT": "ETH/USD",
    "DOGE/USDT": "DOGE/USD",
    "SHIB/USDT": "SHIB/USD",
    "SOL/USDT": "SOL/USD",
}


# ---------------------------------------------------------------------------
# CCXT helpers
# ---------------------------------------------------------------------------

def create_exchange(exchange_id: str):
    """Create a CCXT exchange instance."""
    import ccxt

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })
    return exchange


def fetch_ohlcv(
    exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeframe: str = "1h",
) -> list[list]:
    """Fetch OHLCV candles in chunks, respecting rate limits.

    Returns list of [timestamp_ms, open, high, low, close, volume].
    """
    all_candles = []
    since = start_ms
    limit = 500  # Most exchanges support up to 500-1000

    while since < end_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"    WARN: fetch_ohlcv failed at {datetime.fromtimestamp(since/1000, tz=timezone.utc)}: {e}")
            # Wait and retry once
            time.sleep(2)
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            except Exception as e2:
                print(f"    ERROR: retry also failed: {e2}")
                break

        if not candles:
            break

        all_candles.extend(candles)

        # Advance past the last candle
        last_ts = candles[-1][0]
        if last_ts <= since:
            # No progress, move forward by 1 hour
            since += 3600 * 1000
        else:
            since = last_ts + 1

        # Rate limit courtesy
        time.sleep(exchange.rateLimit / 1000.0)

    # Filter to requested range and deduplicate
    seen = set()
    filtered = []
    for c in all_candles:
        if c[0] <= end_ms and c[0] not in seen:
            seen.add(c[0])
            filtered.append(c)

    filtered.sort(key=lambda x: x[0])
    return filtered


# ---------------------------------------------------------------------------
# Volatility / feature computation
# ---------------------------------------------------------------------------

def compute_log_returns(closes: np.ndarray) -> np.ndarray:
    """Compute log returns from close prices."""
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(closes[1:] / closes[:-1])
    return np.nan_to_num(returns, nan=0.0)


def realized_vol(log_returns: np.ndarray, window: int) -> float:
    """Annualized realized volatility from hourly log returns.

    vol = std(returns[-window:]) * sqrt(HOURS_PER_YEAR)
    """
    if len(log_returns) < window:
        window = len(log_returns)
    if window < 2:
        return 0.0
    subset = log_returns[-window:]
    std = float(np.std(subset, ddof=1))
    return std * math.sqrt(HOURS_PER_YEAR)


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """Average True Range (ATR) normalized by the latest close price.

    Uses the standard ATR definition: max(H-L, |H-Cprev|, |L-Cprev|).
    Returns ATR / close as a fraction (comparable across price levels).
    """
    if len(closes) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(hl, hc, lc))

    if len(true_ranges) < period:
        return 0.0

    atr = float(np.mean(true_ranges[-period:]))
    current_close = float(closes[-1])
    if current_close == 0:
        return 0.0
    return atr / current_close


def compute_trend_regime(closes: np.ndarray) -> tuple[str, float]:
    """Classify trend regime from price series.

    Uses 8h, 24h, and 72h returns weighted together.
    Returns (regime_label, trend_strength).

    regime_label: "strong_down", "down", "neutral", "up", "strong_up"
    trend_strength: 0.0 to 1.0
    """
    n = len(closes)
    if n < 2:
        return "neutral", 0.0

    # Compute returns at different horizons
    ret_8h = 0.0
    ret_24h = 0.0
    ret_72h = 0.0

    if n > 8 and closes[-9] > 0:
        ret_8h = (closes[-1] - closes[-9]) / closes[-9]
    if n > 24 and closes[-25] > 0:
        ret_24h = (closes[-1] - closes[-25]) / closes[-25]
    if n > 72 and closes[-73] > 0:
        ret_72h = (closes[-1] - closes[-73]) / closes[-73]

    # Weighted score: shorter horizons matter more
    score = 0.5 * ret_8h + 0.3 * ret_24h + 0.2 * ret_72h

    # Classify
    if score > 0.03:
        regime = "strong_up"
    elif score > 0.01:
        regime = "up"
    elif score < -0.03:
        regime = "strong_down"
    elif score < -0.01:
        regime = "down"
    else:
        regime = "neutral"

    # Trend strength: magnitude of the score, clamped to [0, 1]
    strength = min(abs(score) / 0.05, 1.0)

    return regime, strength


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Simplified ADX (Average Directional Index) for trend strength.

    Returns a value between 0 and 1 (normalized from the traditional 0-100 range).
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    # Directional movement
    plus_dm = np.maximum(np.diff(highs), 0)
    minus_dm = np.maximum(-np.diff(lows), 0)

    # Zero out when the other is larger
    mask = plus_dm > minus_dm
    minus_dm[mask & (minus_dm > 0)] = 0.0
    mask = minus_dm > plus_dm
    plus_dm[mask & (plus_dm > 0)] = 0.0

    # True range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )

    # Smoothed averages (simple moving average for simplicity)
    if len(tr) < period:
        return 0.0

    atr = np.mean(tr[-period:])
    if atr == 0:
        return 0.0

    plus_di = np.mean(plus_dm[-period:]) / atr
    minus_di = np.mean(minus_dm[-period:]) / atr

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0

    dx = abs(plus_di - minus_di) / di_sum
    # Normalize to 0-1 (DX is already 0-1 as a ratio)
    return float(min(dx, 1.0))


def build_snapshot(
    candles: list[list],
    idx: int,
    symbol: str,
    exchange_name: str,
) -> dict | None:
    """Build a crypto_snapshot dict from OHLCV candles at position idx.

    Uses candles[0:idx+1] to compute all features (no lookahead).
    Returns None if not enough data.
    """
    if idx < 24:
        # Need at least 24 candles for basic vol
        return None

    # Extract arrays up to and including idx
    data = candles[:idx + 1]
    timestamps = np.array([c[0] for c in data])
    opens = np.array([c[1] for c in data], dtype=float)
    highs = np.array([c[2] for c in data], dtype=float)
    lows = np.array([c[3] for c in data], dtype=float)
    closes = np.array([c[4] for c in data], dtype=float)
    volumes = np.array([c[5] for c in data], dtype=float)

    current_close = closes[-1]
    if current_close <= 0:
        return None

    log_rets = compute_log_returns(closes)
    if len(log_rets) < 24:
        return None

    # Realized volatilities
    rv_24h = realized_vol(log_rets, 24)
    rv_7d = realized_vol(log_rets, 24 * 7)
    rv_30d = realized_vol(log_rets, 24 * 30)
    rv_current = rv_24h  # Default "realized_vol" to 24h

    # ATR
    atr_14 = compute_atr(highs, lows, closes, 14)
    atr_24 = compute_atr(highs, lows, closes, 24)

    # Trend regime
    trend_regime, trend_strength = compute_trend_regime(closes)

    # ADX-based trend strength (blend with return-based)
    adx_strength = compute_adx(highs, lows, closes, 14)
    trend_strength = 0.6 * trend_strength + 0.4 * adx_strength

    # Volume metrics
    total_vol = float(np.sum(volumes[-24:])) if len(volumes) >= 24 else float(np.sum(volumes))
    # Whale volume approximation: volume from candles where volume > 2x median
    recent_vols = volumes[-24:] if len(volumes) >= 24 else volumes
    median_vol = float(np.median(recent_vols)) if len(recent_vols) > 0 else 0
    whale_volume = float(np.sum(recent_vols[recent_vols > 2 * median_vol])) if median_vol > 0 else 0.0

    # Orderbook depth proxies (from volume * price as rough proxy)
    bid_depth = float(total_vol * current_close * 0.01)  # ~1% of 24h volume as depth
    ask_depth = float(total_vol * current_close * 0.01)

    # Vol history for VolatilityRegime feature (rolling 7d windows)
    vol_history = []
    for j in range(max(0, len(log_rets) - 24 * 30), len(log_rets), 24):
        window = log_rets[j:j + 24 * 7]
        if len(window) >= 48:
            vol_history.append(float(np.std(window, ddof=1) * math.sqrt(HOURS_PER_YEAR)))

    # Implied vol: approximate as realized_vol * 1.1 (crypto vol premium)
    # In production this would come from Deribit DVOL
    implied_vol = rv_24h * 1.1 if rv_24h > 0 else rv_7d * 1.1

    # Funding rate: not available from spot OHLCV, set to 0 (neutral)
    # Open interest: not available from spot OHLCV
    funding_rate = 0.0
    open_interest = 0.0

    # Build the base symbol (e.g., "BTC" from "BTC/USDT")
    base_symbol = symbol.split("/")[0]

    snapshot = {
        # Price data
        "brti_price": current_close,
        "spot_price": current_close,
        "futures_price": current_close,  # Approximation from spot

        # Volatility
        "realized_vol": rv_current,
        "realized_vol_24h": rv_24h,
        "realized_vol_7d": rv_7d,
        "realized_vol_30d": rv_30d,
        "implied_vol": implied_vol,
        "vol_history": vol_history,

        # ATR
        "atr_14": atr_14,
        "atr_24": atr_24,

        # Trend
        "trend_regime": trend_regime,
        "trend_strength": trend_strength,

        # Volume / depth
        "total_volume": total_vol,
        "whale_volume": whale_volume,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,

        # Derivatives data (placeholder from spot)
        "funding_rate": funding_rate,
        "open_interest": open_interest,

        # Metadata
        "symbol": symbol,
        "exchange": exchange_name,
    }

    return snapshot


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def create_crypto_context_table(db: duckdb.DuckDBPyConnection) -> None:
    """Create the crypto_context table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS crypto_context (
            symbol      TEXT NOT NULL,
            timestamp   TIMESTAMP NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        )
    """)


def insert_snapshots(
    db: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, datetime, str]],
) -> int:
    """Insert snapshot rows into crypto_context.

    Deletes existing rows for the affected symbols/time ranges first,
    then bulk inserts.  This makes re-runs fully idempotent.
    """
    if not rows:
        return 0

    # Group rows by symbol to delete existing data per symbol
    by_symbol: dict[str, list[tuple[str, datetime, str]]] = {}
    for symbol, ts, json_str in rows:
        by_symbol.setdefault(symbol, []).append((symbol, ts, json_str))

    db.execute("BEGIN TRANSACTION")
    try:
        total_deleted = 0
        for symbol, symbol_rows in by_symbol.items():
            # Find time range for this symbol
            timestamps = [t for _, t, _ in symbol_rows]
            min_ts = min(timestamps)
            max_ts = max(timestamps)

            # Delete existing rows in this range
            result = db.execute(
                "DELETE FROM crypto_context WHERE symbol = ? "
                "AND timestamp >= ? AND timestamp <= ?",
                [symbol, min_ts, max_ts],
            )
            total_deleted += result.fetchone()[0] if result.description else 0

        # Bulk insert all rows
        for symbol, ts, json_str in rows:
            db.execute(
                "INSERT INTO crypto_context (symbol, timestamp, snapshot_json) "
                "VALUES (?, ?, ?)",
                [symbol, ts, json_str],
            )

        db.execute("COMMIT")

        if total_deleted > 0:
            print(f"  (replaced {total_deleted} existing rows)")
        return len(rows)
    except Exception:
        db.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_symbols(db_path: Path) -> dict[str, tuple[datetime, datetime]]:
    """Discover which crypto symbols are needed and their date ranges.

    Returns {ccxt_symbol: (min_time, max_time)} from the orderbook_snapshots.
    """
    db = duckdb.connect(str(db_path), read_only=True)
    try:
        tickers = db.execute("SELECT DISTINCT ticker FROM orderbook_snapshots").fetchall()
    finally:
        db.close()

    symbol_ranges: dict[str, tuple[datetime, datetime]] = {}

    # Map each ticker to its CCXT symbol
    for (ticker,) in tickers:
        # Extract prefix: KXBTC15M-... -> KXBTC, KXDOGE-... -> KXDOGE
        m = re.match(r"(KX[A-Z]+)", ticker)
        if m:
            prefix = m.group(1)
            # Strip trailing 15M/D suffix patterns for matching
            # KXBTC15M -> KXBTC, KXBTCD -> KXBTCD
            ccxt_symbol = TICKER_SYMBOL_MAP.get(prefix)
            if ccxt_symbol is None:
                # Try without trailing D (daily variant)
                base_prefix = prefix.rstrip("D")
                ccxt_symbol = TICKER_SYMBOL_MAP.get(base_prefix)

            if ccxt_symbol:
                if ccxt_symbol not in symbol_ranges:
                    symbol_ranges[ccxt_symbol] = (
                        datetime.max.replace(tzinfo=timezone.utc),
                        datetime.min.replace(tzinfo=timezone.utc),
                    )

    # Now get actual date ranges per symbol from the DB
    db = duckdb.connect(str(db_path), read_only=True)
    try:
        for (ticker,) in tickers:
            m = re.match(r"(KX[A-Z]+)", ticker)
            if not m:
                continue
            prefix = m.group(1)
            ccxt_symbol = TICKER_SYMBOL_MAP.get(prefix)
            if ccxt_symbol is None:
                ccxt_symbol = TICKER_SYMBOL_MAP.get(prefix.rstrip("D"))
            if ccxt_symbol is None:
                continue

            result = db.execute(
                "SELECT MIN(snapshot_time), MAX(snapshot_time) "
                "FROM orderbook_snapshots WHERE ticker = ?",
                [ticker],
            ).fetchone()
            if result and result[0]:
                min_t = result[0]
                max_t = result[1]
                if not isinstance(min_t, datetime):
                    min_t = datetime.fromisoformat(str(min_t))
                if not isinstance(max_t, datetime):
                    max_t = datetime.fromisoformat(str(max_t))
                if min_t.tzinfo is None:
                    min_t = min_t.replace(tzinfo=timezone.utc)
                if max_t.tzinfo is None:
                    max_t = max_t.replace(tzinfo=timezone.utc)

                cur_min, cur_max = symbol_ranges[ccxt_symbol]
                symbol_ranges[ccxt_symbol] = (
                    min(cur_min, min_t),
                    max(cur_max, max_t),
                )
    finally:
        db.close()

    return symbol_ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill crypto context data for backtesting"
    )
    parser.add_argument(
        "--db",
        default="data/backtest_crypto.duckdb",
        help="DuckDB database path (default: data/backtest_crypto.duckdb)",
    )
    parser.add_argument(
        "--exchange",
        default="",
        help="Force a specific exchange (binanceus, kraken). Auto-detects if not set.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Extra lookback days before data range for volatility calculation (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without writing to DB",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print("=" * 60)
    print("  CRYPTO CONTEXT BACKFILL")
    print(f"  Database: {db_path}")
    print("=" * 60)

    # Step 1: Discover required symbols and date ranges
    print("\n[1/4] Discovering required symbols from orderbook_snapshots...")
    symbol_ranges = discover_symbols(db_path)

    if not symbol_ranges:
        print("  No crypto tickers found in database.")
        sys.exit(0)

    for symbol, (min_t, max_t) in sorted(symbol_ranges.items()):
        print(f"  {symbol:12s}  {min_t.strftime('%Y-%m-%d %H:%M')} -> {max_t.strftime('%Y-%m-%d %H:%M')}")

    # Step 2: Initialize exchange
    print("\n[2/4] Connecting to exchange...")
    import ccxt

    exchange = None
    exchange_name = ""

    if args.exchange:
        exchanges_to_try = [args.exchange]
    else:
        exchanges_to_try = EXCHANGE_PRIORITY

    for exch_id in exchanges_to_try:
        try:
            exchange = create_exchange(exch_id)
            exchange.load_markets()
            exchange_name = exch_id
            print(f"  Connected to {exch_id} ({len(exchange.markets)} markets)")
            break
        except Exception as e:
            print(f"  {exch_id} failed: {e}")
            exchange = None

    if exchange is None:
        print("ERROR: Could not connect to any exchange")
        sys.exit(1)

    # Step 3: Fetch OHLCV and compute snapshots
    print(f"\n[3/4] Fetching OHLCV data (lookback={args.lookback_days}d)...")
    lookback = timedelta(days=args.lookback_days)

    all_rows: list[tuple[str, datetime, str]] = []

    for symbol, (min_t, max_t) in sorted(symbol_ranges.items()):
        # Adjust symbol for the exchange
        fetch_symbol = symbol
        if exchange_name == "kraken" and symbol in KRAKEN_SYMBOL_MAP:
            fetch_symbol = KRAKEN_SYMBOL_MAP[symbol]

        # Check if the exchange has this symbol
        if fetch_symbol not in exchange.markets:
            # Try alternate symbol forms
            alt_symbol = symbol.replace("/USDT", "/USD")
            if alt_symbol in exchange.markets:
                fetch_symbol = alt_symbol
            else:
                print(f"  SKIP: {symbol} ({fetch_symbol}) not available on {exchange_name}")
                continue

        # Fetch range: lookback before min_t to max_t
        fetch_start = min_t - lookback
        fetch_end = max_t + timedelta(hours=1)  # Include the last hour

        start_ms = int(fetch_start.timestamp() * 1000)
        end_ms = int(fetch_end.timestamp() * 1000)

        print(f"\n  Fetching {fetch_symbol}...")
        print(f"    Range: {fetch_start.strftime('%Y-%m-%d %H:%M')} -> {fetch_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"    ({(fetch_end - fetch_start).days} days, ~{(fetch_end - fetch_start).days * 24} candles)")

        candles = fetch_ohlcv(exchange, fetch_symbol, start_ms, end_ms)
        print(f"    Fetched {len(candles)} candles")

        if len(candles) < 25:
            print(f"    WARN: Not enough candles for {symbol}, skipping")
            continue

        # Compute snapshots for each candle that falls within the backtest range
        min_ts_ms = int(min_t.timestamp() * 1000)
        max_ts_ms = int(max_t.timestamp() * 1000)
        n_snapshots = 0

        for idx in range(len(candles)):
            ts_ms = candles[idx][0]

            # Only store snapshots within the backtest date range
            # (but compute using all lookback data)
            if ts_ms < min_ts_ms - 3600 * 1000:
                continue
            if ts_ms > max_ts_ms + 3600 * 1000:
                break

            snapshot = build_snapshot(candles, idx, symbol, exchange_name)
            if snapshot is None:
                continue

            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            # Convert vol_history list to JSON-safe format
            snapshot_json = json.dumps(snapshot, default=str)

            all_rows.append((symbol, ts, snapshot_json))
            n_snapshots += 1

        print(f"    Computed {n_snapshots} context snapshots")

    # Step 4: Write to DB
    if args.dry_run:
        print(f"\n[4/4] DRY RUN: Would insert {len(all_rows)} rows into crypto_context")
        for symbol, ts, _ in all_rows[:5]:
            print(f"  {symbol} @ {ts}")
        if len(all_rows) > 5:
            print(f"  ... and {len(all_rows) - 5} more")
        return

    print(f"\n[4/4] Writing {len(all_rows)} rows to crypto_context table...")

    db = duckdb.connect(str(db_path))
    try:
        create_crypto_context_table(db)
        inserted = insert_snapshots(db, all_rows)
        print(f"  Wrote {inserted} rows to crypto_context")

        # Verify
        count = db.execute("SELECT COUNT(*) FROM crypto_context").fetchone()[0]
        symbols = db.execute("SELECT DISTINCT symbol FROM crypto_context").fetchall()
        ts_range = db.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM crypto_context"
        ).fetchone()

        print(f"\n  crypto_context table:")
        print(f"    Total rows:  {count}")
        print(f"    Symbols:     {[s[0] for s in symbols]}")
        print(f"    Time range:  {ts_range[0]} -> {ts_range[1]}")

        # Show a sample snapshot
        sample = db.execute(
            "SELECT symbol, timestamp, snapshot_json "
            "FROM crypto_context ORDER BY timestamp LIMIT 1"
        ).fetchone()
        if sample:
            snap = json.loads(sample[2])
            print(f"\n  Sample snapshot ({sample[0]} @ {sample[1]}):")
            for key in ["brti_price", "realized_vol_24h", "realized_vol_7d",
                        "implied_vol", "atr_14", "trend_regime", "trend_strength"]:
                val = snap.get(key)
                if isinstance(val, float):
                    print(f"    {key:20s} = {val:.6f}")
                else:
                    print(f"    {key:20s} = {val}")

    finally:
        db.close()

    print(f"\n{'=' * 60}")
    print("  BACKFILL COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
