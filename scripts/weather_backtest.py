#!/usr/bin/env python3
"""Weather model validator & ensemble archiver.

Three modes:
  1. validate  — Fetch recently settled Kalshi weather markets, pull the
                 ensemble forecast that was available before settlement,
                 run the weather model, and compare prediction vs outcome.
  2. archive   — Fetch current ensemble data for all configured locations
                 and persist to DuckDB for future backtesting.
  3. backfill  — Pull the maximum available historical ensemble data from
                 Open-Meteo (last ~3 days) and store it.

Usage::

    # Validate model against last N days of settled weather markets
    python scripts/weather_backtest.py validate --days 3

    # Archive current ensembles (run on a cron every 6h)
    python scripts/weather_backtest.py archive

    # Backfill max available history
    python scripts/weather_backtest.py backfill
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.weather.noaa import ForecastEnsemble, NOAAEnsembleFetcher
from moneygone.data.weather.ecmwf import ECMWFOpenDataFetcher
from moneygone.exchange.rest_client import KalshiRestClient
from moneygone.exchange.types import Market, MarketResult
from moneygone.models.weather_ensemble import WeatherEnsembleModel, bias_corrected_exceedance
from moneygone.data.store import DataStore
from moneygone.data.schemas import ALL_TABLES
from moneygone.utils.logging import setup_logging

log = structlog.get_logger("weather_backtest")

# ---------------------------------------------------------------------------
# Location config (same as stress-test.yaml)
# ---------------------------------------------------------------------------

LOCATIONS = [
    # Coordinates are NWS station locations used for Kalshi settlement,
    # NOT city centers.  Kalshi settles from NWS Daily Climatological Reports.
    {"name": "New York", "lat": 40.7668, "lon": -73.9829, "aliases": ["nyc", "ny"], "station": "KNYC"},   # Central Park
    {"name": "Chicago", "lat": 41.7856, "lon": -87.7527, "aliases": ["chi"], "station": "KMDW"},          # Midway Airport
    {"name": "Los Angeles", "lat": 33.9425, "lon": -118.4080, "aliases": ["lax", "la"], "station": "KLAX"},# LAX Airport
    {"name": "Miami", "lat": 25.7954, "lon": -80.2901, "aliases": ["mia"], "station": "KMIA"},            # MIA Airport
    {"name": "Dallas", "lat": 32.8972, "lon": -97.0377, "aliases": ["dal"], "station": "KDFW"},           # DFW Airport
    {"name": "Denver", "lat": 39.8617, "lon": -104.6732, "aliases": ["den"], "station": "KDEN"},          # DEN Airport
    {"name": "Seattle", "lat": 47.4499, "lon": -122.3118, "aliases": ["sea"], "station": "KSEA"},         # SeaTac Airport
    {"name": "Atlanta", "lat": 33.6367, "lon": -84.4279, "aliases": ["atl"], "station": "KATL"},          # ATL Airport
    {"name": "Houston", "lat": 29.6458, "lon": -95.2772, "aliases": ["hou"], "station": "KHOU"},          # Hobby Airport
    {"name": "Phoenix", "lat": 33.4343, "lon": -112.0116, "aliases": ["phx"], "station": "KPHX"},         # PHX Airport
    {"name": "Minneapolis", "lat": 44.8820, "lon": -93.2218, "aliases": ["min", "msp"], "station": "KMSP"},# MSP Airport
    {"name": "Oklahoma City", "lat": 35.3931, "lon": -97.6008, "aliases": ["okc"], "station": "KOKC"},    # Will Rogers Airport
    {"name": "New Orleans", "lat": 29.9933, "lon": -90.2590, "aliases": ["nola"], "station": "KMSY"},     # MSY Airport
    {"name": "Las Vegas", "lat": 36.0803, "lon": -115.1524, "aliases": ["lv", "vegas"], "station": "KLAS"},# LAS Airport
    {"name": "Washington DC", "lat": 38.8514, "lon": -77.0377, "aliases": ["dc", "wsh"], "station": "KDCA"},# Reagan Airport
    {"name": "Austin", "lat": 30.1945, "lon": -97.6699, "aliases": ["aus"], "station": "KAUS"},           # AUS Airport
]

# Kalshi weather series prefixes
_WEATHER_SERIES = [
    "KXHIGH", "KXLOW", "KXHIGHT", "KXLOWT", "KXRAIN",
]

# Parse threshold from ticker: e.g. KXHIGHNY-26APR09-T60 -> 60.0
_THRESHOLD_RE = re.compile(r"-[TB]([\d.]+)$")

# Parse location code from ticker: KXHIGHNY -> NY, KXLOWTMIA -> MIA
_LOCATION_CODE_RE = re.compile(
    r"^KX(?:HIGH|LOW)T?(.+?)-\d"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_location(ticker: str, title: str) -> dict | None:
    """Match a weather market to a configured location."""
    m = _LOCATION_CODE_RE.match(ticker.upper())
    if not m:
        return None
    code = m.group(1).lower()

    text = (title + " " + ticker).lower()

    for loc in LOCATIONS:
        name_lower = loc["name"].lower()
        # Check code match
        aliases = [a.lower() for a in loc.get("aliases", [])]
        all_names = [name_lower.replace(" ", "")] + aliases
        if code in all_names:
            return loc
        # Check title match
        if name_lower in text or any(a in text for a in aliases):
            return loc
    return None


def _parse_threshold_and_direction(
    market: "Market",
) -> tuple[float | None, float | None]:
    """Extract threshold (°F) and direction from market using API fields.

    Returns (threshold_f, direction) where direction is:
      +1.0 = YES means above threshold
      -1.0 = YES means below threshold
    """
    ticker_upper = market.ticker.upper()
    last_segment = ticker_upper.split("-")[-1]

    if last_segment.startswith("B"):
        # Bracket market: skip for now
        return None, None

    if not last_segment.startswith("T"):
        return None, None

    # Get threshold from cap_strike or floor_strike
    if market.floor_strike is not None:
        threshold_f = float(market.floor_strike)
    elif market.cap_strike is not None:
        threshold_f = float(market.cap_strike)
    else:
        # Fallback: parse from ticker
        m = _THRESHOLD_RE.search(market.ticker)
        if not m:
            return None, None
        threshold_f = float(m.group(1))

    # Direction from strike_type
    strike = market.strike_type.lower() if market.strike_type else ""
    if strike in ("greater", "greater_or_equal"):
        direction = 1.0   # YES = above threshold
    elif strike in ("less", "less_or_equal"):
        direction = -1.0  # YES = below threshold
    else:
        # Fallback: guess from title
        title_lower = market.title.lower() if market.title else ""
        if ">" in title_lower or "above" in title_lower or "or above" in title_lower:
            direction = 1.0
        elif "<" in title_lower or "below" in title_lower or "or below" in title_lower:
            direction = -1.0
        else:
            direction = 1.0  # default assume above

    return threshold_f, direction


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


@dataclass
class ValidationResult:
    ticker: str
    location: str
    threshold_f: float
    threshold_c: float
    direction: float  # 1.0 = above
    actual_result: str  # "yes" or "no"
    model_prob: float
    raw_prob: float  # raw exceedance
    confidence: float
    ensemble_spread: float
    ensemble_mean: float
    n_members: int
    forecast_horizon_h: float
    market_last_price: float
    correct: bool  # model agreed with outcome


# ---------------------------------------------------------------------------
# Validate: test model against settled markets
# ---------------------------------------------------------------------------

async def validate(
    days: int = 3,
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
) -> list[ValidationResult]:
    """Fetch settled weather markets and validate the ensemble model."""

    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )

    client = KalshiRestClient(config.exchange)
    noaa = NOAAEnsembleFetcher()
    ecmwf = ECMWFOpenDataFetcher()
    model = WeatherEnsembleModel()

    results: list[ValidationResult] = []
    seen_tickers: set[str] = set()

    # Build location-specific series tickers to query
    series_tickers: list[str] = []
    for loc in LOCATIONS:
        code = loc.get("aliases", [loc["name"].replace(" ", "").lower()])[0].upper()
        for prefix in ["KXHIGH", "KXHIGHT", "KXLOW", "KXLOWT"]:
            series_tickers.append(f"{prefix}{code}")

    # Fetch settled weather markets per series ticker
    log.info("validate.fetching_settled", days=days, n_series=len(series_tickers))
    settled_markets: list[Market] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days + 1)

    for series in series_tickers:
        try:
            batch = await client.get_all_markets(
                series_ticker=series,
                status="settled",
                max_pages=5,
            )
        except Exception as e:
            log.debug("validate.series_fetch_failed", series=series, error=str(e))
            continue
        for m in batch:
            if m.close_time < cutoff:
                continue
            if m.result in (MarketResult.NOT_SETTLED, MarketResult.VOIDED):
                continue
            if m.ticker not in seen_tickers:
                seen_tickers.add(m.ticker)
                settled_markets.append(m)

    log.info("validate.found_settled", count=len(settled_markets))

    # Cache daily ensemble fetches per location+variable
    # Key: "{location}_{daily_var}" → dict of date_str → list[float] per member
    daily_cache: dict[str, dict[str, list[float]]] = {}

    async def _fetch_daily_ensemble(
        loc: dict, daily_var: str, forecast_days: int = 7, past_days: int = 5
    ) -> dict[str, list[float]]:
        """Fetch daily max/min ensemble for a location, cached.

        Returns {date_str: [member_0_val, member_1_val, ...]} in °C.
        Uses past_days to include recently-settled dates.
        """
        cache_key = f"{loc['name']}_{daily_var}"
        if cache_key in daily_cache:
            return daily_cache[cache_key]

        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "daily": daily_var,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "models": "gfs025",
            "timezone": "America/New_York",  # Kalshi weather settles on US local dates
        }
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble",
                params=params,
            )
            resp.raise_for_status()
            payload = resp.json()

        daily = payload.get("daily", {})
        dates: list[str] = daily.get("time", [])

        # Collect member columns
        member_cols: list[list[float]] = []
        for key in sorted(daily.keys()):
            if key.startswith(daily_var) and "member" in key:
                member_cols.append(
                    [float(v) if v is not None else float("nan") for v in daily[key]]
                )

        result: dict[str, list[float]] = {}
        for d_idx, date_str in enumerate(dates):
            vals = [
                mc[d_idx] for mc in member_cols
                if d_idx < len(mc) and mc[d_idx] == mc[d_idx]  # skip NaN
            ]
            if vals:
                result[date_str] = vals

        daily_cache[cache_key] = result
        return result

    for market in settled_markets:
        # Parse market
        loc = _match_location(market.ticker, market.title)
        if loc is None:
            continue

        threshold_f, direction = _parse_threshold_and_direction(market)
        if threshold_f is None or direction is None:
            continue

        threshold_c = _f_to_c(threshold_f)

        # Determine which daily variable based on ticker
        ticker_upper = market.ticker.upper()
        if "KXHIGH" in ticker_upper:
            daily_var = "temperature_2m_max"
        elif "KXLOW" in ticker_upper:
            daily_var = "temperature_2m_min"
        else:
            continue

        # Determine actual outcome
        actual = "yes" if market.result == MarketResult.YES else "no"

        # Fetch daily ensemble (with past_days to cover settled dates)
        try:
            daily_data = await _fetch_daily_ensemble(loc, daily_var, past_days=days + 2)
        except Exception as e:
            log.warning("validate.ensemble_fetch_failed", location=loc["name"], error=str(e))
            continue

        if not daily_data:
            continue

        # Find the date this market settles on from the close_time
        close_utc = market.close_time.astimezone(timezone.utc)
        # Market tickers contain the date: KXHIGHNY-26APR09 → 2026-04-09
        # The close_time minus a few hours gives the settlement date
        # Use the date from the ticker for exact matching
        ticker_date_match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker_upper)
        if ticker_date_match:
            yr = int("20" + ticker_date_match.group(1))
            mon_str = ticker_date_match.group(2)
            day = int(ticker_date_match.group(3))
            months = {m: i for i, m in enumerate(calendar.month_abbr) if m}
            mon = months.get(mon_str.title(), 0)
            if mon > 0:
                settle_date = f"{yr}-{mon:02d}-{day:02d}"
            else:
                settle_date = (close_utc - timedelta(hours=6)).strftime("%Y-%m-%d")
        else:
            settle_date = (close_utc - timedelta(hours=6)).strftime("%Y-%m-%d")

        values_at_date = daily_data.get(settle_date, [])
        if not values_at_date:
            # Try nearby dates
            log.debug("validate.no_ensemble_for_date", ticker=market.ticker, date=settle_date)
            continue

        if direction > 0:
            exceed_count = sum(1 for v in values_at_date if v > threshold_c)
        else:
            exceed_count = sum(1 for v in values_at_date if v < threshold_c)

        exceedance_prob = exceed_count / len(values_at_date)
        ens_mean = statistics.mean(values_at_date)
        ens_std = statistics.stdev(values_at_date) if len(values_at_date) > 1 else 0.0

        # Forecast horizon: difference between now and settlement date
        try:
            settle_dt = datetime.strptime(settle_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            horizon_h = max(0, (settle_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0)
        except ValueError:
            horizon_h = 0.0

        # Run model
        features = {
            "ensemble_exceedance_prob": exceedance_prob,
            "ensemble_mean": ens_mean,
            "ensemble_spread": ens_std,
            "model_disagreement": 0.0,  # No ECMWF comparison for speed
            "forecast_revision_magnitude": 0.0,
            "forecast_revision_direction": 0.0,
            "forecast_horizon": horizon_h,
            "climatological_anomaly": 0.0,
        }
        prediction = model.predict_proba(features)

        # Did the model agree with the outcome?
        model_says_yes = prediction.probability > 0.5
        actual_yes = actual == "yes"
        correct = model_says_yes == actual_yes

        result = ValidationResult(
            ticker=market.ticker,
            location=loc["name"],
            threshold_f=threshold_f,
            threshold_c=threshold_c,
            direction=direction,
            actual_result=actual,
            model_prob=prediction.probability,
            raw_prob=exceedance_prob,
            confidence=prediction.confidence,
            ensemble_spread=ens_std,
            ensemble_mean=ens_mean,
            n_members=len(values_at_date),
            forecast_horizon_h=horizon_h,
            market_last_price=float(market.last_price),
            correct=correct,
        )
        results.append(result)

    await client.close()
    await noaa.close()
    await ecmwf.close()

    return results


def print_validation_report(results: list[ValidationResult]) -> None:
    """Print a human-readable validation report."""
    if not results:
        print("\nNo results to report.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 80)
    print(f"WEATHER MODEL VALIDATION REPORT")
    print(f"=" * 80)
    print(f"Markets tested: {total}")
    print(f"Correct calls:  {correct}/{total} ({accuracy:.1%})")
    print()

    # Calibration: bucket by model probability
    buckets: dict[str, list[ValidationResult]] = defaultdict(list)
    for r in results:
        if r.model_prob < 0.2:
            buckets["0-20%"].append(r)
        elif r.model_prob < 0.4:
            buckets["20-40%"].append(r)
        elif r.model_prob < 0.6:
            buckets["40-60%"].append(r)
        elif r.model_prob < 0.8:
            buckets["60-80%"].append(r)
        else:
            buckets["80-100%"].append(r)

    print("CALIBRATION (model prob vs actual outcome rate):")
    print(f"  {'Bucket':<12} {'Count':>6} {'Actual YES%':>12} {'Avg Model P':>12} {'Calibration':>12}")
    for bucket_name in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
        items = buckets.get(bucket_name, [])
        if not items:
            continue
        actual_yes_rate = sum(1 for r in items if r.actual_result == "yes") / len(items)
        avg_model_p = statistics.mean(r.model_prob for r in items)
        cal_error = abs(actual_yes_rate - avg_model_p)
        print(f"  {bucket_name:<12} {len(items):>6} {actual_yes_rate:>11.1%} {avg_model_p:>11.3f} {cal_error:>11.3f}")

    # By location
    print("\nBY LOCATION:")
    loc_results: dict[str, list[ValidationResult]] = defaultdict(list)
    for r in results:
        loc_results[r.location].append(r)
    print(f"  {'Location':<18} {'Count':>6} {'Accuracy':>10} {'Avg Spread':>11} {'Members':>8}")
    for loc_name in sorted(loc_results.keys()):
        items = loc_results[loc_name]
        loc_acc = sum(1 for r in items if r.correct) / len(items)
        avg_spread = statistics.mean(r.ensemble_spread for r in items)
        avg_members = statistics.mean(r.n_members for r in items)
        print(f"  {loc_name:<18} {len(items):>6} {loc_acc:>9.1%} {avg_spread:>10.2f} {avg_members:>7.0f}")

    # Confident wrong calls (high confidence but wrong)
    wrong_confident = [
        r for r in results
        if not r.correct and r.confidence > 0.5
        and abs(r.model_prob - 0.5) > 0.2
    ]
    if wrong_confident:
        print(f"\nCONFIDENT WRONG CALLS ({len(wrong_confident)}):")
        for r in sorted(wrong_confident, key=lambda x: -abs(x.model_prob - 0.5)):
            print(
                f"  {r.ticker:<45} model={r.model_prob:.3f} actual={r.actual_result}"
                f"  raw_exc={r.raw_prob:.2f} spread={r.ensemble_spread:.2f}"
                f"  conf={r.confidence:.2f} horizon={r.forecast_horizon_h:.0f}h"
            )

    # Edge analysis: where model disagreed with market
    print("\nEDGE ANALYSIS (model vs last market price):")
    profitable_count = 0
    total_edge_markets = 0
    for r in results:
        market_prob = r.market_last_price
        if market_prob <= 0 or market_prob >= 1:
            continue
        model_edge = r.model_prob - market_prob
        if abs(model_edge) < 0.05:
            continue  # Skip tiny edges
        total_edge_markets += 1
        model_side = "yes" if r.model_prob > market_prob else "no"
        if (model_side == "yes" and r.actual_result == "yes") or \
           (model_side == "no" and r.actual_result == "no"):
            profitable_count += 1

    if total_edge_markets > 0:
        edge_accuracy = profitable_count / total_edge_markets
        print(f"  Markets with >5% model-vs-market edge: {total_edge_markets}")
        print(f"  Model's edge calls correct: {profitable_count}/{total_edge_markets} ({edge_accuracy:.1%})")
    else:
        print("  No markets with significant model-vs-market edge found.")

    print()


# ---------------------------------------------------------------------------
# Archive: save current ensemble data
# ---------------------------------------------------------------------------

async def archive_ensembles(
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
    data_dir: str | None = None,
) -> int:
    """Fetch and archive current ensemble data for all locations."""
    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )

    if data_dir is None:
        data_dir_path = Path(config.data_dir)
    else:
        data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    store = DataStore(data_dir_path / "weather_archive.duckdb")
    store.initialize_schema([
        """
        CREATE TABLE IF NOT EXISTS ensemble_archive (
            location_name VARCHAR NOT NULL,
            lat           DOUBLE NOT NULL,
            lon           DOUBLE NOT NULL,
            variable      VARCHAR NOT NULL,
            model_source  VARCHAR NOT NULL,
            init_time     TIMESTAMP NOT NULL,
            valid_time    TIMESTAMP NOT NULL,
            member_values JSON NOT NULL,
            ensemble_mean DOUBLE NOT NULL,
            ensemble_std  DOUBLE NOT NULL,
            n_members     INTEGER NOT NULL,
            ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS validation_results (
            ticker          VARCHAR NOT NULL,
            location_name   VARCHAR NOT NULL,
            threshold_f     DOUBLE NOT NULL,
            threshold_c     DOUBLE NOT NULL,
            direction       DOUBLE NOT NULL,
            actual_result   VARCHAR NOT NULL,
            model_prob      DOUBLE NOT NULL,
            raw_exceedance  DOUBLE NOT NULL,
            confidence      DOUBLE NOT NULL,
            ensemble_spread DOUBLE NOT NULL,
            ensemble_mean   DOUBLE NOT NULL,
            n_members       INTEGER NOT NULL,
            forecast_horizon_h DOUBLE NOT NULL,
            market_last_price  DOUBLE NOT NULL,
            correct         BOOLEAN NOT NULL,
            validated_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
        """,
    ])

    noaa = NOAAEnsembleFetcher()
    ecmwf = ECMWFOpenDataFetcher()
    total_rows = 0

    for loc in LOCATIONS:
        for variable in ["temperature_2m", "precipitation"]:
            for source_name, fetcher in [("noaa_gefs", noaa), ("ecmwf_ifs", ecmwf)]:
                try:
                    ensemble = await fetcher.fetch_ensemble(
                        lat=loc["lat"],
                        lon=loc["lon"],
                        variable=variable,
                        location_name=loc["name"],
                        forecast_days=7,
                    )
                except Exception as e:
                    log.warning(
                        "archive.fetch_failed",
                        location=loc["name"],
                        variable=variable,
                        source=source_name,
                        error=str(e),
                    )
                    continue

                # Insert one row per valid_time
                rows = []
                for t_idx, vt in enumerate(ensemble.valid_times):
                    values_at_t = [
                        m[t_idx]
                        for m in ensemble.member_values
                        if t_idx < len(m)
                    ]
                    if not values_at_t:
                        continue
                    rows.append((
                        loc["name"],
                        loc["lat"],
                        loc["lon"],
                        variable,
                        source_name,
                        ensemble.init_time,
                        vt,
                        json.dumps(values_at_t),
                        statistics.mean(values_at_t),
                        statistics.stdev(values_at_t) if len(values_at_t) > 1 else 0.0,
                        len(values_at_t),
                    ))

                if rows:
                    store._conn.executemany(
                        """
                        INSERT INTO ensemble_archive
                            (location_name, lat, lon, variable, model_source,
                             init_time, valid_time, member_values,
                             ensemble_mean, ensemble_std, n_members)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    total_rows += len(rows)
                    log.info(
                        "archive.stored",
                        location=loc["name"],
                        variable=variable,
                        source=source_name,
                        rows=len(rows),
                        members=len(ensemble.member_values),
                    )

                # Rate limit: be gentle with free API
                await asyncio.sleep(0.5)

    await noaa.close()
    await ecmwf.close()
    store.close()

    log.info("archive.complete", total_rows=total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# Backfill: pull max available history from Open-Meteo
# ---------------------------------------------------------------------------

async def backfill_ensembles(
    past_days: int = 5,
    data_dir: str | None = None,
) -> int:
    """Backfill historical ensemble data using past_days parameter."""

    if data_dir is None:
        data_dir_path = Path("data")
    else:
        data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    store = DataStore(data_dir_path / "weather_archive.duckdb")
    store.initialize_schema([
        """
        CREATE TABLE IF NOT EXISTS ensemble_archive (
            location_name VARCHAR NOT NULL,
            lat           DOUBLE NOT NULL,
            lon           DOUBLE NOT NULL,
            variable      VARCHAR NOT NULL,
            model_source  VARCHAR NOT NULL,
            init_time     TIMESTAMP NOT NULL,
            valid_time    TIMESTAMP NOT NULL,
            member_values JSON NOT NULL,
            ensemble_mean DOUBLE NOT NULL,
            ensemble_std  DOUBLE NOT NULL,
            n_members     INTEGER NOT NULL,
            ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
        """,
    ])

    total_rows = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for loc in LOCATIONS:
            for variable in ["temperature_2m", "precipitation"]:
                om_var = variable

                # NOAA GEFS via Open-Meteo with past_days
                params = {
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                    "hourly": om_var,
                    "past_days": past_days,
                    "forecast_days": 7,
                    "models": "gfs025",
                }

                try:
                    resp = await client.get(
                        "https://ensemble-api.open-meteo.com/v1/ensemble",
                        params=params,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as e:
                    log.warning(
                        "backfill.fetch_failed",
                        location=loc["name"],
                        variable=variable,
                        error=str(e),
                    )
                    continue

                hourly = payload.get("hourly", {})
                time_strings = hourly.get("time", [])

                valid_times = [
                    datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                    for t in time_strings
                ]

                # Collect member columns
                member_columns: list[list[float]] = []
                for key in sorted(hourly.keys()):
                    if key.startswith(om_var) and "member" in key:
                        values = hourly[key]
                        member_columns.append(
                            [float(v) if v is not None else float("nan") for v in values]
                        )

                if not member_columns:
                    continue

                init_time = valid_times[0] if valid_times else datetime.now(tz=timezone.utc)

                rows_inserted = 0
                for t_idx, vt in enumerate(valid_times):
                    values_at_t = []
                    for m in member_columns:
                        if t_idx < len(m):
                            v = m[t_idx]
                            if v == v:  # not NaN
                                values_at_t.append(v)

                    if not values_at_t:
                        continue

                    store._conn.execute(
                        """
                        INSERT INTO ensemble_archive
                            (location_name, lat, lon, variable, model_source,
                             init_time, valid_time, member_values,
                             ensemble_mean, ensemble_std, n_members)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            loc["name"],
                            loc["lat"],
                            loc["lon"],
                            variable,
                            "noaa_gefs",
                            init_time,
                            vt,
                            json.dumps(values_at_t),
                            statistics.mean(values_at_t),
                            statistics.stdev(values_at_t) if len(values_at_t) > 1 else 0.0,
                            len(values_at_t),
                        ),
                    )
                    rows_inserted += 1

                total_rows += rows_inserted
                log.info(
                    "backfill.stored",
                    location=loc["name"],
                    variable=variable,
                    rows=rows_inserted,
                    members=len(member_columns),
                    past_days=past_days,
                )

                # Rate limit
                await asyncio.sleep(0.3)

    store.close()
    log.info("backfill.complete", total_rows=total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# Save validation results
# ---------------------------------------------------------------------------

def save_validation_results(
    results: list[ValidationResult],
    data_dir: str | None = None,
) -> None:
    """Persist validation results to DuckDB for tracking over time."""
    if not results:
        return

    if data_dir is None:
        data_dir_path = Path("data")
    else:
        data_dir_path = Path(data_dir)

    store = DataStore(data_dir_path / "weather_archive.duckdb")
    store.initialize_schema([
        """
        CREATE TABLE IF NOT EXISTS validation_results (
            ticker          VARCHAR NOT NULL,
            location_name   VARCHAR NOT NULL,
            threshold_f     DOUBLE NOT NULL,
            threshold_c     DOUBLE NOT NULL,
            direction       DOUBLE NOT NULL,
            actual_result   VARCHAR NOT NULL,
            model_prob      DOUBLE NOT NULL,
            raw_exceedance  DOUBLE NOT NULL,
            confidence      DOUBLE NOT NULL,
            ensemble_spread DOUBLE NOT NULL,
            ensemble_mean   DOUBLE NOT NULL,
            n_members       INTEGER NOT NULL,
            forecast_horizon_h DOUBLE NOT NULL,
            market_last_price  DOUBLE NOT NULL,
            correct         BOOLEAN NOT NULL,
            validated_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
        """,
    ])

    for r in results:
        store._conn.execute(
            """
            INSERT INTO validation_results
                (ticker, location_name, threshold_f, threshold_c, direction,
                 actual_result, model_prob, raw_exceedance, confidence,
                 ensemble_spread, ensemble_mean, n_members,
                 forecast_horizon_h, market_last_price, correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.ticker, r.location, r.threshold_f, r.threshold_c,
                r.direction, r.actual_result, r.model_prob, r.raw_prob,
                r.confidence, r.ensemble_spread, r.ensemble_mean,
                r.n_members, r.forecast_horizon_h, r.market_last_price,
                r.correct,
            ),
        )

    store.close()
    log.info("validation.results_saved", count=len(results))


# ---------------------------------------------------------------------------
# Bias analysis: compare ensemble to NWS station observations
# ---------------------------------------------------------------------------

# Station mapping: location name → NWS station ID
_LOCATION_STATION: dict[str, str] = {
    loc["name"]: loc["station"] for loc in LOCATIONS
}


@dataclass
class BiasResult:
    station: str
    location: str
    variable: str  # "high" or "low"
    n_days: int
    mean_bias_f: float   # ensemble - observed (°F), positive = warm bias
    mean_bias_c: float
    std_bias_f: float
    rmse_f: float
    daily_biases: list[tuple[str, float, float, float]]  # (date, obs_f, ens_f, bias_f)


async def analyze_bias(days: int = 30) -> list[BiasResult]:
    """Compare ensemble daily max/min forecasts to NWS CLI observations.

    Fetches:
    1. NWS CLI observations (actual high/low) for each station
    2. Open-Meteo ensemble daily max/min (hindcast via past_days)
    3. Computes per-station, per-variable bias statistics
    """
    from moneygone.data.weather.nws_observations import NWSObservationFetcher

    nws = NWSObservationFetcher()
    end_date = date.today() - timedelta(days=1)  # yesterday (latest CLI available)
    start_date = end_date - timedelta(days=days - 1)

    results: list[BiasResult] = []

    for loc in LOCATIONS:
        station = loc.get("station", "")
        if not station:
            continue

        # Fetch NWS observations
        try:
            observations = await nws.fetch_station_range(station, start_date, end_date)
        except Exception as e:
            log.warning("bias.nws_fetch_failed", station=station, error=str(e))
            continue

        obs_by_date: dict[str, tuple[int | None, int | None]] = {}
        for obs in observations:
            obs_by_date[obs.valid_date.isoformat()] = (obs.high_f, obs.low_f)

        if not obs_by_date:
            log.warning("bias.no_observations", station=station)
            continue

        # Fetch ensemble hindcast for matching dates
        past_days_needed = (date.today() - start_date).days + 1
        for var_name, daily_var in [("high", "temperature_2m_max"), ("low", "temperature_2m_min")]:
            try:
                async with httpx.AsyncClient(timeout=60.0) as http:
                    resp = await http.get(
                        "https://ensemble-api.open-meteo.com/v1/ensemble",
                        params={
                            "latitude": loc["lat"],
                            "longitude": loc["lon"],
                            "daily": daily_var,
                            "past_days": min(past_days_needed, 92),
                            "forecast_days": 1,
                            "models": "gfs025",
                            "timezone": "America/New_York",
                        },
                    )
                    resp.raise_for_status()
                    payload = resp.json()
            except Exception as e:
                log.warning("bias.ensemble_fetch_failed", location=loc["name"], var=var_name, error=str(e))
                continue

            daily = payload.get("daily", {})
            ens_dates = daily.get("time", [])

            # Get ensemble mean per date
            member_cols: list[list[float]] = []
            for key in sorted(daily.keys()):
                if key.startswith(daily_var) and "member" in key:
                    member_cols.append(daily[key])

            ens_by_date: dict[str, float] = {}
            for d_idx, d_str in enumerate(ens_dates):
                vals = []
                for mc in member_cols:
                    if d_idx < len(mc) and mc[d_idx] is not None:
                        v = mc[d_idx]
                        if v == v:  # not NaN
                            vals.append(v)
                if vals:
                    # Ensemble mean in °C, convert to °F
                    mean_c = sum(vals) / len(vals)
                    mean_f = mean_c * 9.0 / 5.0 + 32.0
                    ens_by_date[d_str] = mean_f

            # Compare
            daily_biases: list[tuple[str, float, float, float]] = []
            for d_str in sorted(obs_by_date.keys()):
                obs_val = obs_by_date[d_str][0] if var_name == "high" else obs_by_date[d_str][1]
                if obs_val is None or d_str not in ens_by_date:
                    continue
                ens_val = ens_by_date[d_str]
                bias = ens_val - obs_val
                daily_biases.append((d_str, float(obs_val), ens_val, bias))

            if len(daily_biases) < 3:
                continue

            biases = [b[3] for b in daily_biases]
            mean_bias = sum(biases) / len(biases)
            variance = sum((b - mean_bias) ** 2 for b in biases) / len(biases)
            std_bias = variance ** 0.5
            rmse = (sum(b ** 2 for b in biases) / len(biases)) ** 0.5

            results.append(BiasResult(
                station=station,
                location=loc["name"],
                variable=var_name,
                n_days=len(daily_biases),
                mean_bias_f=mean_bias,
                mean_bias_c=mean_bias * 5.0 / 9.0,
                std_bias_f=std_bias,
                rmse_f=rmse,
                daily_biases=daily_biases,
            ))

            log.info(
                "bias.computed",
                station=station,
                location=loc["name"],
                variable=var_name,
                n_days=len(daily_biases),
                mean_bias_f=round(mean_bias, 2),
                rmse_f=round(rmse, 2),
            )

        await asyncio.sleep(0.3)

    await nws.close()
    return results


def print_bias_report(results: list[BiasResult]) -> None:
    """Print bias analysis report."""
    if not results:
        print("\nNo bias results.")
        return

    print("\n" + "=" * 90)
    print("ENSEMBLE vs NWS STATION BIAS ANALYSIS")
    print("=" * 90)
    print("Positive bias = ensemble warmer than observed station")
    print()
    print(
        "  %-18s %-8s %-6s %8s %8s %8s %8s"
        % ("Location", "Station", "Var", "N Days", "Bias°F", "Std°F", "RMSE°F")
    )
    print("  " + "-" * 76)

    for r in sorted(results, key=lambda x: (x.location, x.variable)):
        print(
            "  %-18s %-8s %-6s %8d %+7.1f %7.1f %7.1f"
            % (r.location, r.station, r.variable, r.n_days,
               r.mean_bias_f, r.std_bias_f, r.rmse_f)
        )

    # Summary
    high_results = [r for r in results if r.variable == "high"]
    low_results = [r for r in results if r.variable == "low"]

    if high_results:
        avg_high_bias = sum(r.mean_bias_f for r in high_results) / len(high_results)
        avg_high_rmse = sum(r.rmse_f for r in high_results) / len(high_results)
        print(
            "\n  AVERAGE HIGH BIAS: %+.1f°F  RMSE: %.1f°F  (across %d stations)"
            % (avg_high_bias, avg_high_rmse, len(high_results))
        )

    if low_results:
        avg_low_bias = sum(r.mean_bias_f for r in low_results) / len(low_results)
        avg_low_rmse = sum(r.rmse_f for r in low_results) / len(low_results)
        print(
            "  AVERAGE LOW BIAS:  %+.1f°F  RMSE: %.1f°F  (across %d stations)"
            % (avg_low_bias, avg_low_rmse, len(low_results))
        )

    # Per-station bias table for model config
    print("\n  BIAS CORRECTIONS (for model config):")
    print("  Station biases to subtract from ensemble forecast before exceedance calc:")
    for loc in LOCATIONS:
        loc_results = [r for r in results if r.location == loc["name"]]
        if not loc_results:
            continue
        high_bias = next((r.mean_bias_f for r in loc_results if r.variable == "high"), 0.0)
        low_bias = next((r.mean_bias_f for r in loc_results if r.variable == "low"), 0.0)
        high_rmse = next((r.rmse_f for r in loc_results if r.variable == "high"), 0.0)
        low_rmse = next((r.rmse_f for r in loc_results if r.variable == "low"), 0.0)
        print(
            '    "%s": {"high_bias_f": %.1f, "low_bias_f": %.1f, "high_rmse_f": %.1f, "low_rmse_f": %.1f}'
            % (loc["name"], high_bias, low_bias, high_rmse, low_rmse)
        )

    print()


# ---------------------------------------------------------------------------
# Corrected validation: apply bias correction before exceedance calc
# ---------------------------------------------------------------------------

async def validate_corrected(
    days: int = 3,
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
    bias_days: int = 30,
) -> tuple[list[ValidationResult], list[ValidationResult]]:
    """Run validation with and without bias correction.

    Returns (uncorrected_results, corrected_results).
    """
    # First, compute bias
    print("Computing bias corrections from %d days of NWS data..." % bias_days)
    bias_results = await analyze_bias(days=bias_days)
    print_bias_report(bias_results)

    # Build bias lookup
    bias_lookup: dict[str, dict[str, float]] = {}  # location → {high_bias_f, low_bias_f, high_rmse_f, low_rmse_f}
    for r in bias_results:
        if r.location not in bias_lookup:
            bias_lookup[r.location] = {}
        bias_lookup[r.location][r.variable + "_bias_f"] = r.mean_bias_f
        bias_lookup[r.location][r.variable + "_rmse_f"] = r.rmse_f

    # Now run validation
    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )

    client = KalshiRestClient(config.exchange)
    model = WeatherEnsembleModel()

    uncorrected: list[ValidationResult] = []
    corrected: list[ValidationResult] = []
    seen_tickers: set[str] = set()

    # Build series tickers
    series_tickers: list[str] = []
    for loc in LOCATIONS:
        code = loc.get("aliases", [loc["name"].replace(" ", "").lower()])[0].upper()
        for prefix in ["KXHIGH", "KXHIGHT", "KXLOW", "KXLOWT"]:
            series_tickers.append(prefix + code)

    log.info("validate_corrected.fetching_settled", days=days, n_series=len(series_tickers))
    settled_markets: list[Market] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days + 1)

    for series in series_tickers:
        try:
            batch = await client.get_all_markets(
                series_ticker=series, status="settled", max_pages=5,
            )
        except Exception:
            continue
        for m in batch:
            if m.close_time < cutoff:
                continue
            if m.result in (MarketResult.NOT_SETTLED, MarketResult.VOIDED):
                continue
            if m.ticker not in seen_tickers:
                seen_tickers.add(m.ticker)
                settled_markets.append(m)

    log.info("validate_corrected.found_settled", count=len(settled_markets))

    # Ensemble cache
    daily_cache: dict[str, dict[str, list[float]]] = {}

    async def _fetch_daily(loc: dict, daily_var: str) -> dict[str, list[float]]:
        cache_key = "%s_%s" % (loc["name"], daily_var)
        if cache_key in daily_cache:
            return daily_cache[cache_key]
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "daily": daily_var, "past_days": days + 2, "forecast_days": 7,
            "models": "gfs025", "timezone": "America/New_York",
        }
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble", params=params,
            )
            resp.raise_for_status()
            payload = resp.json()
        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        member_cols = []
        for key in sorted(daily.keys()):
            if key.startswith(daily_var) and "member" in key:
                member_cols.append([float(v) if v is not None else float("nan") for v in daily[key]])
        result = {}
        for d_idx, d_str in enumerate(dates):
            vals = [mc[d_idx] for mc in member_cols if d_idx < len(mc) and mc[d_idx] == mc[d_idx]]
            if vals:
                result[d_str] = vals
        daily_cache[cache_key] = result
        return result

    for market in settled_markets:
        loc = _match_location(market.ticker, market.title)
        if loc is None:
            continue

        threshold_f, direction = _parse_threshold_and_direction(market)
        if threshold_f is None or direction is None:
            continue

        threshold_c = _f_to_c(threshold_f)
        ticker_upper = market.ticker.upper()

        if "KXHIGH" in ticker_upper:
            daily_var = "temperature_2m_max"
            var_name = "high"
        elif "KXLOW" in ticker_upper:
            daily_var = "temperature_2m_min"
            var_name = "low"
        else:
            continue

        actual = "yes" if market.result == MarketResult.YES else "no"

        try:
            daily_data = await _fetch_daily(loc, daily_var)
        except Exception:
            continue

        # Parse settle date from ticker
        ticker_date_match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker_upper)
        if ticker_date_match:
            yr = int("20" + ticker_date_match.group(1))
            mon_str = ticker_date_match.group(2)
            day_val = int(ticker_date_match.group(3))
            months = {m: i for i, m in enumerate(calendar.month_abbr) if m}
            mon = months.get(mon_str.title(), 0)
            settle_date = "%d-%02d-%02d" % (yr, mon, day_val) if mon > 0 else ""
        else:
            settle_date = ""

        if not settle_date:
            continue

        values_at_date = daily_data.get(settle_date, [])
        if not values_at_date:
            continue

        # Get bias for this location/variable
        loc_bias = bias_lookup.get(loc["name"], {})
        bias_f = loc_bias.get(var_name + "_bias_f", 0.0)
        rmse_f = loc_bias.get(var_name + "_rmse_f", 0.0)
        bias_c = bias_f * 5.0 / 9.0

        # Ensemble statistics
        ens_mean = sum(values_at_date) / len(values_at_date)
        ens_std = (sum((v - ens_mean) ** 2 for v in values_at_date) / len(values_at_date)) ** 0.5

        # ---- UNCORRECTED exceedance ----
        if direction > 0:
            unc_exceed = sum(1 for v in values_at_date if v > threshold_c)
        else:
            unc_exceed = sum(1 for v in values_at_date if v < threshold_c)
        unc_exceedance = unc_exceed / len(values_at_date)

        # ---- CORRECTED exceedance: Gaussian model using bias + RMSE ----
        # Model: T_station = T_ensemble_mean - bias + N(0, σ_residual)
        # This properly handles the case where all 30 members agree but
        # the station observation has large residual uncertainty.
        import math
        from scipy import stats  # type: ignore[import-untyped]

        corrected_mean_c = ens_mean - bias_c
        rmse_c = rmse_f * 5.0 / 9.0
        bias_abs_c = abs(bias_c)
        residual_std_c = math.sqrt(max(rmse_c ** 2 - bias_abs_c ** 2, 0.01))
        # Effective uncertainty: ensemble spread + residual + NWS rounding (0.5°F ≈ 0.28°C)
        effective_std = math.sqrt(ens_std ** 2 + residual_std_c ** 2 + 0.28 ** 2)

        if direction > 0:
            z = (threshold_c - corrected_mean_c) / max(effective_std, 0.01)
            cor_exceedance = float(1.0 - stats.norm.cdf(z))
        else:
            z = (threshold_c - corrected_mean_c) / max(effective_std, 0.01)
            cor_exceedance = float(stats.norm.cdf(z))

        # Build features and run model for both
        for exc_prob, result_list, label in [
            (unc_exceedance, uncorrected, "uncorrected"),
            (cor_exceedance, corrected, "corrected"),
        ]:
            features = {
                "ensemble_exceedance_prob": exc_prob,
                "ensemble_mean": ens_mean,
                "ensemble_spread": ens_std,
                "model_disagreement": 0.0,
                "forecast_revision_magnitude": 0.0,
                "forecast_revision_direction": 0.0,
                "forecast_horizon": 0.0,
                "climatological_anomaly": 0.0,
            }
            prediction = model.predict_proba(features)

            model_says_yes = prediction.probability > 0.5
            actual_yes = actual == "yes"

            result_list.append(ValidationResult(
                ticker=market.ticker,
                location=loc["name"],
                threshold_f=threshold_f,
                threshold_c=threshold_c,
                direction=direction,
                actual_result=actual,
                model_prob=prediction.probability,
                raw_prob=exc_prob,
                confidence=prediction.confidence,
                ensemble_spread=ens_std,
                ensemble_mean=ens_mean,
                n_members=len(values_at_date),
                forecast_horizon_h=0.0,
                market_last_price=float(market.last_price),
                correct=model_says_yes == actual_yes,
            ))

    await client.close()
    return uncorrected, corrected


# ---------------------------------------------------------------------------
# Forward Predict: make predictions for active markets using live forecasts
# ---------------------------------------------------------------------------

@dataclass
class ForwardPrediction:
    ticker: str
    location: str
    variable: str  # "high" or "low"
    threshold_f: float
    direction: float
    settle_date: str
    model_prob: float
    raw_exceedance: float
    bias_exceedance: float | None
    ensemble_mean_c: float
    ensemble_std_c: float
    n_members: int
    forecast_horizon_h: float
    market_yes_price: float
    market_no_price: float
    edge_vs_market: float  # model_prob - market_yes_price (positive = model says YES is cheap)
    confidence: float
    timestamp: str


async def predict_forward(
    config_base: str = "config/default.yaml",
    config_overlay: str | None = "config/stress-test.yaml",
) -> list[ForwardPrediction]:
    """Make predictions for active weather markets using live ensemble forecasts.

    Fetches real forward-looking ensemble data (not hindcast) and computes
    bias-corrected probabilities.  Results are saved to JSON for later
    verification against settlements.
    """
    import math
    from scipy import stats as sp_stats

    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )
    client = KalshiRestClient(config.exchange)
    model = WeatherEnsembleModel()

    predictions: list[ForwardPrediction] = []
    now = datetime.now(timezone.utc)

    # Fetch active weather markets
    series_tickers: list[str] = []
    for loc in LOCATIONS:
        code = loc.get("aliases", [loc["name"].replace(" ", "").lower()])[0].upper()
        for prefix in ["KXHIGH", "KXHIGHT", "KXLOW", "KXLOWT"]:
            series_tickers.append(prefix + code)

    active_markets: list[Market] = []
    seen = set()
    for series in series_tickers:
        try:
            batch = await client.get_all_markets(
                series_ticker=series, status="open", max_pages=3,
            )
        except Exception:
            continue
        for m in batch:
            if m.ticker not in seen:
                seen.add(m.ticker)
                active_markets.append(m)

    log.info("predict_forward.found_active", count=len(active_markets))

    # Fetch ensemble forecasts (REAL forward-looking, no past_days)
    ensemble_cache: dict[str, dict[str, list[float]]] = {}

    async def _fetch_ensemble(loc: dict, daily_var: str) -> dict[str, list[float]]:
        cache_key = "%s_%s" % (loc["name"], daily_var)
        if cache_key in ensemble_cache:
            return ensemble_cache[cache_key]
        # Don't specify model — let Open-Meteo use its default GFS ensemble
        # which gives better ensemble spread than gfs025 alone
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "daily": daily_var, "forecast_days": 7,
            "timezone": "America/New_York",
        }
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble", params=params,
            )
            resp.raise_for_status()
            payload = resp.json()
        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        member_cols = []
        for key in sorted(daily.keys()):
            if key.startswith(daily_var) and "member" in key:
                member_cols.append([float(v) if v is not None else float("nan") for v in daily[key]])
        result = {}
        for d_idx, d_str in enumerate(dates):
            vals = [mc[d_idx] for mc in member_cols if d_idx < len(mc) and mc[d_idx] == mc[d_idx]]
            if vals:
                result[d_str] = vals
        ensemble_cache[cache_key] = result
        return result

    for market in active_markets:
        loc = _match_location(market.ticker, market.title)
        if loc is None:
            continue

        threshold_f, direction = _parse_threshold_and_direction(market)
        if threshold_f is None or direction is None:
            continue

        threshold_c = _f_to_c(threshold_f)
        ticker_upper = market.ticker.upper()

        if "KXHIGH" in ticker_upper:
            daily_var = "temperature_2m_max"
            var_name = "high"
        elif "KXLOW" in ticker_upper:
            daily_var = "temperature_2m_min"
            var_name = "low"
        else:
            continue

        # Parse settle date
        ticker_date_match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker_upper)
        if not ticker_date_match:
            continue
        yr = int("20" + ticker_date_match.group(1))
        mon_str = ticker_date_match.group(2)
        day_val = int(ticker_date_match.group(3))
        months = {m: i for i, m in enumerate(calendar.month_abbr) if m}
        mon = months.get(mon_str.title(), 0)
        if mon == 0:
            continue
        settle_date = "%d-%02d-%02d" % (yr, mon, day_val)

        try:
            daily_data = await _fetch_ensemble(loc, daily_var)
        except Exception:
            continue

        values = daily_data.get(settle_date, [])
        if not values:
            continue

        ens_mean = sum(values) / len(values)
        ens_std = (sum((v - ens_mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5

        # Compute forecast horizon (hours from now to settle date noon)
        settle_dt = datetime(yr, mon, day_val, 12, 0, tzinfo=timezone.utc)
        horizon_h = max((settle_dt - now).total_seconds() / 3600.0, 0.0)

        # Raw exceedance (empirical)
        if direction > 0:
            raw_exc = sum(1 for v in values if v >= threshold_c) / len(values)
        else:
            raw_exc = sum(1 for v in values if v < threshold_c) / len(values)

        # Bias-corrected exceedance
        bias_exc = bias_corrected_exceedance(
            ensemble_mean_c=ens_mean,
            ensemble_std_c=ens_std,
            threshold_c=threshold_c,
            direction=direction,
            location=loc["name"],
            variable=var_name,
        )

        # Run through model
        features = {
            "station_bias_exceedance": bias_exc,
            "ensemble_exceedance_prob": raw_exc,
            "ensemble_mean": ens_mean,
            "ensemble_spread": ens_std,
            "model_disagreement": 0.0,
            "forecast_revision_magnitude": 0.0,
            "forecast_revision_direction": 0.0,
            "forecast_horizon": horizon_h,
            "climatological_anomaly": 0.0,
        }
        prediction = model.predict_proba(features)

        market_yes = float(market.last_price) if market.last_price else 0.5
        market_no = 1.0 - market_yes

        edge = prediction.probability - market_yes

        predictions.append(ForwardPrediction(
            ticker=market.ticker,
            location=loc["name"],
            variable=var_name,
            threshold_f=threshold_f,
            direction=direction,
            settle_date=settle_date,
            model_prob=prediction.probability,
            raw_exceedance=raw_exc,
            bias_exceedance=bias_exc,
            ensemble_mean_c=ens_mean,
            ensemble_std_c=ens_std,
            n_members=len(values),
            forecast_horizon_h=horizon_h,
            market_yes_price=market_yes,
            market_no_price=market_no,
            edge_vs_market=edge,
            confidence=prediction.confidence,
            timestamp=now.isoformat(),
        ))

    await client.close()

    # Sort by absolute edge
    predictions.sort(key=lambda p: abs(p.edge_vs_market), reverse=True)
    return predictions


def print_forward_predictions(predictions: list[ForwardPrediction]) -> None:
    """Print forward prediction report."""
    print("\n" + "=" * 100)
    print("FORWARD PREDICTIONS — %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("Model: WeatherEnsembleModel v5")
    print("=" * 100)

    if not predictions:
        print("  No active weather markets found.")
        return

    print("\n  %-42s %5s %6s %6s %6s %6s %6s %5s %6s" % (
        "Ticker", "Dir", "Thr°F", "Model", "Mkt", "Edge", "Bias", "Sprd", "Hrz",
    ))
    print("  " + "-" * 95)

    for p in predictions:
        dir_str = "above" if p.direction > 0 else "below"
        edge_str = "%+.1f%%" % (p.edge_vs_market * 100)
        print("  %-42s %5s %6.0f %5.1f%% %5.1f%% %6s %5.1f%% %5.2f %5.0fh" % (
            p.ticker,
            dir_str,
            p.threshold_f,
            p.model_prob * 100,
            p.market_yes_price * 100,
            edge_str,
            (p.bias_exceedance or 0) * 100,
            p.ensemble_std_c,
            p.forecast_horizon_h,
        ))

    # Summary stats
    print("\n  SUMMARY:")
    print("  Total markets: %d" % len(predictions))
    edges = [p.edge_vs_market for p in predictions]
    big_edges = [p for p in predictions if abs(p.edge_vs_market) > 0.10]
    print("  Markets with >10%% edge: %d" % len(big_edges))
    if big_edges:
        avg_edge = sum(abs(p.edge_vs_market) for p in big_edges) / len(big_edges)
        print("  Avg absolute edge (>10%%): %.1f%%" % (avg_edge * 100))

    # By location
    by_loc: dict[str, list[ForwardPrediction]] = defaultdict(list)
    for p in predictions:
        by_loc[p.location].append(p)
    print("\n  BY LOCATION:")
    for loc_name in sorted(by_loc.keys()):
        loc_preds = by_loc[loc_name]
        avg_spread = sum(p.ensemble_std_c for p in loc_preds) / len(loc_preds)
        avg_edge = sum(abs(p.edge_vs_market) for p in loc_preds) / len(loc_preds)
        print("    %-20s %3d markets  avg_spread=%.2f°C  avg_edge=%.1f%%" % (
            loc_name, len(loc_preds), avg_spread, avg_edge * 100,
        ))


def save_forward_predictions(predictions: list[ForwardPrediction]) -> None:
    """Save predictions to JSON for later verification."""
    out_dir = Path("data/forward_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out_path = out_dir / f"predictions_{today}.json"

    records = []
    for p in predictions:
        records.append({
            "ticker": p.ticker,
            "location": p.location,
            "variable": p.variable,
            "threshold_f": p.threshold_f,
            "direction": p.direction,
            "settle_date": p.settle_date,
            "model_prob": round(p.model_prob, 4),
            "raw_exceedance": round(p.raw_exceedance, 4),
            "bias_exceedance": round(p.bias_exceedance, 4) if p.bias_exceedance is not None else None,
            "ensemble_mean_c": round(p.ensemble_mean_c, 2),
            "ensemble_std_c": round(p.ensemble_std_c, 2),
            "n_members": p.n_members,
            "forecast_horizon_h": round(p.forecast_horizon_h, 1),
            "market_yes_price": round(p.market_yes_price, 3),
            "edge_vs_market": round(p.edge_vs_market, 4),
            "confidence": round(p.confidence, 3),
            "timestamp": p.timestamp,
        })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n  Predictions saved to {out_path}")
    print(f"  Verify later with: python scripts/weather_backtest.py verify --file {out_path}")


async def verify_predictions(file_path: str, config_base: str, config_overlay: str | None) -> None:
    """Verify saved predictions against actual settlements."""
    with open(file_path) as f:
        records = json.load(f)

    config = load_config(
        Path(config_base),
        Path(config_overlay) if config_overlay else None,
    )
    client = KalshiRestClient(config.exchange)

    correct = 0
    total = 0
    edge_correct = 0
    edge_total = 0
    results_by_bucket: dict[str, list[tuple[float, bool]]] = defaultdict(list)

    print("\n" + "=" * 100)
    print("VERIFICATION OF FORWARD PREDICTIONS")
    print("Prediction file: %s" % file_path)
    print("=" * 100)

    for rec in records:
        ticker = rec["ticker"]
        try:
            market = await client.get_market(ticker)
        except Exception:
            continue

        if market.result in (MarketResult.NOT_SETTLED, MarketResult.VOIDED):
            continue

        actual_yes = market.result == MarketResult.YES
        model_prob = rec["model_prob"]
        model_says_yes = model_prob > 0.5
        is_correct = model_says_yes == actual_yes

        total += 1
        if is_correct:
            correct += 1

        # Bucket
        if model_prob < 0.2:
            bucket = "0-20%"
        elif model_prob < 0.4:
            bucket = "20-40%"
        elif model_prob < 0.6:
            bucket = "40-60%"
        elif model_prob < 0.8:
            bucket = "60-80%"
        else:
            bucket = "80-100%"
        results_by_bucket[bucket].append((model_prob, actual_yes))

        # Edge analysis
        edge = rec.get("edge_vs_market", 0)
        if abs(edge) > 0.05:
            edge_total += 1
            # Model said YES was cheap (edge > 0) → should settle YES
            # Model said NO was cheap (edge < 0) → should settle NO
            edge_side_correct = (edge > 0 and actual_yes) or (edge < 0 and not actual_yes)
            if edge_side_correct:
                edge_correct += 1

        result_str = "✓" if is_correct else "✗"
        print("  %s %-42s model=%.3f market=%.3f actual=%-3s edge=%+.1f%%" % (
            result_str, ticker, model_prob, rec["market_yes_price"],
            "yes" if actual_yes else "no", edge * 100,
        ))

    await client.close()

    if total == 0:
        print("\n  No settled markets found yet. Try again later.")
        return

    print("\n  RESULTS:")
    print("  Settled: %d/%d" % (total, len(records)))
    print("  Accuracy: %d/%d (%.1f%%)" % (correct, total, 100 * correct / total))

    if edge_total > 0:
        print("  Edge calls correct: %d/%d (%.1f%%)" % (
            edge_correct, edge_total, 100 * edge_correct / edge_total,
        ))

    print("\n  CALIBRATION:")
    for bucket in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
        entries = results_by_bucket.get(bucket, [])
        if not entries:
            continue
        actual_rate = sum(1 for _, y in entries if y) / len(entries)
        avg_prob = sum(p for p, _ in entries) / len(entries)
        print("    %-10s %3d  actual=%.1f%%  model=%.1f%%" % (
            bucket, len(entries), actual_rate * 100, avg_prob * 100,
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Weather model validator & archiver")
    sub = parser.add_subparsers(dest="command", required=True)

    val_parser = sub.add_parser("validate", help="Validate model against settled markets")
    val_parser.add_argument("--days", type=int, default=3, help="Days of history to validate")
    val_parser.add_argument("--config", default="config/default.yaml")
    val_parser.add_argument("--overlay", default="config/stress-test.yaml")

    arc_parser = sub.add_parser("archive", help="Archive current ensemble data")
    arc_parser.add_argument("--config", default="config/default.yaml")
    arc_parser.add_argument("--overlay", default="config/stress-test.yaml")
    arc_parser.add_argument("--data-dir", default=None)

    bf_parser = sub.add_parser("backfill", help="Backfill max available ensemble history")
    bf_parser.add_argument("--past-days", type=int, default=5, help="Days to look back")
    bf_parser.add_argument("--data-dir", default=None)

    bias_parser = sub.add_parser("bias", help="Analyze ensemble vs NWS station bias")
    bias_parser.add_argument("--days", type=int, default=30, help="Days of history for bias analysis")

    corr_parser = sub.add_parser("validate-corrected", help="Validate with bias correction")
    corr_parser.add_argument("--days", type=int, default=3, help="Days of settled markets")
    corr_parser.add_argument("--bias-days", type=int, default=30, help="Days for bias calculation")
    corr_parser.add_argument("--config", default="config/default.yaml")
    corr_parser.add_argument("--overlay", default="config/stress-test.yaml")

    pred_parser = sub.add_parser("predict", help="Make forward predictions for active markets")
    pred_parser.add_argument("--config", default="config/default.yaml")
    pred_parser.add_argument("--overlay", default="config/stress-test.yaml")

    ver_parser = sub.add_parser("verify", help="Verify saved predictions against settlements")
    ver_parser.add_argument("--file", required=True, help="Path to prediction JSON file")
    ver_parser.add_argument("--config", default="config/default.yaml")
    ver_parser.add_argument("--overlay", default="config/stress-test.yaml")

    args = parser.parse_args()
    setup_logging("INFO")

    if args.command == "validate":
        results = await validate(
            days=args.days,
            config_base=args.config,
            config_overlay=args.overlay,
        )
        print_validation_report(results)
        save_validation_results(results)

    elif args.command == "archive":
        count = await archive_ensembles(
            config_base=args.config,
            config_overlay=args.overlay,
            data_dir=args.data_dir,
        )
        print(f"\nArchived {count} ensemble timesteps.")

    elif args.command == "backfill":
        count = await backfill_ensembles(
            past_days=args.past_days,
            data_dir=args.data_dir,
        )
        print(f"\nBackfilled {count} ensemble timesteps.")

    elif args.command == "bias":
        results = await analyze_bias(days=args.days)
        print_bias_report(results)

    elif args.command == "validate-corrected":
        uncorrected, corrected_results = await validate_corrected(
            days=args.days,
            config_base=args.config,
            config_overlay=args.overlay,
            bias_days=args.bias_days,
        )
        print("\n" + "=" * 80)
        print("UNCORRECTED MODEL")
        print("=" * 80)
        print_validation_report(uncorrected)

        print("\n" + "=" * 80)
        print("BIAS-CORRECTED MODEL")
        print("=" * 80)
        print_validation_report(corrected_results)

        # Side-by-side comparison
        if uncorrected and corrected_results:
            unc_acc = sum(1 for r in uncorrected if r.correct) / len(uncorrected)
            cor_acc = sum(1 for r in corrected_results if r.correct) / len(corrected_results)
            unc_edge_correct = 0
            cor_edge_correct = 0
            unc_edge_total = 0
            cor_edge_total = 0
            for r in uncorrected:
                if abs(r.model_prob - r.market_last_price) > 0.05 and 0 < r.market_last_price < 1:
                    unc_edge_total += 1
                    model_side = "yes" if r.model_prob > r.market_last_price else "no"
                    if model_side == r.actual_result:
                        unc_edge_correct += 1
            for r in corrected_results:
                if abs(r.model_prob - r.market_last_price) > 0.05 and 0 < r.market_last_price < 1:
                    cor_edge_total += 1
                    model_side = "yes" if r.model_prob > r.market_last_price else "no"
                    if model_side == r.actual_result:
                        cor_edge_correct += 1

            print("\n" + "=" * 80)
            print("COMPARISON")
            print("=" * 80)
            print("  Accuracy:     %.1f%% uncorrected → %.1f%% corrected" % (unc_acc * 100, cor_acc * 100))
            if unc_edge_total > 0 and cor_edge_total > 0:
                print("  Edge calls:   %d/%d (%.1f%%) → %d/%d (%.1f%%)" % (
                    unc_edge_correct, unc_edge_total, 100 * unc_edge_correct / unc_edge_total,
                    cor_edge_correct, cor_edge_total, 100 * cor_edge_correct / cor_edge_total,
                ))
            print()

    elif args.command == "predict":
        predictions = await predict_forward(
            config_base=args.config,
            config_overlay=args.overlay,
        )
        print_forward_predictions(predictions)
        save_forward_predictions(predictions)

    elif args.command == "verify":
        await verify_predictions(
            file_path=args.file,
            config_base=args.config,
            config_overlay=args.overlay,
        )


if __name__ == "__main__":
    asyncio.run(main())
