"""NWS station observation fetcher via Iowa Environmental Mesonet (IEM).

IEM parses the NWS Daily Climatological Report (CLI) — the same product
Kalshi uses for settlement — into clean JSON.  No API key required.

Usage::

    fetcher = NWSObservationFetcher()
    obs = await fetcher.fetch_daily("KNYC", date(2026, 4, 9))
    print(obs.high_f, obs.low_f)  # 55, 38

    # Bulk: all stations for a single date
    all_obs = await fetcher.fetch_all_stations(date(2026, 4, 9))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# IEM CLI endpoints
_CLI_STATION_URL = "https://mesonet.agron.iastate.edu/json/cli.py"
_CLI_BULK_URL = "https://mesonet.agron.iastate.edu/geojson/cli.py"

# Station → WFO (Weather Forecast Office) mapping for CLI product
_STATION_WFO: dict[str, str] = {
    "KNYC": "OKX",
    "KMDW": "LOT",
    "KLAX": "LOX",
    "KMIA": "MFL",
    "KDFW": "FWD",
    "KDEN": "BOU",
    "KSEA": "SEW",
    "KATL": "FFC",
    "KHOU": "HGX",
    "KPHX": "PSR",
    "KMSP": "MPX",
    "KOKC": "OUN",
    "KMSY": "LIX",
    "KLAS": "VEF",
    "KDCA": "LWX",
    "KAUS": "EWX",
}


@dataclass(frozen=True, slots=True)
class DailyObservation:
    """One day of observed weather from an NWS CLI report."""

    station: str
    valid_date: date
    high_f: int | None  # Daily high temperature (°F)
    low_f: int | None   # Daily low temperature (°F)
    high_normal_f: int | None  # Normal high (°F)
    low_normal_f: int | None   # Normal low (°F)
    high_depart_f: int | None  # Departure from normal
    low_depart_f: int | None
    high_record_f: int | None  # Record high
    low_record_f: int | None   # Record low
    precip_inches: float | None
    snow_inches: float | None

    @property
    def high_c(self) -> float | None:
        """High temperature in Celsius."""
        return (self.high_f - 32) * 5 / 9 if self.high_f is not None else None

    @property
    def low_c(self) -> float | None:
        """Low temperature in Celsius."""
        return (self.low_f - 32) * 5 / 9 if self.low_f is not None else None


def _parse_int(val: Any) -> int | None:
    if val is None or val == "M" or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_float(val: Any) -> float | None:
    if val is None or val == "M" or val == "T" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class NWSObservationFetcher:
    """Async fetcher for NWS CLI daily observations via IEM."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_station_month(
        self, station: str, year: int, month: int
    ) -> list[DailyObservation]:
        """Fetch all CLI observations for a station in a given month."""
        client = await self._get_client()
        resp = await client.get(
            _CLI_STATION_URL,
            params={"station": station, "year": year, "month": month},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        observations = []
        for r in results:
            valid_str = r.get("valid", "")
            if not valid_str:
                continue
            try:
                valid_date = date.fromisoformat(valid_str)
            except ValueError:
                continue

            obs = DailyObservation(
                station=r.get("station", station),
                valid_date=valid_date,
                high_f=_parse_int(r.get("high")),
                low_f=_parse_int(r.get("low")),
                high_normal_f=_parse_int(r.get("high_normal")),
                low_normal_f=_parse_int(r.get("low_normal")),
                high_depart_f=_parse_int(r.get("high_depart")),
                low_depart_f=_parse_int(r.get("low_depart")),
                high_record_f=_parse_int(r.get("high_record")),
                low_record_f=_parse_int(r.get("low_record")),
                precip_inches=_parse_float(r.get("precip")),
                snow_inches=_parse_float(r.get("snow")),
            )
            observations.append(obs)

        logger.info(
            "nws.fetch_station_month",
            station=station,
            year=year,
            month=month,
            count=len(observations),
        )
        return observations

    async def fetch_daily(
        self, station: str, target_date: date
    ) -> DailyObservation | None:
        """Fetch a single day's CLI observation for a station."""
        observations = await self.fetch_station_month(
            station, target_date.year, target_date.month
        )
        for obs in observations:
            if obs.valid_date == target_date:
                return obs
        return None

    async def fetch_all_stations(
        self, target_date: date
    ) -> dict[str, DailyObservation]:
        """Fetch CLI observations for all stations on a single date.

        Uses the bulk GeoJSON endpoint for efficiency.
        """
        client = await self._get_client()
        dt_str = target_date.isoformat()
        resp = await client.get(_CLI_BULK_URL, params={"dt": dt_str})
        resp.raise_for_status()
        data = resp.json()

        result: dict[str, DailyObservation] = {}
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            station = props.get("station", "")
            if not station:
                continue

            obs = DailyObservation(
                station=station,
                valid_date=target_date,
                high_f=_parse_int(props.get("high")),
                low_f=_parse_int(props.get("low")),
                high_normal_f=_parse_int(props.get("high_normal")),
                low_normal_f=_parse_int(props.get("low_normal")),
                high_depart_f=_parse_int(props.get("high_depart")),
                low_depart_f=_parse_int(props.get("low_depart")),
                high_record_f=_parse_int(props.get("high_record")),
                low_record_f=_parse_int(props.get("low_record")),
                precip_inches=_parse_float(props.get("precip")),
                snow_inches=_parse_float(props.get("snow")),
            )
            result[station] = obs

        logger.info("nws.fetch_all_stations", date=dt_str, count=len(result))
        return result

    async def fetch_station_range(
        self, station: str, start: date, end: date
    ) -> list[DailyObservation]:
        """Fetch CLI observations for a date range (inclusive).

        Fetches month-by-month and filters to the requested range.
        """
        observations: list[DailyObservation] = []
        seen_dates: set[date] = set()

        current = start.replace(day=1)
        end_month = end.replace(day=1)

        while current <= end_month:
            month_obs = await self.fetch_station_month(
                station, current.year, current.month
            )
            for obs in month_obs:
                if start <= obs.valid_date <= end and obs.valid_date not in seen_dates:
                    seen_dates.add(obs.valid_date)
                    observations.append(obs)

            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

            # Be gentle with the API
            await asyncio.sleep(0.3)

        observations.sort(key=lambda x: x.valid_date)
        return observations
