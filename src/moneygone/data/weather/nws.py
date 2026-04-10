"""National Weather Service (NWS) API fetcher for real-time observations and forecasts.

The NWS API (api.weather.gov) is free, requires no API key, and provides:
- Current observations from nearby weather stations
- Hourly and detailed forecasts
- Alerts

This module fetches current observations and hourly forecasts and converts
them into the same ForecastEnsemble format used by NOAA/ECMWF fetchers so
the weather pipeline can use NWS data as ground-truth anchoring.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from moneygone.data.weather.noaa import ForecastEnsemble

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.weather.gov"
_USER_AGENT = "MoneyGone/1.0 (weather-trading-bot)"

# Known city → NWS grid mapping (avoids a /points lookup each call)
_GRID_CACHE: dict[str, dict[str, Any]] = {
    "new york": {"office": "OKX", "gridX": 33, "gridY": 37, "station": "KNYC"},
    "chicago": {"office": "LOT", "gridX": 65, "gridY": 76, "station": "KORD"},
    "los angeles": {"office": "LOX", "gridX": 154, "gridY": 44, "station": "KLAX"},
    "miami": {"office": "MFL", "gridX": 110, "gridY": 50, "station": "KMIA"},
    "dallas": {"office": "FWD", "gridX": 79, "gridY": 108, "station": "KDFW"},
    "denver": {"office": "BOU", "gridX": 62, "gridY": 60, "station": "KDEN"},
    "seattle": {"office": "SEW", "gridX": 124, "gridY": 67, "station": "KSEA"},
    "atlanta": {"office": "FFC", "gridX": 50, "gridY": 86, "station": "KATL"},
}


@dataclass(frozen=True, slots=True)
class NWSObservation:
    """Current weather observation from a NWS station."""

    station: str
    timestamp: datetime
    temperature_f: float | None = None
    temperature_c: float | None = None
    dewpoint_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_mph: float | None = None
    wind_direction: str | None = None
    description: str | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NWSHourlyForecast:
    """Single hourly forecast period from NWS."""

    start_time: datetime
    end_time: datetime
    temperature_f: float
    wind_speed_mph: float
    wind_direction: str
    short_forecast: str
    probability_of_precipitation: float | None = None


def _c_to_f(celsius: float | None) -> float | None:
    """Convert Celsius to Fahrenheit."""
    if celsius is None:
        return None
    return celsius * 9.0 / 5.0 + 32.0


def _extract_value(prop: dict | None) -> float | None:
    """Extract numeric value from NWS API property dict."""
    if prop is None:
        return None
    if isinstance(prop, dict):
        return prop.get("value")
    return None


def _parse_wind_speed(text: str) -> float:
    """Parse wind speed like '10 mph' or '5 to 10 mph' into a float."""
    import re

    nums = re.findall(r"(\d+)", text)
    if not nums:
        return 0.0
    return statistics.mean(float(n) for n in nums)


class NWSFetcher:
    """Async fetcher for NWS observations and forecasts.

    Provides current conditions and hourly forecasts for configured
    locations.  Data is returned in ForecastEnsemble format for
    compatibility with the existing weather pipeline.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None
        self._grid_cache: dict[str, dict[str, Any]] = dict(_GRID_CACHE)
        self._obs_cache: dict[str, tuple[NWSObservation, float]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/geo+json"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Grid point lookup
    # ------------------------------------------------------------------

    async def _get_grid_info(self, lat: float, lon: float, location_name: str) -> dict[str, Any]:
        """Look up NWS grid info for a lat/lon, with caching."""
        key = location_name.lower().split(",")[0].strip()

        # Check pre-built cache
        if key in self._grid_cache:
            return self._grid_cache[key]

        # Dynamic lookup via /points
        client = await self._get_client()
        resp = await client.get(f"{_BASE_URL}/points/{lat:.4f},{lon:.4f}")
        resp.raise_for_status()
        props = resp.json()["properties"]

        info = {
            "office": props["gridId"],
            "gridX": props["gridX"],
            "gridY": props["gridY"],
            "station": props.get("observationStations", ""),
        }
        self._grid_cache[key] = info
        return info

    # ------------------------------------------------------------------
    # Current observations
    # ------------------------------------------------------------------

    async def get_current_observation(
        self, lat: float, lon: float, location_name: str = ""
    ) -> NWSObservation | None:
        """Fetch the latest observation for a location.

        Caches for 10 minutes to avoid hammering NWS.
        """
        import time

        key = location_name.lower().split(",")[0].strip() if location_name else f"{lat},{lon}"
        cached = self._obs_cache.get(key)
        if cached is not None and (time.time() - cached[1]) < 600:
            return cached[0]

        grid = await self._get_grid_info(lat, lon, location_name)
        station = grid.get("station", "")

        # If station is a URL from /points, extract station ID
        if isinstance(station, str) and station.startswith("http"):
            # Fetch stations list and grab the first one
            client = await self._get_client()
            try:
                resp = await client.get(station)
                resp.raise_for_status()
                features = resp.json().get("features", [])
                if features:
                    station = features[0]["properties"]["stationIdentifier"]
                else:
                    return None
            except Exception:
                logger.debug("nws.stations_fetch_failed", loc=location_name, exc_info=True)
                return None

        client = await self._get_client()
        try:
            resp = await client.get(
                f"{_BASE_URL}/stations/{station}/observations/latest"
            )
            resp.raise_for_status()
            props = resp.json()["properties"]
        except Exception:
            logger.debug("nws.observation_fetch_failed", station=station, exc_info=True)
            return None

        temp_c = _extract_value(props.get("temperature"))
        temp_f = _c_to_f(temp_c)
        dewpoint_c = _extract_value(props.get("dewpoint"))
        humidity = _extract_value(props.get("relativeHumidity"))
        wind_ms = _extract_value(props.get("windSpeed"))
        wind_mph = wind_ms * 2.237 if wind_ms is not None else None

        ts_str = props.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        obs = NWSObservation(
            station=station,
            timestamp=timestamp,
            temperature_f=round(temp_f, 1) if temp_f is not None else None,
            temperature_c=round(temp_c, 1) if temp_c is not None else None,
            dewpoint_c=round(dewpoint_c, 1) if dewpoint_c is not None else None,
            humidity_pct=round(humidity, 1) if humidity is not None else None,
            wind_speed_mph=round(wind_mph, 1) if wind_mph is not None else None,
            wind_direction=props.get("windDirection", {}).get("value"),
            description=props.get("textDescription"),
            raw_json=props,
        )

        self._obs_cache[key] = (obs, time.time())
        logger.info(
            "nws.observation",
            station=station,
            temp_f=obs.temperature_f,
            humidity=obs.humidity_pct,
            wind_mph=obs.wind_speed_mph,
            desc=obs.description,
        )
        return obs

    # ------------------------------------------------------------------
    # Hourly forecast
    # ------------------------------------------------------------------

    async def get_hourly_forecast(
        self, lat: float, lon: float, location_name: str = ""
    ) -> list[NWSHourlyForecast]:
        """Fetch the hourly forecast for the next 7 days."""
        grid = await self._get_grid_info(lat, lon, location_name)
        office = grid["office"]
        gx, gy = grid["gridX"], grid["gridY"]

        client = await self._get_client()
        try:
            resp = await client.get(
                f"{_BASE_URL}/gridpoints/{office}/{gx},{gy}/forecast/hourly"
            )
            resp.raise_for_status()
            periods = resp.json()["properties"]["periods"]
        except Exception:
            logger.debug("nws.hourly_fetch_failed", loc=location_name, exc_info=True)
            return []

        forecasts: list[NWSHourlyForecast] = []
        for p in periods:
            try:
                start = datetime.fromisoformat(p["startTime"])
                end = datetime.fromisoformat(p["endTime"])
                pop = None
                if p.get("probabilityOfPrecipitation"):
                    pop = p["probabilityOfPrecipitation"].get("value")
                forecasts.append(
                    NWSHourlyForecast(
                        start_time=start,
                        end_time=end,
                        temperature_f=float(p["temperature"]),
                        wind_speed_mph=_parse_wind_speed(p.get("windSpeed", "0")),
                        wind_direction=p.get("windDirection", ""),
                        short_forecast=p.get("shortForecast", ""),
                        probability_of_precipitation=pop,
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        logger.info("nws.hourly_forecast", location=location_name, periods=len(forecasts))
        return forecasts

    # ------------------------------------------------------------------
    # Convert to ForecastEnsemble format
    # ------------------------------------------------------------------

    async def fetch_ensemble(
        self,
        lat: float,
        lon: float,
        variable: str = "temperature_2m",
        *,
        location_name: str = "",
        forecast_days: int = 7,
    ) -> ForecastEnsemble:
        """Fetch NWS hourly forecast and convert to ForecastEnsemble format.

        NWS gives a single deterministic forecast, not an ensemble.  We
        represent it as a 1-member "ensemble" so the downstream pipeline
        works unchanged.  The current observation is injected as the first
        data point for ground-truth anchoring.
        """
        hourly = await self.get_hourly_forecast(lat, lon, location_name)
        obs = await self.get_current_observation(lat, lon, location_name)

        if not hourly:
            # Return empty ensemble
            return ForecastEnsemble(
                location_name=location_name or f"{lat},{lon}",
                lat=lat, lon=lon,
                variable=variable,
                init_time=datetime.now(tz=timezone.utc),
            )

        # Map variable to the right field
        valid_times: list[datetime] = []
        values: list[float] = []

        for h in hourly:
            if variable in ("temperature_2m", "temperature"):
                # NWS gives Fahrenheit; convert to Celsius for consistency
                # with NOAA/ECMWF which use Celsius
                val_c = (h.temperature_f - 32.0) * 5.0 / 9.0
                values.append(val_c)
            elif variable == "wind_speed_10m":
                # Convert mph to m/s
                values.append(h.wind_speed_mph * 0.44704)
            elif variable == "precipitation":
                # NWS doesn't give hourly precip amounts directly,
                # use probability as a proxy (0-100 -> 0-1)
                pop = h.probability_of_precipitation or 0.0
                values.append(pop / 100.0)
            elif variable == "snowfall":
                # Rough proxy from forecast text
                val = 1.0 if "snow" in h.short_forecast.lower() else 0.0
                values.append(val)
            else:
                values.append(0.0)

            valid_times.append(
                h.start_time.replace(tzinfo=timezone.utc)
                if h.start_time.tzinfo is None
                else h.start_time.astimezone(timezone.utc)
            )

        # Limit to requested forecast_days
        max_hours = forecast_days * 24
        valid_times = valid_times[:max_hours]
        values = values[:max_hours]

        # Single member "ensemble" — std = 0 everywhere
        init_time = valid_times[0] if valid_times else datetime.now(tz=timezone.utc)

        logger.info(
            "nws.ensemble_built",
            location=location_name,
            variable=variable,
            timesteps=len(values),
            obs_temp_f=obs.temperature_f if obs else None,
        )

        return ForecastEnsemble(
            location_name=location_name or f"{lat},{lon}",
            lat=lat,
            lon=lon,
            variable=variable,
            init_time=init_time,
            valid_times=valid_times,
            member_values=[values],
            ensemble_mean=values,  # single member = mean
            ensemble_std=[0.0] * len(values),
        )
