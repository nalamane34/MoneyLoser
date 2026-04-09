"""Real-time weather observation feeds for resolution sniping.

Combines two data sources for redundancy and speed:

1. **NOAA Observation API** -- official US weather observations from NWS
   stations.  Provides authoritative readings but can lag by 30-60 minutes.
2. **Open-Meteo Current Weather API** -- near-real-time interpolated
   observations with minimal latency.

The :class:`LiveWeatherFeed` monitors current conditions and fires
:class:`ThresholdSignal` events when a weather variable crosses a
contract-relevant threshold (e.g., temperature exceeds 90 F in Chicago).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Coroutine

import httpx
import structlog

from moneygone.utils.time import now_utc, parse_iso

logger = structlog.get_logger(__name__)

_NOAA_BASE = "https://api.weather.gov"
_OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeatherObservation:
    """A single weather observation from a station or API."""

    station_id: str
    temperature_f: float | None
    temperature_c: float | None
    humidity: float | None
    wind_speed: float | None  # mph
    precipitation: float | None  # mm in last hour
    observation_time: datetime


@dataclass(frozen=True, slots=True)
class ThresholdSignal:
    """Signal that a weather variable has crossed a contract threshold."""

    variable: str  # "temperature", "precipitation", "wind_speed"
    current_value: float
    threshold: float
    exceeded: bool  # True if current_value >= threshold (for "above") etc.
    margin: float  # current_value - threshold (positive = exceeded for "above")
    direction: str  # "above" or "below"
    station_id: str
    hours_remaining_in_day: float
    confidence: float  # 0-1
    detected_at: datetime


# ---------------------------------------------------------------------------
# Live Weather Feed
# ---------------------------------------------------------------------------


class LiveWeatherFeed:
    """Async client for real-time weather observations.

    Fetches from NOAA and Open-Meteo, detects threshold crossings,
    and supports continuous polling with callbacks.

    Parameters
    ----------
    client:
        Optional ``httpx.AsyncClient``.  Created internally if ``None``.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout
        self._polling_tasks: list[asyncio.Task[None]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "User-Agent": "MoneyGone/1.0 (weather-monitor)",
                    "Accept": "application/geo+json",
                },
            )
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Cancel polling tasks and close the HTTP client."""
        for task in self._polling_tasks:
            task.cancel()
        self._polling_tasks.clear()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # NOAA observations
    # ------------------------------------------------------------------

    async def get_current_observation(self, station_id: str) -> WeatherObservation | None:
        """Fetch the latest observation from a NOAA weather station.

        Parameters
        ----------
        station_id:
            NWS station identifier (e.g. ``"KORD"`` for Chicago O'Hare).

        Returns
        -------
        WeatherObservation | None
            The observation, or ``None`` on failure.
        """
        url = f"{_NOAA_BASE}/stations/{station_id}/observations/latest"
        client = await self._get_client()

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "noaa.http_error",
                station=station_id,
                status=exc.response.status_code,
            )
            return None
        except httpx.TransportError as exc:
            logger.error(
                "noaa.transport_error",
                station=station_id,
                error=str(exc),
            )
            return None

        return self._parse_noaa_observation(payload, station_id)

    # ------------------------------------------------------------------
    # Open-Meteo current temperature
    # ------------------------------------------------------------------

    async def get_current_temperature(self, lat: float, lon: float) -> float | None:
        """Fetch current temperature from Open-Meteo.

        Parameters
        ----------
        lat, lon:
            Geographic coordinates.

        Returns
        -------
        float | None
            Temperature in Celsius, or ``None`` on failure.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "temperature_unit": "celsius",
        }
        client = await self._get_client()

        try:
            resp = await client.get(_OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "open_meteo.http_error",
                lat=lat,
                lon=lon,
                status=exc.response.status_code,
            )
            return None
        except httpx.TransportError as exc:
            logger.error(
                "open_meteo.transport_error",
                lat=lat,
                lon=lon,
                error=str(exc),
            )
            return None

        current = payload.get("current", {})
        temp = current.get("temperature_2m")
        if temp is not None:
            return float(temp)
        return None

    async def get_current_observation_open_meteo(
        self,
        lat: float,
        lon: float,
        station_id: str = "",
    ) -> WeatherObservation | None:
        """Fetch a full observation from Open-Meteo.

        Parameters
        ----------
        lat, lon:
            Geographic coordinates.
        station_id:
            Label for the observation (defaults to ``"lat,lon"``).

        Returns
        -------
        WeatherObservation | None
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "temperature_unit": "celsius",
            "wind_speed_unit": "mph",
        }
        client = await self._get_client()

        try:
            resp = await client.get(_OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.error(
                "open_meteo.observation_error",
                lat=lat,
                lon=lon,
                error=str(exc),
            )
            return None

        current = payload.get("current", {})
        time_str = current.get("time", "")

        temp_c = current.get("temperature_2m")
        temp_f = (temp_c * 9 / 5 + 32) if temp_c is not None else None

        sid = station_id or f"{lat},{lon}"

        try:
            obs_time = parse_iso(time_str) if time_str else now_utc()
        except (ValueError, TypeError):
            obs_time = now_utc()

        return WeatherObservation(
            station_id=sid,
            temperature_f=round(temp_f, 1) if temp_f is not None else None,
            temperature_c=round(temp_c, 1) if temp_c is not None else None,
            humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            precipitation=current.get("precipitation"),
            observation_time=obs_time,
        )

    # ------------------------------------------------------------------
    # Threshold checking
    # ------------------------------------------------------------------

    def check_threshold(
        self,
        observation: WeatherObservation,
        threshold: float,
        variable: str = "temperature",
        direction: str = "above",
    ) -> ThresholdSignal | None:
        """Check if a weather variable has crossed a threshold.

        Parameters
        ----------
        observation:
            Current weather observation.
        threshold:
            The numeric threshold to check against.
        variable:
            Which variable to check: ``"temperature"`` (F), ``"humidity"``,
            ``"wind_speed"``, ``"precipitation"``.
        direction:
            ``"above"`` means signal fires when value >= threshold.
            ``"below"`` means signal fires when value <= threshold.

        Returns
        -------
        ThresholdSignal | None
            Signal if threshold is crossed, else ``None``.
        """
        current_value = self._get_variable(observation, variable)
        if current_value is None:
            return None

        if direction == "above":
            exceeded = current_value >= threshold
            margin = current_value - threshold
        else:
            exceeded = current_value <= threshold
            margin = threshold - current_value

        # Calculate hours remaining in the day (UTC).
        now = now_utc()
        end_of_day_hour = 24
        hours_remaining = max(0.0, end_of_day_hour - now.hour - now.minute / 60)

        # Confidence heuristic: higher when margin is larger and more time
        # has passed (less chance of reversal with less time remaining).
        if exceeded:
            margin_factor = min(1.0, abs(margin) / max(threshold * 0.1, 1.0))
            time_factor = max(0.5, 1.0 - hours_remaining / 24)
            confidence = 0.7 + 0.3 * margin_factor * time_factor
        else:
            confidence = 0.0

        if not exceeded:
            return None

        return ThresholdSignal(
            variable=variable,
            current_value=current_value,
            threshold=threshold,
            exceeded=exceeded,
            margin=round(margin, 2),
            direction=direction,
            station_id=observation.station_id,
            hours_remaining_in_day=round(hours_remaining, 2),
            confidence=round(min(confidence, 1.0), 4),
            detected_at=now_utc(),
        )

    # ------------------------------------------------------------------
    # Continuous polling
    # ------------------------------------------------------------------

    async def start_polling(
        self,
        stations: list[dict[str, Any]],
        interval_seconds: float = 60.0,
        callback: Callable[[list[WeatherObservation], list[ThresholdSignal]], Coroutine[Any, Any, None]] | None = None,
    ) -> asyncio.Task[None]:
        """Start continuous monitoring of weather stations.

        Parameters
        ----------
        stations:
            List of station configs.  Each dict must have at least:
            ``{"station_id": "KORD"}`` for NOAA, or
            ``{"lat": 41.97, "lon": -87.90, "station_id": "KORD"}``
            for Open-Meteo.  Optional keys: ``"thresholds"`` (list of
            ``{"variable": ..., "threshold": ..., "direction": ...}``).
        interval_seconds:
            Seconds between poll cycles.
        callback:
            Async callable ``(observations, signals) -> None``.

        Returns
        -------
        asyncio.Task
        """
        task = asyncio.create_task(
            self._poll_loop(stations, interval_seconds, callback),
            name="weather_poll",
        )
        self._polling_tasks.append(task)
        return task

    async def _poll_loop(
        self,
        stations: list[dict[str, Any]],
        interval: float,
        callback: Callable[[list[WeatherObservation], list[ThresholdSignal]], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Internal polling loop for weather stations."""
        logger.info(
            "weather.polling_started",
            stations=len(stations),
            interval=interval,
        )

        while True:
            try:
                observations: list[WeatherObservation] = []
                signals: list[ThresholdSignal] = []

                for station_cfg in stations:
                    obs = await self._fetch_station(station_cfg)
                    if obs is None:
                        continue
                    observations.append(obs)

                    # Check any configured thresholds.
                    for thresh_cfg in station_cfg.get("thresholds", []):
                        signal = self.check_threshold(
                            obs,
                            threshold=thresh_cfg["threshold"],
                            variable=thresh_cfg.get("variable", "temperature"),
                            direction=thresh_cfg.get("direction", "above"),
                        )
                        if signal is not None:
                            signals.append(signal)
                            logger.info(
                                "weather.threshold_crossed",
                                station=obs.station_id,
                                variable=signal.variable,
                                value=signal.current_value,
                                threshold=signal.threshold,
                                confidence=signal.confidence,
                            )

                if callback and (observations or signals):
                    await callback(observations, signals)

            except asyncio.CancelledError:
                logger.info("weather.polling_stopped")
                raise
            except Exception:
                logger.exception("weather.poll_error")

            await asyncio.sleep(interval)

    async def _fetch_station(self, cfg: dict[str, Any]) -> WeatherObservation | None:
        """Fetch observation for a single station config."""
        station_id = cfg.get("station_id", "")
        lat = cfg.get("lat")
        lon = cfg.get("lon")

        # Prefer Open-Meteo if coordinates available (lower latency).
        if lat is not None and lon is not None:
            return await self.get_current_observation_open_meteo(
                lat, lon, station_id=station_id
            )

        # Fall back to NOAA station observations.
        if station_id:
            return await self.get_current_observation(station_id)

        logger.warning("weather.no_source", cfg=cfg)
        return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_noaa_observation(
        payload: dict[str, Any],
        station_id: str,
    ) -> WeatherObservation | None:
        """Parse NOAA observation JSON (GeoJSON format)."""
        try:
            props = payload.get("properties", {})

            # Temperature: NOAA returns in Celsius with unitCode.
            temp_c_data = props.get("temperature", {})
            temp_c = temp_c_data.get("value")
            temp_f = (temp_c * 9 / 5 + 32) if temp_c is not None else None

            # Humidity (percentage).
            humidity_data = props.get("relativeHumidity", {})
            humidity = humidity_data.get("value")

            # Wind speed: NOAA returns in km/h (wmoUnit:km_h-1).
            wind_data = props.get("windSpeed", {})
            wind_kmh = wind_data.get("value")
            wind_mph = (wind_kmh * 0.621371) if wind_kmh is not None else None

            # Precipitation in last hour (mm).
            precip_data = props.get("precipitationLastHour", {})
            precip = precip_data.get("value")

            # Observation time.
            time_str = props.get("timestamp", "")
            try:
                obs_time = parse_iso(time_str) if time_str else now_utc()
            except (ValueError, TypeError):
                obs_time = now_utc()

            return WeatherObservation(
                station_id=station_id,
                temperature_f=round(temp_f, 1) if temp_f is not None else None,
                temperature_c=round(temp_c, 1) if temp_c is not None else None,
                humidity=round(humidity, 1) if humidity is not None else None,
                wind_speed=round(wind_mph, 1) if wind_mph is not None else None,
                precipitation=precip,
                observation_time=obs_time,
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "noaa.parse_error",
                station=station_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _get_variable(obs: WeatherObservation, variable: str) -> float | None:
        """Extract a named variable from an observation."""
        mapping: dict[str, float | None] = {
            "temperature": obs.temperature_f,
            "temperature_f": obs.temperature_f,
            "temperature_c": obs.temperature_c,
            "humidity": obs.humidity,
            "wind_speed": obs.wind_speed,
            "precipitation": obs.precipitation,
        }
        return mapping.get(variable)
