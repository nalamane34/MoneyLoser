"""ECMWF open-data ensemble forecast fetcher using the Open-Meteo API.

Open-Meteo exposes ECMWF IFS ensemble forecasts (51 members) via a
dedicated endpoint.  This module mirrors the
:class:`~moneygone.data.weather.noaa.NOAAEnsembleFetcher` interface
so both can be used interchangeably.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

import httpx
import structlog

from moneygone.data.weather.noaa import ForecastEnsemble

logger = structlog.get_logger(__name__)

# Open-Meteo ECMWF ensemble endpoint
_BASE_URL = "https://ensemble-api.open-meteo.com/v1/ecmwf"

# Variable mapping (ECMWF uses slightly different parameter names).
_VARIABLE_MAP: dict[str, str] = {
    "temperature_2m": "temperature_2m",
    "precipitation": "precipitation",
    "wind_speed_10m": "wind_speed_10m",
    "wind_gusts_10m": "wind_gusts_10m",
    "relative_humidity_2m": "relative_humidity_2m",
    "snowfall": "snowfall",
    "surface_pressure": "surface_pressure",
}


class ECMWFOpenDataFetcher:
    """Async fetcher for ECMWF IFS ensemble forecasts via Open-Meteo.

    Parameters
    ----------
    client:
        Optional pre-configured :class:`httpx.AsyncClient`.
    """

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

    async def fetch_ensemble(
        self,
        lat: float,
        lon: float,
        variable: str = "temperature_2m",
        *,
        location_name: str = "",
        forecast_days: int = 7,
    ) -> ForecastEnsemble:
        """Fetch ECMWF IFS ensemble forecast for a location and variable.

        Parameters
        ----------
        lat, lon:
            Geographic coordinates.
        variable:
            Canonical variable name (see ``_VARIABLE_MAP``).
        location_name:
            Optional label stored with the result.
        forecast_days:
            Number of forecast days (1-15).

        Returns
        -------
        ForecastEnsemble
        """
        om_var = _VARIABLE_MAP.get(variable, variable)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": om_var,
            "forecast_days": min(forecast_days, 15),
        }

        client = await self._get_client()
        logger.info(
            "ecmwf.fetch_ensemble",
            lat=lat,
            lon=lon,
            variable=variable,
            forecast_days=forecast_days,
        )

        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

        hourly = payload.get("hourly", {})
        time_strings: list[str] = hourly.get("time", [])

        valid_times = [
            datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            for t in time_strings
        ]

        # Collect ensemble member columns.
        member_columns: list[list[float]] = []
        for key in sorted(hourly.keys()):
            if key.startswith(om_var) and "member" in key:
                values = hourly[key]
                member_columns.append(
                    [float(v) if v is not None else 0.0 for v in values]
                )

        if not member_columns and om_var in hourly:
            member_columns = [
                [float(v) if v is not None else 0.0 for v in hourly[om_var]]
            ]

        n_times = len(valid_times)

        ensemble_mean: list[float] = []
        ensemble_std: list[float] = []
        for t_idx in range(n_times):
            values_at_t = [m[t_idx] for m in member_columns if t_idx < len(m)]
            if values_at_t:
                ensemble_mean.append(statistics.mean(values_at_t))
                ensemble_std.append(
                    statistics.stdev(values_at_t) if len(values_at_t) > 1 else 0.0
                )
            else:
                ensemble_mean.append(0.0)
                ensemble_std.append(0.0)

        init_time = valid_times[0] if valid_times else datetime.now(tz=timezone.utc)

        logger.info(
            "ecmwf.fetch_ensemble.done",
            members=len(member_columns),
            timesteps=n_times,
        )

        return ForecastEnsemble(
            location_name=location_name or f"{lat},{lon}",
            lat=lat,
            lon=lon,
            variable=variable,
            init_time=init_time,
            valid_times=valid_times,
            member_values=member_columns,
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
        )
