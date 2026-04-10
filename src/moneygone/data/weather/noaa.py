"""NOAA GEFS ensemble forecast fetcher using the Open-Meteo API.

Open-Meteo provides free access to NOAA GEFS ensemble data without
API keys.  This module fetches ensemble members for a given location
and weather variable, returning a structured :class:`ForecastEnsemble`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Open-Meteo GEFS ensemble endpoint
_BASE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Mapping from our canonical variable names to Open-Meteo parameter names.
_VARIABLE_MAP: dict[str, str] = {
    "temperature_2m": "temperature_2m",
    "precipitation": "precipitation",
    "wind_speed_10m": "wind_speed_10m",
    "wind_gusts_10m": "wind_gusts_10m",
    "relative_humidity_2m": "relative_humidity_2m",
    "snowfall": "snowfall",
    "surface_pressure": "surface_pressure",
}


@dataclass(frozen=True, slots=True)
class ForecastEnsemble:
    """Container for an ensemble forecast at a single location.

    Attributes
    ----------
    location_name:
        Human-readable location identifier (e.g. ``"Chicago"``).
    lat, lon:
        Coordinates of the forecast grid point.
    variable:
        Weather variable (e.g. ``"temperature_2m"``).
    init_time:
        Model initialization / reference time.
    valid_times:
        Ordered list of forecast valid times.
    member_values:
        ``member_values[i]`` is the list of values for ensemble member *i*,
        one per valid time.
    ensemble_mean:
        Mean across members at each valid time.
    ensemble_std:
        Standard deviation across members at each valid time.
    """

    location_name: str
    lat: float
    lon: float
    variable: str
    init_time: datetime
    valid_times: list[datetime] = field(default_factory=list)
    member_values: list[list[float]] = field(default_factory=list)
    ensemble_mean: list[float] = field(default_factory=list)
    ensemble_std: list[float] = field(default_factory=list)


class NOAAEnsembleFetcher:
    """Async fetcher for NOAA GEFS ensemble data via Open-Meteo.

    Parameters
    ----------
    client:
        Optional pre-configured :class:`httpx.AsyncClient`.  If ``None``
        a new client is created per call.
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
        """Fetch GEFS ensemble forecast for a location and variable.

        Parameters
        ----------
        lat, lon:
            Geographic coordinates.
        variable:
            Canonical variable name (see ``_VARIABLE_MAP``).
        location_name:
            Optional label stored with the result.
        forecast_days:
            Number of days to forecast (1-16).

        Returns
        -------
        ForecastEnsemble
        """
        om_var = _VARIABLE_MAP.get(variable, variable)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": om_var,
            "forecast_days": min(forecast_days, 16),
            "models": "gfs025",
        }

        client = await self._get_client()
        logger.info(
            "noaa.fetch_ensemble",
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

        # Parse valid times.
        valid_times = [
            datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            for t in time_strings
        ]

        # Collect ensemble member columns.  Open-Meteo returns columns
        # named like ``temperature_2m_member01``, ``..._member02``, etc.
        member_columns: list[list[float]] = []
        for key in sorted(hourly.keys()):
            if key.startswith(om_var) and "member" in key:
                values = hourly[key]
                member_columns.append(
                    [float(v) if v is not None else 0.0 for v in values]
                )

        # If no per-member columns, fall back to the mean column.
        # This is a DEGRADED mode — the ensemble endpoint should return
        # 31 GEFS members.  Log a warning so we notice immediately.
        if not member_columns and om_var in hourly:
            logger.warning(
                "noaa.no_ensemble_members",
                variable=om_var,
                available_keys=[k for k in hourly.keys() if k != "time"],
                msg="Falling back to single deterministic column — ensemble data missing",
            )
            member_columns = [
                [float(v) if v is not None else 0.0 for v in hourly[om_var]]
            ]

        n_times = len(valid_times)

        # Compute per-timestep statistics across members.
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

        # Use the first valid time as a proxy for init_time since
        # Open-Meteo does not expose the model initialization time directly.
        init_time = valid_times[0] if valid_times else datetime.now(tz=timezone.utc)

        logger.info(
            "noaa.fetch_ensemble.done",
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
