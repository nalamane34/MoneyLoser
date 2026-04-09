"""FRED economic data release feed for resolution sniping.

Monitors the Federal Reserve Economic Data (FRED) API for new releases of
key economic indicators.  When a new data point appears, it generates an
:class:`EconomicSignal` if the value crosses a contract-relevant threshold.

Common series:
    - ``CPIAUCSL`` -- Consumer Price Index (All Urban Consumers)
    - ``UNRATE``   -- Unemployment Rate
    - ``GDP``      -- Gross Domestic Product
    - ``FEDFUNDS`` -- Federal Funds Effective Rate
    - ``T10Y2Y``   -- 10-Year/2-Year Treasury Spread
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

import httpx
import structlog

from moneygone.utils.time import now_utc, parse_iso

logger = structlog.get_logger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FREDConfig:
    """Configuration for connecting to the FRED API."""

    api_key: str = ""  # Will read from FRED_API_KEY env var if empty
    series_ids: list[str] = field(default_factory=lambda: [
        "CPIAUCSL", "UNRATE", "GDP", "FEDFUNDS", "PAYEMS", "T10Y2Y",
    ])

    def resolve_api_key(self) -> str:
        """Return the API key, falling back to the FRED_API_KEY env var."""
        return self.api_key or os.environ.get("FRED_API_KEY", "")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EconomicRelease:
    """A single economic data release from FRED."""

    series_id: str
    title: str
    value: float
    previous_value: float | None
    release_date: datetime
    observation_date: str  # YYYY-MM-DD as returned by FRED


@dataclass(frozen=True, slots=True)
class EconomicSignal:
    """Signal that an economic indicator has crossed a threshold."""

    series_id: str
    title: str
    value: float
    threshold: float
    direction: str  # "above" or "below"
    exceeded: bool
    margin: float  # value - threshold (positive means exceeded for "above")
    previous_value: float | None
    confidence: float  # 0-1
    detected_at: datetime


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------


class EconomicReleaseFeed:
    """Async client for monitoring FRED economic data releases.

    Polls the FRED API for the latest observations on configured series
    and detects new releases by comparing against previously seen data.

    Parameters
    ----------
    config:
        FRED API configuration with key and series IDs.
    client:
        Optional ``httpx.AsyncClient``.
    request_timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        config: FREDConfig,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 15.0,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._timeout = request_timeout
        # Track last-seen observation date per series.
        self._last_seen: dict[str, str] = {}
        self._polling_tasks: list[asyncio.Task[None]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"User-Agent": "MoneyGone/1.0"},
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
    # Fetching
    # ------------------------------------------------------------------

    async def get_latest_release(self, series_id: str) -> EconomicRelease | None:
        """Fetch the most recent observation for a FRED series.

        Parameters
        ----------
        series_id:
            FRED series identifier (e.g. ``"CPIAUCSL"``).

        Returns
        -------
        EconomicRelease | None
            Latest release, or ``None`` on failure.
        """
        client = await self._get_client()

        # First, get series metadata for the title.
        title = await self._get_series_title(series_id, client)

        # Fetch the latest two observations (current + previous).
        params = {
            "series_id": series_id,
            "api_key": self._config.resolve_api_key(),
            "file_type": "json",
            "sort_order": "desc",
            "limit": 2,
        }

        try:
            resp = await client.get(
                f"{_FRED_BASE}/series/observations", params=params
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "fred.http_error",
                series=series_id,
                status=exc.response.status_code,
            )
            return None
        except httpx.TransportError as exc:
            logger.error(
                "fred.transport_error",
                series=series_id,
                error=str(exc),
            )
            return None

        observations = payload.get("observations", [])
        if not observations:
            logger.warning("fred.no_observations", series=series_id)
            return None

        latest = observations[0]
        value_str = latest.get("value", ".")
        if value_str == "." or not value_str:
            logger.warning("fred.missing_value", series=series_id)
            return None

        try:
            value = float(value_str)
        except ValueError:
            logger.warning(
                "fred.invalid_value", series=series_id, value=value_str
            )
            return None

        previous_value: float | None = None
        if len(observations) > 1:
            prev_str = observations[1].get("value", ".")
            if prev_str != "." and prev_str:
                try:
                    previous_value = float(prev_str)
                except ValueError:
                    pass

        obs_date = latest.get("date", "")
        release_date_str = latest.get("realtime_start", obs_date)

        try:
            release_date = parse_iso(release_date_str) if release_date_str else now_utc()
        except (ValueError, TypeError):
            release_date = now_utc()

        return EconomicRelease(
            series_id=series_id,
            title=title,
            value=value,
            previous_value=previous_value,
            release_date=release_date,
            observation_date=obs_date,
        )

    async def get_release_observations(
        self,
        release_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch observations for a FRED release with cursor-based pagination.

        Uses the ``/fred/release/observations`` endpoint which returns
        all series observations tied to a specific release.

        Parameters
        ----------
        release_id:
            FRED release identifier (e.g. ``"10"`` for CPI).
        limit:
            Max observations per page (FRED default is 1000).

        Returns
        -------
        list[dict]
            List of observation dicts with ``series_id``, ``date``,
            ``value``, etc.
        """
        client = await self._get_client()
        all_observations: list[dict[str, Any]] = []
        offset = 0

        while True:
            params = {
                "release_id": release_id,
                "api_key": self._config.resolve_api_key(),
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
                "offset": offset,
            }
            try:
                resp = await client.get(
                    f"{_FRED_BASE}/release/observations", params=params
                )
                resp.raise_for_status()
                payload = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "fred.release_obs_http_error",
                    release_id=release_id,
                    status=exc.response.status_code,
                )
                break
            except httpx.TransportError as exc:
                logger.error(
                    "fred.release_obs_transport_error",
                    release_id=release_id,
                    error=str(exc),
                )
                break

            observations = payload.get("observations", [])
            if not observations:
                break

            all_observations.extend(observations)

            # Check if there are more pages.
            total = payload.get("count", len(all_observations))
            offset += len(observations)
            if offset >= total:
                break

        return all_observations

    async def get_upcoming_release_dates(
        self,
        *,
        include_empty: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch upcoming FRED release dates.

        Uses the ``/fred/releases/dates`` endpoint to discover when
        economic data will be published -- critical for resolution
        sniping (e.g. knowing exactly when CPI drops).

        Parameters
        ----------
        include_empty:
            Include releases with no data yet (``True`` by default,
            which is what we want for upcoming dates).
        limit:
            Max number of release dates to return.

        Returns
        -------
        list[dict]
            Each dict has ``release_id``, ``release_name``, and
            ``date`` fields.
        """
        client = await self._get_client()
        params: dict[str, Any] = {
            "api_key": self._config.resolve_api_key(),
            "file_type": "json",
            "include_release_dates_with_no_data": "true" if include_empty else "false",
            "sort_order": "asc",
            "limit": limit,
        }
        try:
            resp = await client.get(
                f"{_FRED_BASE}/releases/dates", params=params
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "fred.release_dates_http_error",
                status=exc.response.status_code,
            )
            return []
        except httpx.TransportError as exc:
            logger.error(
                "fred.release_dates_transport_error",
                error=str(exc),
            )
            return []

        return payload.get("release_dates", [])

    # ------------------------------------------------------------------
    # Threshold checking
    # ------------------------------------------------------------------

    def check_threshold(
        self,
        release: EconomicRelease,
        threshold: float,
        direction: str = "above",
    ) -> EconomicSignal | None:
        """Check if an economic release value crosses a threshold.

        Parameters
        ----------
        release:
            The economic data release.
        threshold:
            Numeric threshold.
        direction:
            ``"above"`` or ``"below"``.

        Returns
        -------
        EconomicSignal | None
            Signal if threshold is crossed, else ``None``.
        """
        if direction == "above":
            exceeded = release.value >= threshold
            margin = release.value - threshold
        else:
            exceeded = release.value <= threshold
            margin = threshold - release.value

        if not exceeded:
            return None

        # Confidence: higher when margin is larger relative to threshold.
        if threshold != 0:
            margin_pct = abs(margin) / abs(threshold)
        else:
            margin_pct = abs(margin)
        confidence = min(1.0, 0.8 + 0.2 * min(margin_pct, 1.0))

        return EconomicSignal(
            series_id=release.series_id,
            title=release.title,
            value=release.value,
            threshold=threshold,
            direction=direction,
            exceeded=exceeded,
            margin=round(margin, 4),
            previous_value=release.previous_value,
            confidence=round(confidence, 4),
            detected_at=now_utc(),
        )

    # ------------------------------------------------------------------
    # Continuous monitoring
    # ------------------------------------------------------------------

    async def monitor_upcoming_releases(
        self,
        series_ids: list[str] | None = None,
        interval_seconds: float = 300.0,
        callback: Callable[[EconomicRelease, EconomicSignal | None], Coroutine[Any, Any, None]] | None = None,
    ) -> asyncio.Task[None]:
        """Watch for new economic data releases.

        Polls FRED for each series and detects when a new observation date
        appears.  Calls the callback with the new release.

        Parameters
        ----------
        series_ids:
            Series to monitor.  Defaults to the config's ``series_ids``.
        interval_seconds:
            Seconds between poll cycles (default 5 minutes; FRED data
            updates are infrequent).
        callback:
            Async callable ``(release, signal_or_none) -> None``.

        Returns
        -------
        asyncio.Task
        """
        ids = series_ids or self._config.series_ids
        task = asyncio.create_task(
            self._monitor_loop(ids, interval_seconds, callback),
            name="fred_monitor",
        )
        self._polling_tasks.append(task)
        return task

    async def _monitor_loop(
        self,
        series_ids: list[str],
        interval: float,
        callback: Callable[[EconomicRelease, EconomicSignal | None], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Internal monitoring loop."""
        logger.info(
            "fred.monitoring_started",
            series=series_ids,
            interval=interval,
        )

        while True:
            try:
                for series_id in series_ids:
                    release = await self.get_latest_release(series_id)
                    if release is None:
                        continue

                    prev_date = self._last_seen.get(series_id)
                    is_new = prev_date is None or release.observation_date != prev_date

                    if is_new:
                        self._last_seen[series_id] = release.observation_date
                        logger.info(
                            "fred.new_release",
                            series=series_id,
                            value=release.value,
                            date=release.observation_date,
                            previous_date=prev_date,
                        )

                        if callback:
                            # The callback can decide whether to check thresholds.
                            await callback(release, None)

            except asyncio.CancelledError:
                logger.info("fred.monitoring_stopped")
                raise
            except Exception:
                logger.exception("fred.monitor_error")

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_series_title(
        self, series_id: str, client: httpx.AsyncClient
    ) -> str:
        """Fetch the title for a FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self._config.resolve_api_key(),
            "file_type": "json",
        }
        try:
            resp = await client.get(f"{_FRED_BASE}/series", params=params)
            resp.raise_for_status()
            payload = resp.json()
            serieses = payload.get("seriess", [])
            if serieses:
                return serieses[0].get("title", series_id)
        except (httpx.HTTPStatusError, httpx.TransportError):
            pass
        return series_id
