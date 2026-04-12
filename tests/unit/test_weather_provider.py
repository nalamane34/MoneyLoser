from __future__ import annotations

from moneygone.config import default_weather_locations
from moneygone.execution.category_providers import WeatherDataProvider


class _FakeFetcher:
    async def close(self) -> None:
        return None


def test_weather_location_aliases_cover_common_kalshi_city_codes() -> None:
    provider = WeatherDataProvider(
        noaa_fetcher=_FakeFetcher(),
        ecmwf_fetcher=_FakeFetcher(),
        locations=default_weather_locations(),
        nws_fetcher=_FakeFetcher(),
    )

    assert provider._match_location("KXHIGHTSFO-26APR12-T64 San Francisco high temp")["name"] == "San Francisco"
    assert provider._match_location("KXLOWTPHX-26APR12-T65 Phoenix low temp")["name"] == "Phoenix"
    assert provider._match_location("KXHIGHTDC-26APR12-T70 Washington DC high temp")["name"] == "Washington DC"
    assert provider._match_location("KXLOWTNOLA-26APR12-B57.5 New Orleans low temp")["name"] == "New Orleans"
