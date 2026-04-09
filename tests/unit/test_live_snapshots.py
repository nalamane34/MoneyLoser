"""Tests for store-backed live sports snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from moneygone.data.sports.live_snapshots import StoreBackedSportsSnapshotProvider
from moneygone.data.sports.power_ratings import TeamRating
from moneygone.data.sports.stats import TeamInjurySummary
from moneygone.exchange.types import Market, MarketResult, MarketStatus


class _FakePowerRatings:
    async def get_ratings(self, league: str) -> dict[str, TeamRating]:
        assert league == "nba"
        return {
            "Los Angeles Lakers": TeamRating(
                team_id="1",
                display_name="Los Angeles Lakers",
                abbreviation="LAL",
                win_pct=0.65,
                wins=53,
                losses=29,
                avg_pts_for=118.0,
                avg_pts_against=112.0,
                point_differential=6.0,
            ),
            "Boston Celtics": TeamRating(
                team_id="2",
                display_name="Boston Celtics",
                abbreviation="BOS",
                win_pct=0.70,
                wins=57,
                losses=25,
                avg_pts_for=120.0,
                avg_pts_against=111.0,
                point_differential=9.0,
            ),
            "Phoenix Suns": TeamRating(
                team_id="3",
                display_name="Phoenix Suns",
                abbreviation="PHX",
                win_pct=0.58,
                wins=48,
                losses=34,
                avg_pts_for=116.0,
                avg_pts_against=113.0,
                point_differential=3.0,
            ),
            "Brooklyn Nets": TeamRating(
                team_id="4",
                display_name="Brooklyn Nets",
                abbreviation="BKN",
                win_pct=0.44,
                wins=36,
                losses=46,
                avg_pts_for=111.0,
                avg_pts_against=114.0,
                point_differential=-3.0,
            ),
            "Milwaukee Bucks": TeamRating(
                team_id="5",
                display_name="Milwaukee Bucks",
                abbreviation="MIL",
                win_pct=0.63,
                wins=52,
                losses=30,
                avg_pts_for=119.0,
                avg_pts_against=113.0,
                point_differential=6.0,
            ),
            "Miami Heat": TeamRating(
                team_id="6",
                display_name="Miami Heat",
                abbreviation="MIA",
                win_pct=0.55,
                wins=45,
                losses=37,
                avg_pts_for=112.0,
                avg_pts_against=110.0,
                point_differential=2.0,
            ),
            "Toronto Raptors": TeamRating(
                team_id="7",
                display_name="Toronto Raptors",
                abbreviation="TOR",
                win_pct=0.46,
                wins=38,
                losses=44,
                avg_pts_for=110.0,
                avg_pts_against=112.0,
                point_differential=-2.0,
            ),
        }

    def lookup(
        self,
        team_name: str,
        ratings: dict[str, TeamRating],
    ) -> TeamRating | None:
        return ratings.get(team_name)

    async def close(self) -> None:
        return None


class _FakeStatsFeed:
    async def get_team_injury_summary(
        self,
        sport: str,
        league: str,
        team_id: str,
        *,
        key_minutes_threshold: float = 20.0,
    ) -> TeamInjurySummary:
        assert sport == "nba"
        assert league == "nba"
        if team_id == "1":
            return TeamInjurySummary(
                team_id="1",
                team_name="Los Angeles Lakers",
                key_injuries=1,
                injury_severity=0.20,
            )
        if team_id == "2":
            return TeamInjurySummary(
                team_id="2",
                team_name="Boston Celtics",
                key_injuries=0,
                injury_severity=0.05,
            )
        return TeamInjurySummary(
            team_id=team_id,
            team_name="Other Team",
            key_injuries=0,
            injury_severity=0.0,
        )

    async def close(self) -> None:
        return None


class _FakeRestClient:
    def __init__(self, event_titles: dict[str, str]) -> None:
        self._event_titles = event_titles
        self.calls: list[str] = []

    async def get_event(self, event_ticker: str, with_nested_markets: bool = False) -> dict[str, str]:
        assert with_nested_markets is False
        self.calls.append(event_ticker)
        return {"title": self._event_titles[event_ticker]}


@pytest.mark.asyncio
async def test_provider_builds_snapshot_from_stored_lines(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_sportsbook_game_lines(
        [
            {
                "event_id": "evt-1",
                "sport": "nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.80,
                "away_price": 2.10,
                "spread_home": -2.5,
                "total": 229.5,
                "captured_at": now - timedelta(hours=1),
            },
            {
                "event_id": "evt-1",
                "sport": "nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.95,
                "away_price": 1.95,
                "spread_home": -1.0,
                "total": 226.5,
                "captured_at": now - timedelta(hours=6),
            },
        ]
    )
    market = Market(
        ticker="KXNBAGAME-LALBOS-LAL",
        event_ticker="EVT-1",
        series_ticker="NBA",
        title="NBA: Los Angeles Lakers vs Boston Celtics - Los Angeles Lakers to win",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.57"),
        yes_ask=Decimal("0.59"),
        last_price=Decimal("0.58"),
        volume=1000,
        open_interest=250,
        close_time=now + timedelta(hours=2),
        result=MarketResult.NOT_SETTLED,
        category="sports",
    )
    provider = StoreBackedSportsSnapshotProvider(
        data_store,
        leagues=["nba"],
        stats_feed=_FakeStatsFeed(),
        power_ratings=_FakePowerRatings(),
    )

    matched = await provider.refresh([market])
    snapshot = await provider.get_snapshot(market)

    assert [m.ticker for m in matched] == [market.ticker]
    assert snapshot is not None
    assert snapshot["event_id"] == "evt-1"
    assert snapshot["is_home_team"] == 1
    assert snapshot["kalshi_implied_prob"] == pytest.approx(0.59)
    assert snapshot["pinnacle_moneyline_home"] == pytest.approx(1.80)
    assert snapshot["opening_moneyline_home"] == pytest.approx(1.95)
    assert snapshot["home_key_injuries"] == 1
    assert snapshot["home_team_rating"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_provider_orients_winner_markets_from_yes_sub_title_and_city_aliases(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_sportsbook_game_lines(
        [
            {
                "event_id": "evt-lal-phx",
                "sport": "nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Phoenix Suns",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.72,
                "away_price": 2.20,
                "captured_at": now - timedelta(minutes=30),
            }
        ]
    )
    market = Market(
        ticker="KXNBAGAME-26APR10PHXLAL-LAL",
        event_ticker="KXNBAGAME-26APR10PHXLAL",
        series_ticker="KXNBAGAME",
        title="Phoenix at Los Angeles L Winner?",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.56"),
        yes_ask=Decimal("0.58"),
        last_price=Decimal("0.57"),
        volume=500,
        open_interest=100,
        close_time=now + timedelta(hours=2),
        category="sports",
        yes_sub_title="Los Angeles L",
        no_sub_title="Phoenix",
    )
    provider = StoreBackedSportsSnapshotProvider(
        data_store,
        leagues=["nba"],
        stats_feed=_FakeStatsFeed(),
        power_ratings=_FakePowerRatings(),
    )

    matched = await provider.refresh([market])
    snapshot = await provider.get_snapshot(market)

    assert [m.ticker for m in matched] == [market.ticker]
    assert snapshot is not None
    assert snapshot["event_id"] == "evt-lal-phx"
    assert snapshot["is_home_team"] == 1


@pytest.mark.asyncio
async def test_provider_uses_event_title_fallback_for_ambiguous_market_titles(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_sportsbook_game_lines(
        [
            {
                "event_id": "evt-mia-tor",
                "sport": "nba",
                "home_team": "Toronto Raptors",
                "away_team": "Miami Heat",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.90,
                "away_price": 1.95,
                "captured_at": now - timedelta(minutes=20),
            },
            {
                "event_id": "evt-mia-bos",
                "sport": "nba",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=5),
                "home_price": 1.80,
                "away_price": 2.05,
                "captured_at": now - timedelta(minutes=20),
            },
        ]
    )
    rest_client = _FakeRestClient({"EVT-MIA-TOR": "Miami at Toronto Winner?"})
    market = Market(
        ticker="KXNBAGAME-26APR09MIATOR-MIA",
        event_ticker="EVT-MIA-TOR",
        series_ticker="KXNBAGAME",
        title="Will Miami win?",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.49"),
        yes_ask=Decimal("0.51"),
        last_price=Decimal("0.50"),
        volume=300,
        open_interest=75,
        close_time=now + timedelta(hours=2),
        category="sports",
    )
    provider = StoreBackedSportsSnapshotProvider(
        data_store,
        leagues=["nba"],
        rest_client=rest_client,
        stats_feed=_FakeStatsFeed(),
        power_ratings=_FakePowerRatings(),
    )

    matched = await provider.refresh([market])
    snapshot = await provider.get_snapshot(market)

    assert [m.ticker for m in matched] == [market.ticker]
    assert snapshot is not None
    assert snapshot["event_id"] == "evt-mia-tor"
    assert snapshot["is_home_team"] == 0
    assert rest_client.calls == ["EVT-MIA-TOR"]


@pytest.mark.asyncio
async def test_provider_ignores_non_h2h_nba_markets(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_sportsbook_game_lines(
        [
            {
                "event_id": "evt-mia-tor",
                "sport": "nba",
                "home_team": "Toronto Raptors",
                "away_team": "Miami Heat",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.90,
                "away_price": 1.95,
                "captured_at": now - timedelta(minutes=20),
            }
        ]
    )
    market = Market(
        ticker="KXNBA1HWINNER-26APR09MIATOR-MIA",
        event_ticker="KXNBA1HWINNER-26APR09MIATOR",
        series_ticker="KXNBA1HWINNER",
        title="Miami vs Toronto: First Half Winner?",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.49"),
        yes_ask=Decimal("0.51"),
        last_price=Decimal("0.50"),
        volume=300,
        open_interest=75,
        close_time=now + timedelta(hours=2),
        category="sports",
        yes_sub_title="Miami",
        no_sub_title="Toronto",
    )
    provider = StoreBackedSportsSnapshotProvider(
        data_store,
        leagues=["nba"],
        stats_feed=_FakeStatsFeed(),
        power_ratings=_FakePowerRatings(),
    )

    matched = await provider.refresh([market])
    snapshot = await provider.get_snapshot(market)

    assert matched == []
    assert snapshot is None


@pytest.mark.asyncio
async def test_provider_rejects_single_team_overlap_false_positive(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_sportsbook_game_lines(
        [
            {
                "event_id": "evt-chi-was",
                "sport": "nba",
                "home_team": "Washington Wizards",
                "away_team": "Chicago Bulls",
                "bookmaker": "pinnacle",
                "commence_time": now + timedelta(hours=2),
                "home_price": 1.90,
                "away_price": 1.95,
                "captured_at": now - timedelta(minutes=20),
            }
        ]
    )
    market = Market(
        ticker="KXNBAGAME-26APR10ORLCHI-CHI",
        event_ticker="KXNBAGAME-26APR10ORLCHI",
        series_ticker="KXNBAGAME",
        title="Orlando at Chicago Winner?",
        status=MarketStatus.OPEN,
        yes_bid=Decimal("0.49"),
        yes_ask=Decimal("0.51"),
        last_price=Decimal("0.50"),
        volume=300,
        open_interest=75,
        close_time=now + timedelta(hours=2),
        category="sports",
        yes_sub_title="Chicago",
        no_sub_title="Orlando",
    )
    provider = StoreBackedSportsSnapshotProvider(
        data_store,
        leagues=["nba"],
        stats_feed=_FakeStatsFeed(),
        power_ratings=_FakePowerRatings(),
    )

    matched = await provider.refresh([market])
    snapshot = await provider.get_snapshot(market)

    assert matched == []
    assert snapshot is None
