"""Tests for game-winner sportsbook and injury enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from moneygone.data.sports.odds import GameOdds, OddsAPIFeed
from moneygone.data.sports.power_ratings import TeamRating
from moneygone.data.sports.stats import PlayerInfo, PlayerStatsFeed, TeamInjurySummary
from moneygone.data.store import DataStore
from moneygone.features.base import FeatureContext
from moneygone.features.game_winner_features import (
    PinnacleVsMarketEdge,
    PinnacleWinProbability,
    PowerRatingEdge,
    TeamInjuryImpact,
)


class TestPlayerStatsFeed:
    """Tests for roster injury severity helpers."""

    def test_parse_avg_minutes_from_site_web_gamelog(self) -> None:
        """The site.web gamelog summary should yield average minutes."""
        payload = {
            "names": ["minutes", "points"],
            "seasonTypes": [
                {
                    "summary": {
                        "stats": [
                            {
                                "type": "avg",
                                "stats": ["28.5", "16.2"],
                            }
                        ]
                    }
                }
            ],
        }

        result = PlayerStatsFeed._parse_avg_minutes_from_gamelog(payload)

        assert result == pytest.approx(28.5, abs=1e-10)

    @pytest.mark.asyncio
    async def test_team_injury_summary_weights_key_rotation_players(self) -> None:
        """High-minute outs should count as key injuries and drive severity."""
        feed = PlayerStatsFeed()

        async def fake_get_team_roster(
            sport: str,  # noqa: ARG001
            league: str,  # noqa: ARG001
            team_id: str,
        ) -> list[PlayerInfo]:
            return [
                PlayerInfo(
                    player_id="p1",
                    name="Starter Out",
                    team_id=team_id,
                    team_name="Chicago Bulls",
                    position="G",
                    jersey="1",
                    status="Out",
                ),
                PlayerInfo(
                    player_id="p2",
                    name="Bench Questionable",
                    team_id=team_id,
                    team_name="Chicago Bulls",
                    position="F",
                    jersey="8",
                    status="Questionable",
                ),
                PlayerInfo(
                    player_id="p3",
                    name="Healthy Rotation",
                    team_id=team_id,
                    team_name="Chicago Bulls",
                    position="C",
                    jersey="11",
                    status="Active",
                ),
            ]

        async def fake_get_player_avg_minutes(
            sport: str,  # noqa: ARG001
            league: str,  # noqa: ARG001
            player_id: str,
        ) -> float | None:
            return {
                "p1": 34.0,
                "p2": 12.0,
            }.get(player_id)

        feed.get_team_roster = fake_get_team_roster  # type: ignore[method-assign]
        feed.get_player_avg_minutes = fake_get_player_avg_minutes  # type: ignore[method-assign]

        summary = await feed.get_team_injury_summary("basketball", "nba", "4")

        assert summary.team_name == "Chicago Bulls"
        assert summary.key_injuries == 1
        assert 0.0 < summary.injury_severity <= 1.0
        assert len(summary.impacted_players) == 2


@dataclass
class _FakePowerRatings:
    ratings: dict[str, TeamRating]

    async def get_ratings(self, league: str) -> dict[str, TeamRating]:  # noqa: ARG002
        return self.ratings

    @staticmethod
    def lookup(team_name: str, ratings: dict[str, TeamRating]) -> TeamRating | None:
        return ratings.get(team_name)

    async def close(self) -> None:
        return None


@dataclass
class _FakeStatsFeed:
    summaries: dict[str, TeamInjurySummary]

    async def get_team_injury_summary(
        self,
        sport: str,  # noqa: ARG001
        league: str,  # noqa: ARG001
        team_id: str,
        *,
        key_minutes_threshold: float = 20.0,  # noqa: ARG002
    ) -> TeamInjurySummary:
        return self.summaries[team_id]

    async def close(self) -> None:
        return None


class _FakeOddsFeed(OddsAPIFeed):
    def __init__(self, games: list[GameOdds]) -> None:
        super().__init__(api_key="test-key")
        self._games = games

    async def get_upcoming_games(  # type: ignore[override]
        self,
        league: str,  # noqa: ARG002
        *,
        bookmakers: list[str] | None = None,  # noqa: ARG002
    ) -> list[GameOdds]:
        return self._games


class TestGameWinnerSnapshots:
    """Tests for enriched game-winner snapshot building."""

    @pytest.mark.asyncio
    async def test_snapshots_include_pinnacle_injuries_and_power_ratings(self) -> None:
        """The enriched snapshot builder should populate the new sharp-book fields."""
        games = [
            GameOdds(
                event_id="evt-1",
                home_team="Chicago Bulls",
                away_team="Washington Wizards",
                commence_time="2026-04-10T00:00:00Z",
                bookmakers=[
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Chicago Bulls", "price": 1.40},
                                    {"name": "Washington Wizards", "price": 3.00},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Chicago Bulls", "point": -5.5},
                                    {"name": "Washington Wizards", "point": 5.5},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 231.5},
                                    {"name": "Under", "point": 231.5},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Chicago Bulls", "price": 1.36},
                                    {"name": "Washington Wizards", "price": 3.15},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Chicago Bulls", "point": -6.0},
                                    {"name": "Washington Wizards", "point": 6.0},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 232.0},
                                    {"name": "Under", "point": 232.0},
                                ],
                            },
                        ],
                    },
                ],
            )
        ]
        odds_feed = _FakeOddsFeed(games)

        ratings = {
            "Chicago Bulls": TeamRating(
                team_id="4",
                display_name="Chicago Bulls",
                abbreviation="CHI",
                win_pct=0.64,
                wins=52,
                losses=29,
                avg_pts_for=117.0,
                avg_pts_against=111.0,
                point_differential=6.0,
            ),
            "Washington Wizards": TeamRating(
                team_id="27",
                display_name="Washington Wizards",
                abbreviation="WSH",
                win_pct=0.28,
                wins=23,
                losses=58,
                avg_pts_for=108.0,
                avg_pts_against=118.0,
                point_differential=-10.0,
            ),
        }
        power_feed = _FakePowerRatings(ratings=ratings)
        stats_feed = _FakeStatsFeed(
            summaries={
                "4": TeamInjurySummary(
                    team_id="4",
                    team_name="Chicago Bulls",
                    key_injuries=1,
                    injury_severity=0.32,
                ),
                "27": TeamInjurySummary(
                    team_id="27",
                    team_name="Washington Wizards",
                    key_injuries=2,
                    injury_severity=0.58,
                ),
            }
        )

        snapshots = await odds_feed.get_game_winner_snapshots(
            "nba",
            stats_feed=stats_feed,
            power_ratings=power_feed,
        )

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot["pinnacle_moneyline_home"] == pytest.approx(1.36, abs=1e-10)
        assert snapshot["pinnacle_moneyline_away"] == pytest.approx(3.15, abs=1e-10)
        assert snapshot["pinnacle_home_win_prob"] is not None
        assert snapshot["sportsbook_home_win_prob"] is not None
        assert snapshot["home_key_injuries"] == 1
        assert snapshot["away_key_injuries"] == 2
        assert snapshot["home_injury_severity"] == pytest.approx(0.32, abs=1e-10)
        assert snapshot["away_team_rating"] == pytest.approx(-10.0, abs=1e-10)

        context = FeatureContext(
            ticker="TEST",
            observation_time=datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc),
            sports_snapshot={
                **snapshot,
                "is_home_team": 1,
                "kalshi_implied_prob": 0.62,
            },
        )

        pinnacle_prob = PinnacleWinProbability().compute(context)
        assert pinnacle_prob == pytest.approx(snapshot["pinnacle_home_win_prob"], abs=1e-10)
        assert PinnacleVsMarketEdge().compute(context) == pytest.approx(
            snapshot["pinnacle_home_win_prob"] - 0.62,
            abs=1e-10,
        )
        assert PowerRatingEdge().compute(context) == pytest.approx(16.0, abs=1e-10)
        assert TeamInjuryImpact().compute(context) == pytest.approx(0.26, abs=1e-10)

    @pytest.mark.asyncio
    async def test_snapshots_load_opening_lines_from_store(
        self,
        data_store: DataStore,
    ) -> None:
        """Stored Pinnacle history should backfill the opening line fields."""
        games = [
            GameOdds(
                event_id="evt-2",
                home_team="Chicago Bulls",
                away_team="Washington Wizards",
                commence_time="2026-04-10T00:00:00Z",
                bookmakers=[
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Chicago Bulls", "price": 1.44},
                                    {"name": "Washington Wizards", "price": 2.90},
                                ],
                            }
                        ],
                    }
                ],
            )
        ]
        data_store.insert_sportsbook_game_lines(
            [
                {
                    "event_id": "evt-2",
                    "sport": "nba",
                    "home_team": "Chicago Bulls",
                    "away_team": "Washington Wizards",
                    "bookmaker": "pinnacle",
                    "commence_time": datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                    "home_price": 1.60,
                    "away_price": 2.50,
                    "captured_at": datetime(2026, 4, 9, 8, 0, tzinfo=timezone.utc),
                },
                {
                    "event_id": "evt-2",
                    "sport": "nba",
                    "home_team": "Chicago Bulls",
                    "away_team": "Washington Wizards",
                    "bookmaker": "pinnacle",
                    "commence_time": datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                    "home_price": 1.50,
                    "away_price": 2.70,
                    "captured_at": datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc),
                },
            ]
        )

        snapshots = await _FakeOddsFeed(games).get_game_winner_snapshots(
            "nba",
            store=data_store,
            stats_feed=_FakeStatsFeed(summaries={}),
            power_ratings=_FakePowerRatings(ratings={}),
        )

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot["opening_moneyline_home"] == pytest.approx(1.60, abs=1e-10)
        assert snapshot["opening_moneyline_away"] == pytest.approx(2.50, abs=1e-10)
        assert snapshot["current_moneyline_home"] == pytest.approx(1.44, abs=1e-10)
        assert snapshot["current_moneyline_away"] == pytest.approx(2.90, abs=1e-10)
        assert snapshot["movement_line_source"] == "pinnacle"
