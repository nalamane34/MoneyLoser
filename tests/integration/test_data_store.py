"""Integration tests for DuckDB DataStore."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from moneygone.data.store import DataStore


pytestmark = pytest.mark.integration


class TestDataStoreSchema:
    """Test schema initialization."""

    def test_initialize_schema(self, data_store: DataStore) -> None:
        """All expected tables should exist after initialization."""
        tables = data_store._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {row[0] for row in tables}

        expected = {
            "market_states",
            "orderbook_snapshots",
            "trades",
            "forecast_ensembles",
            "funding_rates",
            "open_interest",
            "sportsbook_game_lines",
            "features",
            "predictions",
            "fills_log",
            "settlements_log",
        }
        assert expected.issubset(table_names)


class TestMarketStates:
    """Test market state insert and query."""

    def test_insert_and_query_market_state(self, data_store: DataStore) -> None:
        """Insert a market state row and retrieve it."""
        row = {
            "ticker": "TEST-TICKER",
            "event_ticker": "EVT-TEST",
            "title": "Test market",
            "status": "open",
            "yes_bid": 0.60,
            "yes_ask": 0.62,
            "last_price": 0.61,
            "volume": 5000,
            "open_interest": 2000,
            "close_time": datetime(2026, 6, 1, 20, 0),
            "snapshot_time": datetime(2026, 4, 9, 10, 0),
            "result": None,
            "category": "weather",
        }
        data_store.insert_market_states([row])

        result = data_store.get_market_state_at(
            "TEST-TICKER",
            datetime.now(),
        )

        assert result is not None
        assert result["ticker"] == "TEST-TICKER"
        assert result["yes_bid"] == pytest.approx(0.60)
        assert result["yes_ask"] == pytest.approx(0.62)
        assert result["snapshot_time"] == row["snapshot_time"]

    def test_point_in_time_query(self, data_store: DataStore) -> None:
        """Insert at t1 and t2, query at t1.5 should return t1 data."""
        t1 = datetime(2026, 4, 9, 10, 0)
        t2 = datetime(2026, 4, 9, 12, 0)
        t_mid = datetime(2026, 4, 9, 11, 0)

        # Insert with explicit ingested_at via raw SQL to control timestamps
        data_store._conn.execute(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, snapshot_time,
                 result, category, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "PIT-TICKER", "EVT", "Test", "open",
                0.50, 0.52, 0.51, 1000, 500,
                datetime(2026, 6, 1), t1, None, "weather", t1,
            ],
        )
        data_store._conn.execute(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, snapshot_time,
                 result, category, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "PIT-TICKER", "EVT", "Test", "open",
                0.60, 0.62, 0.61, 2000, 800,
                datetime(2026, 6, 1), t2, None, "weather", t2,
            ],
        )

        # Query at t_mid should get t1 data
        result = data_store.get_market_state_at("PIT-TICKER", t_mid)
        assert result is not None
        assert result["yes_bid"] == pytest.approx(0.50)

        # Query at t2 or later should get t2 data
        result_late = data_store.get_market_state_at("PIT-TICKER", t2)
        assert result_late is not None
        assert result_late["yes_bid"] == pytest.approx(0.60)


class TestOrderbooks:
    """Test orderbook insert and query."""

    def test_insert_and_query_orderbook(self, data_store: DataStore) -> None:
        """Insert an orderbook snapshot and retrieve it."""
        row = {
            "ticker": "OB-TICKER",
            "yes_levels": [{"price": 0.60, "contracts": 100}],
            "no_levels": [{"price": 0.40, "contracts": 80}],
            "seq": 999,
            "snapshot_time": datetime(2026, 4, 9, 14, 30),
        }
        data_store.insert_orderbook_snapshots([row])

        result = data_store.get_orderbook_at("OB-TICKER", datetime.now())

        assert result is not None
        assert result["ticker"] == "OB-TICKER"
        assert result["seq"] == 999
        assert isinstance(result["yes_levels"], list)
        assert result["yes_levels"][0]["price"] == pytest.approx(0.60)


class TestTrades:
    """Test trade insert and query."""

    def test_insert_and_query_trades(self, data_store: DataStore) -> None:
        """Insert trades and retrieve them within a time range."""
        t1 = datetime(2026, 4, 9, 10, 0)
        t2 = datetime(2026, 4, 9, 11, 0)
        t3 = datetime(2026, 4, 9, 12, 0)

        rows = [
            {
                "trade_id": "t1",
                "ticker": "TR-TICKER",
                "count": 10,
                "yes_price": 0.55,
                "taker_side": "yes",
                "trade_time": t1,
            },
            {
                "trade_id": "t2",
                "ticker": "TR-TICKER",
                "count": 20,
                "yes_price": 0.60,
                "taker_side": "no",
                "trade_time": t2,
            },
            {
                "trade_id": "t3",
                "ticker": "TR-TICKER",
                "count": 5,
                "yes_price": 0.58,
                "taker_side": "yes",
                "trade_time": t3,
            },
        ]
        data_store.insert_trades(rows)

        # Query only the first two trades
        results = data_store.get_trades_between("TR-TICKER", t1, t2)
        assert len(results) == 2
        assert results[0]["trade_id"] == "t1"
        assert results[1]["trade_id"] == "t2"


class TestSportsbookLines:
    """Test sportsbook line-history inserts and ranked lookups."""

    def test_insert_and_query_opening_and_latest_lines(self, data_store: DataStore) -> None:
        """Opening should come from the first snapshot, latest from the most recent."""
        t1 = datetime(2026, 4, 9, 9, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "event_id": "evt-1",
                "sport": "nba",
                "home_team": "Chicago Bulls",
                "away_team": "Washington Wizards",
                "bookmaker": "pinnacle",
                "commence_time": datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                "home_price": 1.62,
                "away_price": 2.45,
                "spread_home": -4.5,
                "total": 229.5,
                "captured_at": t1,
            },
            {
                "event_id": "evt-1",
                "sport": "nba",
                "home_team": "Chicago Bulls",
                "away_team": "Washington Wizards",
                "bookmaker": "pinnacle",
                "commence_time": datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                "home_price": 1.48,
                "away_price": 2.75,
                "spread_home": -6.0,
                "total": 231.0,
                "captured_at": t2,
            },
        ]
        data_store.insert_sportsbook_game_lines(rows)

        opening = data_store.get_opening_sportsbook_lines(
            bookmaker="pinnacle",
            sport="nba",
            event_ids=["evt-1"],
        )
        latest = data_store.get_latest_sportsbook_lines(
            bookmaker="pinnacle",
            sport="nba",
            event_ids=["evt-1"],
        )
        latest_as_of_t1 = data_store.get_latest_sportsbook_lines(
            bookmaker="pinnacle",
            sport="nba",
            event_ids=["evt-1"],
            as_of=t1,
        )

        assert opening["evt-1"]["home_price"] == pytest.approx(1.62)
        assert opening["evt-1"]["away_price"] == pytest.approx(2.45)
        assert latest["evt-1"]["home_price"] == pytest.approx(1.48)
        assert latest["evt-1"]["away_price"] == pytest.approx(2.75)
        assert latest_as_of_t1["evt-1"]["home_price"] == pytest.approx(1.62)
