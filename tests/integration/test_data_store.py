"""Integration tests for DuckDB DataStore."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
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

    def test_get_market_state_rows_since_returns_ordered_rows(self, data_store: DataStore) -> None:
        t1 = datetime(2026, 4, 9, 10, 0)
        t2 = datetime(2026, 4, 9, 10, 5)
        data_store._conn.execute(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, snapshot_time,
                 result, category, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "STREAM-TICKER", "EVT", "Test", "open",
                0.40, 0.42, 0.41, 100, 20,
                datetime(2026, 6, 1), t1, None, "sports", t1,
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
                "STREAM-TICKER", "EVT", "Test", "open",
                0.45, 0.47, 0.46, 200, 30,
                datetime(2026, 6, 1), t2, None, "sports", t2,
            ],
        )

        rows = data_store.get_market_state_rows_since(t1, limit=10)

        assert [row["yes_bid"] for row in rows[-2:]] == [0.40, 0.45]


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

    def test_get_orderbook_rows_since_parses_json_levels(self, data_store: DataStore) -> None:
        t1 = datetime(2026, 4, 9, 14, 30)
        data_store.insert_orderbook_snapshots([
            {
                "ticker": "OB-STREAM",
                "yes_levels": [{"price": 0.61, "contracts": 12}],
                "no_levels": [{"price": 0.39, "contracts": 8}],
                "seq": 1001,
                "snapshot_time": t1,
            }
        ])

        rows = data_store.get_orderbook_rows_since(
            t1 - timedelta(seconds=1),
            limit=10,
        )

        assert rows[-1]["seq"] == 1001
        assert rows[-1]["yes_levels"][0]["contracts"] == 12

    def test_attached_market_data_tables_support_point_in_time_queries(
        self,
        data_store: DataStore,
        tmp_path,
    ) -> None:
        attached_path = tmp_path / "market_data.duckdb"
        attached_store = DataStore(attached_path)
        attached_store.initialize_schema()
        attached_store.insert_market_states([
            {
                "ticker": "ATTACHED-TICKER",
                "event_ticker": "EVT-ATTACHED",
                "title": "Attached market",
                "status": "open",
                "yes_bid": 0.42,
                "yes_ask": 0.45,
                "last_price": 0.44,
                "volume": 250,
                "open_interest": 75,
                "close_time": datetime(2026, 6, 1, 20, 0),
                "snapshot_time": datetime(2026, 4, 11, 3, 0),
                "result": None,
                "category": "sports",
            }
        ])
        attached_store.insert_orderbook_snapshots([
            {
                "ticker": "ATTACHED-TICKER",
                "yes_levels": [{"price": 0.42, "contracts": 50}],
                "no_levels": [{"price": 0.56, "contracts": 40}],
                "seq": 17,
                "snapshot_time": datetime(2026, 4, 11, 3, 0),
            }
        ])
        attached_store.close()

        data_store.attach_readonly("market_data", attached_path)

        market_row = data_store.get_market_state_at(
            "ATTACHED-TICKER",
            datetime(2026, 4, 11, 3, 5),
            table="market_data.market_states",
        )
        orderbook_row = data_store.get_orderbook_at(
            "ATTACHED-TICKER",
            datetime(2026, 4, 11, 3, 5),
            table="market_data.orderbook_snapshots",
        )

        assert market_row is not None
        assert market_row["yes_bid"] == pytest.approx(0.42)
        assert orderbook_row is not None
        assert orderbook_row["seq"] == 17
        assert orderbook_row["yes_levels"][0]["contracts"] == 50


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


class TestExecutionObservability:
    """Test extended execution persistence used by monitoring."""

    def test_insert_fills_persists_extended_monitoring_columns(
        self,
        data_store: DataStore,
    ) -> None:
        data_store.insert_fills([
            {
                "trade_id": "fill-1",
                "ticker": "KXTEST",
                "side": "yes",
                "action": "buy",
                "count": 3,
                "price": 0.71,
                "is_taker": True,
                "fee_paid": 0.02,
                "category": "sports",
                "model_name": "sharp",
                "predicted_prob": 0.73,
                "predicted_confidence": 0.81,
                "raw_edge": 0.04,
                "fee_adjusted_edge": 0.03,
                "strategy": "engine",
                "signal_source": "sportsbook",
                "expected_profit": 0.09,
                "fill_time": datetime(2026, 4, 11, 3, 10, tzinfo=timezone.utc),
            }
        ])

        rows = data_store.query(
            """
            SELECT fee_paid, category, model_name, predicted_prob,
                   predicted_confidence, raw_edge, fee_adjusted_edge,
                   strategy, signal_source, expected_profit
            FROM fills_log
            WHERE trade_id = 'fill-1'
            """
        )

        assert rows == [
            (0.02, "sports", "sharp", 0.73, 0.81, 0.04, 0.03, "engine", "sportsbook", 0.09)
        ]


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

    def test_load_parquet_handles_schema_drift_with_missing_columns(
        self,
        data_store: DataStore,
        tmp_path,
    ) -> None:
        """Parquet loads should tolerate missing newer columns like draw_price."""
        parquet_path = tmp_path / "sportsbook_lines.parquet"
        conn = duckdb.connect()
        conn.execute(
            """
            CREATE TABLE old_lines AS
            SELECT
                'evt-legacy'::VARCHAR AS event_id,
                'nba'::VARCHAR AS sport,
                'Chicago Bulls'::VARCHAR AS home_team,
                'Boston Celtics'::VARCHAR AS away_team,
                'pinnacle'::VARCHAR AS bookmaker,
                TIMESTAMP '2026-04-10 00:00:00' AS commence_time,
                1.62::DOUBLE AS home_price,
                2.45::DOUBLE AS away_price,
                -4.5::DOUBLE AS spread_home,
                229.5::DOUBLE AS total,
                TIMESTAMP '2026-04-09 09:00:00' AS captured_at,
                TIMESTAMP '2026-04-09 09:00:01' AS ingested_at
            """
        )
        conn.execute(f"COPY old_lines TO '{parquet_path}' (FORMAT PARQUET)")
        conn.close()

        loaded = data_store.load_parquet_into_table(
            "sportsbook_game_lines",
            parquet_path,
        )
        rows = data_store.query(
            "SELECT event_id, draw_price, home_price, away_price FROM sportsbook_game_lines"
        )

        assert loaded == 1
        assert rows == [("evt-legacy", None, 1.62, 2.45)]

    def test_load_parquet_preserves_existing_rows_on_failed_reload(
        self,
        data_store: DataStore,
        tmp_path,
    ) -> None:
        data_store.insert_sportsbook_game_lines([
            {
                "event_id": "evt-existing",
                "sport": "nba",
                "home_team": "Home",
                "away_team": "Away",
                "bookmaker": "pinnacle",
                "commence_time": datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc),
                "home_price": 1.8,
                "away_price": 2.1,
                "captured_at": datetime(2026, 4, 9, 9, 0, tzinfo=timezone.utc),
            }
        ])

        parquet_path = tmp_path / "sportsbook_lines_bad.parquet"
        conn = duckdb.connect()
        conn.execute(
            """
            CREATE TABLE broken_lines AS
            SELECT
                'evt-bad'::VARCHAR AS event_id,
                'nba'::VARCHAR AS sport,
                'Bad Home'::VARCHAR AS home_team,
                'Bad Away'::VARCHAR AS away_team,
                'pinnacle'::VARCHAR AS bookmaker,
                TIMESTAMP '2026-04-10 00:00:00' AS commence_time,
                'not-a-double'::VARCHAR AS home_price,
                2.45::DOUBLE AS away_price,
                NULL::DOUBLE AS draw_price,
                NULL::DOUBLE AS spread_home,
                NULL::DOUBLE AS total,
                TIMESTAMP '2026-04-09 09:00:00' AS captured_at
            """
        )
        conn.execute(f"COPY broken_lines TO '{parquet_path}' (FORMAT PARQUET)")
        conn.close()

        with pytest.raises(Exception):
            data_store.load_parquet_into_table("sportsbook_game_lines", parquet_path)

        rows = data_store.query(
            "SELECT event_id, home_price, away_price FROM sportsbook_game_lines ORDER BY event_id"
        )
        assert rows == [("evt-existing", 1.8, 2.1)]
