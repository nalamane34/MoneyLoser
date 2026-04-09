"""DuckDB storage layer for the MoneyGone data pipeline.

Provides append-only inserts and point-in-time query methods so that any
historical state can be reconstructed for backtesting or auditing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import structlog

from moneygone.data.schemas import ALL_TABLES

logger = structlog.get_logger(__name__)


def _strip_tz(dt: datetime) -> datetime:
    """Convert a timezone-aware datetime to a naive UTC datetime.

    DuckDB ``TIMESTAMP`` columns are timezone-naive, so we must strip
    tzinfo before using datetimes in query parameters.  If the datetime
    is already naive it is returned unchanged.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class DataStore:
    """Thin wrapper around a DuckDB database with typed helpers.

    Parameters
    ----------
    db_path:
        Filesystem path for the DuckDB database file.  Use ``":memory:"``
        for a purely in-memory store (useful for tests).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._conn = duckdb.connect(self._db_path, config={"threads": 1})
        logger.info("datastore.opened", db_path=self._db_path)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create all tables if they do not already exist."""
        for ddl in ALL_TABLES:
            self._conn.execute(ddl)
        logger.info("datastore.schema_initialized", table_count=len(ALL_TABLES))

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()
        logger.info("datastore.closed", db_path=self._db_path)

    # ------------------------------------------------------------------
    # Batch insert helpers
    # ------------------------------------------------------------------

    def insert_market_states(self, rows: list[dict[str, Any]]) -> None:
        """Append market-state snapshots.

        Each *row* dict must contain keys matching the ``market_states``
        columns (excluding ``ingested_at`` which defaults to now).
        """
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, result, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["ticker"],
                    r["event_ticker"],
                    r["title"],
                    r["status"],
                    r.get("yes_bid"),
                    r.get("yes_ask"),
                    r.get("last_price"),
                    r.get("volume"),
                    r.get("open_interest"),
                    r["close_time"],
                    r.get("result"),
                    r.get("category"),
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="market_states", count=len(rows))

    def insert_orderbook_snapshots(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO orderbook_snapshots
                (ticker, yes_levels, no_levels, seq, snapshot_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    r["ticker"],
                    json.dumps(r["yes_levels"]),
                    json.dumps(r["no_levels"]),
                    r.get("seq"),
                    r["snapshot_time"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="orderbook_snapshots", count=len(rows))

    def insert_trades(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO trades
                (trade_id, ticker, count, yes_price, taker_side, trade_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["trade_id"],
                    r["ticker"],
                    r["count"],
                    r["yes_price"],
                    r["taker_side"],
                    r["trade_time"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="trades", count=len(rows))

    def insert_forecast_ensembles(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO forecast_ensembles
                (location_name, lat, lon, variable, init_time, valid_time,
                 member_values, ensemble_mean, ensemble_std)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["location_name"],
                    r["lat"],
                    r["lon"],
                    r["variable"],
                    r["init_time"],
                    r["valid_time"],
                    json.dumps(r["member_values"]),
                    r["ensemble_mean"],
                    r["ensemble_std"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="forecast_ensembles", count=len(rows))

    def insert_funding_rates(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO funding_rates
                (exchange, symbol, rate, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            [
                (r["exchange"], r["symbol"], r["rate"], r["timestamp"])
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="funding_rates", count=len(rows))

    def insert_open_interest(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO open_interest
                (exchange, symbol, value, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            [
                (r["exchange"], r["symbol"], r["value"], r["timestamp"])
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="open_interest", count=len(rows))

    def insert_features(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO features
                (ticker, observation_time, feature_name, feature_value)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    r["ticker"],
                    r["observation_time"],
                    r["feature_name"],
                    r["feature_value"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="features", count=len(rows))

    def insert_predictions(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO predictions
                (ticker, model_name, model_version, probability,
                 raw_probability, confidence, prediction_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["ticker"],
                    r["model_name"],
                    r["model_version"],
                    r["probability"],
                    r["raw_probability"],
                    r["confidence"],
                    r["prediction_time"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="predictions", count=len(rows))

    def insert_fills(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO fills_log
                (trade_id, ticker, side, action, count, price, is_taker, fill_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["trade_id"],
                    r["ticker"],
                    r["side"],
                    r["action"],
                    r["count"],
                    r["price"],
                    r["is_taker"],
                    r["fill_time"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="fills_log", count=len(rows))

    def insert_settlements(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO settlements_log
                (ticker, market_result, revenue, payout, settled_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    r["ticker"],
                    r["market_result"],
                    r["revenue"],
                    r["payout"],
                    r["settled_time"],
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="settlements_log", count=len(rows))

    # ------------------------------------------------------------------
    # Point-in-time query methods
    # ------------------------------------------------------------------

    def get_market_state_at(
        self, ticker: str, as_of: datetime
    ) -> dict[str, Any] | None:
        """Return the latest market-state row for *ticker* ingested before *as_of*."""
        result = self._conn.execute(
            """
            SELECT *
            FROM market_states
            WHERE ticker = ? AND ingested_at <= ?
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            [ticker, _strip_tz(as_of)],
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result))

    def get_orderbook_at(
        self, ticker: str, as_of: datetime
    ) -> dict[str, Any] | None:
        """Return the latest orderbook snapshot for *ticker* before *as_of*."""
        result = self._conn.execute(
            """
            SELECT *
            FROM orderbook_snapshots
            WHERE ticker = ? AND ingested_at <= ?
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            [ticker, _strip_tz(as_of)],
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        row = dict(zip(columns, result))
        # Deserialize JSON levels back to Python objects.
        row["yes_levels"] = json.loads(row["yes_levels"])
        row["no_levels"] = json.loads(row["no_levels"])
        return row

    def get_forecasts_at(
        self,
        location: str,
        variable: str,
        as_of: datetime,
    ) -> list[dict[str, Any]]:
        """Return the latest forecast-ensemble rows for *location* / *variable*.

        Returns all valid-time rows from the most recent ``init_time``
        ingested before *as_of*.
        """
        results = self._conn.execute(
            """
            WITH latest_init AS (
                SELECT MAX(init_time) AS init_time
                FROM forecast_ensembles
                WHERE location_name = ?
                  AND variable = ?
                  AND ingested_at <= ?
            )
            SELECT fe.*
            FROM forecast_ensembles fe
            JOIN latest_init li ON fe.init_time = li.init_time
            WHERE fe.location_name = ?
              AND fe.variable = ?
              AND fe.ingested_at <= ?
            ORDER BY fe.valid_time
            """,
            [location, variable, _strip_tz(as_of), location, variable, _strip_tz(as_of)],
        ).fetchall()
        if not results:
            return []
        columns = [desc[0] for desc in self._conn.description]
        rows = [dict(zip(columns, r)) for r in results]
        for row in rows:
            row["member_values"] = json.loads(row["member_values"])
        return rows

    def get_features_at(
        self, ticker: str, as_of: datetime
    ) -> dict[str, float]:
        """Return a ``{feature_name: feature_value}`` dict with the latest
        value for each feature ingested before *as_of*.
        """
        results = self._conn.execute(
            """
            SELECT feature_name, feature_value
            FROM (
                SELECT feature_name, feature_value,
                       ROW_NUMBER() OVER (
                           PARTITION BY feature_name
                           ORDER BY observation_time DESC
                       ) AS rn
                FROM features
                WHERE ticker = ? AND ingested_at <= ?
            )
            WHERE rn = 1
            """,
            [ticker, _strip_tz(as_of)],
        ).fetchall()
        return {name: value for name, value in results}

    def get_trades_between(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Return all trades for *ticker* with ``trade_time`` in [start, end]."""
        results = self._conn.execute(
            """
            SELECT *
            FROM trades
            WHERE ticker = ? AND trade_time >= ? AND trade_time <= ?
            ORDER BY trade_time
            """,
            [ticker, _strip_tz(start), _strip_tz(end)],
        ).fetchall()
        if not results:
            return []
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in results]

    def get_funding_rates_at(
        self, symbol: str, as_of: datetime
    ) -> dict[str, Any] | None:
        """Return the latest funding-rate row for *symbol* before *as_of*."""
        result = self._conn.execute(
            """
            SELECT *
            FROM funding_rates
            WHERE symbol = ? AND ingested_at <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            [symbol, _strip_tz(as_of)],
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result))

    def get_latest_prediction(
        self, ticker: str, model_name: str
    ) -> dict[str, Any] | None:
        """Return the most recent prediction for *ticker* from *model_name*."""
        result = self._conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE ticker = ? AND model_name = ?
            ORDER BY prediction_time DESC
            LIMIT 1
            """,
            [ticker, model_name],
        ).fetchone()
        if result is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result))

    def get_fills(
        self,
        ticker: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return fill records, optionally filtered by ticker and time range."""
        clauses: list[str] = []
        params: list[Any] = []
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if start is not None:
            clauses.append("fill_time >= ?")
            params.append(_strip_tz(start))
        if end is not None:
            clauses.append("fill_time <= ?")
            params.append(_strip_tz(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        results = self._conn.execute(
            f"SELECT * FROM fills_log {where} ORDER BY fill_time", params
        ).fetchall()
        if not results:
            return []
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in results]

    def get_settlements(
        self,
        ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return settlement records, optionally filtered by ticker."""
        if ticker is not None:
            results = self._conn.execute(
                "SELECT * FROM settlements_log WHERE ticker = ? ORDER BY settled_time",
                [ticker],
            ).fetchall()
        else:
            results = self._conn.execute(
                "SELECT * FROM settlements_log ORDER BY settled_time"
            ).fetchall()
        if not results:
            return []
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in results]
