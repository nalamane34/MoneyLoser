"""DuckDB storage layer for the MoneyGone data pipeline.

Provides append-only inserts and point-in-time query methods so that any
historical state can be reconstructed for backtesting or auditing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import structlog

from moneygone.data.schemas import ALL_TABLES

logger = structlog.get_logger(__name__)
_NAMED_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _strip_tz(dt: datetime) -> datetime:
    """Convert a timezone-aware datetime to a naive UTC datetime.

    DuckDB ``TIMESTAMP`` columns are timezone-naive, so we must strip
    tzinfo before using datetimes in query parameters.  If the datetime
    is already naive it is returned unchanged.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _coerce_timestamp(value: datetime | str | None) -> datetime | str | None:
    """Normalize timestamps for DuckDB inserts while preserving strings."""
    if isinstance(value, datetime):
        return _strip_tz(value)
    return value


class DataStore:
    """Thin wrapper around a DuckDB database with typed helpers.

    Parameters
    ----------
    db_path:
        Filesystem path for the DuckDB database file.  Use ``":memory:"``
        for a purely in-memory store (useful for tests).
    read_only:
        If ``True``, open the database in read-only mode.  Multiple
        processes can read concurrently; writes will raise an error.
    """

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        self._db_path = str(db_path)
        self._read_only = read_only
        self._column_cache: dict[str, set[str]] = {}
        config: dict[str, Any] = {"threads": 1}
        if read_only:
            config["access_mode"] = "read_only"
        self._conn = duckdb.connect(self._db_path, config=config)
        logger.info("datastore.opened", db_path=self._db_path, read_only=read_only)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialize_schema(self, tables: list[str] | None = None) -> None:
        """Create tables if they do not already exist.

        Parameters
        ----------
        tables:
            Specific DDL statements to execute.  If ``None``, creates all
            tables (``ALL_TABLES``).
        """
        ddl_list = tables if tables is not None else ALL_TABLES
        for ddl in ddl_list:
            self._conn.execute(ddl)
        self._run_migrations()
        self._column_cache.clear()
        logger.info("datastore.schema_initialized", table_count=len(ddl_list))

    def _run_migrations(self) -> None:
        """Run lightweight schema migrations for columns added after initial deploy."""
        migrations = [
            ("sportsbook_game_lines", "draw_price", "DOUBLE"),
            ("sports_outcomes", "pinnacle_draw_price", "DOUBLE"),
            ("sports_outcomes", "consensus_draw_price", "DOUBLE"),
        ]
        for table, column, col_type in migrations:
            try:
                cols = [
                    row[0]
                    for row in self._conn.execute(
                        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
                    ).fetchall()
                ]
                if column not in cols and cols:  # table exists but column missing
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    logger.info("datastore.migration", table=table, column=column)
            except Exception:
                pass  # table doesn't exist yet, CREATE TABLE will handle it

    def attach_readonly(self, name: str, db_path: Path | str, retries: int = 5) -> None:
        """Attach another DuckDB file as a read-only schema.

        After attaching, tables can be queried as ``{name}.table_name``.
        Use :meth:`create_attached_views` to create local views that
        transparently redirect queries to attached tables.

        Retries on lock conflicts since the writer process only holds
        the lock briefly during flushes.
        """
        import time

        p = Path(db_path)
        if not p.exists():
            logger.warning("datastore.attach_skipped", name=name, reason="file_not_found", path=str(p))
            return

        for attempt in range(retries):
            try:
                self._conn.execute(f"ATTACH '{p}' AS {name} (READ_ONLY)")
                logger.info("datastore.attached", name=name, path=str(p))
                return
            except duckdb.IOException as exc:
                if "lock" in str(exc).lower() and attempt < retries - 1:
                    wait = 2.0 * (attempt + 1)
                    logger.warning(
                        "datastore.attach_lock_retry",
                        name=name,
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    def create_attached_views(self, attachments: dict[str, list[str]]) -> None:
        """Create local views pointing to tables in attached databases.

        Parameters
        ----------
        attachments:
            ``{schema_name: [table_name, ...]}`` mapping.  For each entry,
            creates ``CREATE OR REPLACE VIEW table_name AS SELECT * FROM
            schema_name.table_name``.  This lets existing query code work
            unchanged against attached databases.
        """
        for schema, tables in attachments.items():
            for table in tables:
                try:
                    self._conn.execute(
                        f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM {schema}.{table}"
                    )
                except Exception:
                    logger.debug("datastore.view_skipped", schema=schema, table=table, exc_info=True)

    def export_table_to_parquet(self, table: str, path: Path | str) -> None:
        """Export a table to a parquet file for cross-process sharing."""
        p = Path(path)
        tmp = p.with_suffix(".parquet.tmp")
        self._conn.execute(f"COPY {table} TO '{tmp}' (FORMAT PARQUET)")
        tmp.rename(p)  # atomic rename
        logger.debug("datastore.exported_parquet", table=table, path=str(p))

    def load_parquet_into_table(self, table: str, path: Path | str) -> int:
        """Replace a table's contents with data from a parquet file.

        Returns the number of rows loaded.
        """
        p = Path(path)
        if not p.exists():
            return 0
        self._conn.execute(f"DELETE FROM {table}")
        self._conn.execute(
            f"INSERT INTO {table} SELECT * FROM read_parquet('{p}')"
        )
        result = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        count = result[0] if result else 0
        logger.debug("datastore.loaded_parquet", table=table, path=str(p), rows=count)
        return count

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()
        logger.info("datastore.closed", db_path=self._db_path)

    def has_column(self, table: str, column: str) -> bool:
        """Return whether *table* currently contains *column*."""
        columns = self._column_cache.get(table)
        if columns is None:
            rows = self._conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            columns = {str(row[1]) for row in rows}
            self._column_cache[table] = columns
        return column in columns

    def query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a SQL query with ``$name`` parameters.

        This gives feature code a stable, minimal interface without exposing
        the underlying DuckDB connection directly.
        """
        bound_sql, bound_params = self._prepare_named_query(sql, params)
        return self._conn.execute(bound_sql, bound_params).fetchall()

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
                 last_price, volume, open_interest, close_time, snapshot_time,
                 result, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _coerce_timestamp(r["close_time"]),
                    _coerce_timestamp(
                        r.get("snapshot_time", datetime.now(timezone.utc))
                    ),
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

    def insert_sportsbook_game_lines(self, rows: list[dict[str, Any]]) -> None:
        """Append sportsbook moneyline snapshots for upcoming games."""
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO sportsbook_game_lines
                (event_id, sport, home_team, away_team, bookmaker,
                 commence_time, home_price, away_price, draw_price,
                 spread_home, total, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["event_id"],
                    str(r["sport"]).lower(),
                    r["home_team"],
                    r["away_team"],
                    str(r["bookmaker"]).lower(),
                    _coerce_timestamp(r.get("commence_time")),
                    r["home_price"],
                    r["away_price"],
                    r.get("draw_price"),
                    r.get("spread_home"),
                    r.get("total"),
                    _coerce_timestamp(r["captured_at"]),
                )
                for r in rows
            ],
        )
        logger.debug("datastore.inserted", table="sportsbook_game_lines", count=len(rows))

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
        time_expr = (
            "COALESCE(snapshot_time, ingested_at)"
            if self.has_column("market_states", "snapshot_time")
            else "ingested_at"
        )
        result = self._conn.execute(
            f"""
            SELECT *
            FROM market_states
            WHERE ticker = ? AND {time_expr} <= ?
            ORDER BY {time_expr} DESC, ingested_at DESC
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
            WHERE ticker = ? AND COALESCE(snapshot_time, ingested_at) <= ?
            ORDER BY COALESCE(snapshot_time, ingested_at) DESC, ingested_at DESC
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

    def get_opening_sportsbook_lines(
        self,
        *,
        bookmaker: str = "pinnacle",
        sport: str | None = None,
        event_ids: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return the earliest stored line for each event/bookmaker."""
        return self._get_ranked_sportsbook_lines(
            order_by="captured_at ASC, ingested_at ASC",
            bookmaker=bookmaker,
            sport=sport,
            event_ids=event_ids,
            as_of=as_of,
        )

    def get_latest_sportsbook_lines(
        self,
        *,
        bookmaker: str = "pinnacle",
        sport: str | None = None,
        event_ids: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return the latest stored line for each event/bookmaker."""
        return self._get_ranked_sportsbook_lines(
            order_by="captured_at DESC, ingested_at DESC",
            bookmaker=bookmaker,
            sport=sport,
            event_ids=event_ids,
            as_of=as_of,
        )

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

    def _get_ranked_sportsbook_lines(
        self,
        *,
        order_by: str,
        bookmaker: str = "pinnacle",
        sport: str | None = None,
        event_ids: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return one sportsbook line row per event according to *order_by*."""
        if event_ids == []:
            return {}

        clauses: list[str] = ["bookmaker = ?"]
        params: list[Any] = [bookmaker.lower()]

        if sport is not None:
            clauses.append("sport = ?")
            params.append(sport.lower())
        if as_of is not None:
            clauses.append("captured_at <= ?")
            params.append(_strip_tz(as_of))
        if event_ids:
            placeholders = ", ".join("?" for _ in event_ids)
            clauses.append(f"event_id IN ({placeholders})")
            params.extend(event_ids)

        where = f"WHERE {' AND '.join(clauses)}"
        results = self._conn.execute(
            f"""
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY event_id, bookmaker
                           ORDER BY {order_by}
                       ) AS rn
                FROM sportsbook_game_lines
                {where}
            )
            WHERE rn = 1
            """,
            params,
        ).fetchall()
        if not results:
            return {}

        columns = [desc[0] for desc in self._conn.description]
        rows = [dict(zip(columns, result)) for result in results]
        return {str(row["event_id"]): row for row in rows}

    @staticmethod
    def _prepare_named_query(
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Convert ``$name`` parameters to DuckDB positional binds."""
        bound_params: list[Any] = []
        params = params or {}

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in params:
                raise KeyError(f"Missing query parameter: {key}")
            value = params[key]
            if isinstance(value, datetime):
                value = _strip_tz(value)
            bound_params.append(value)
            return "?"

        return _NAMED_PARAM_RE.sub(_replace, sql), bound_params
