"""Tests for the feature-facing DataStore.query interface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_datastore_query_supports_named_params(data_store) -> None:
    now = datetime.now(timezone.utc)
    data_store.insert_trades(
        [
            {
                "trade_id": "t-1",
                "ticker": "TEST",
                "count": 1,
                "yes_price": 0.55,
                "taker_side": "yes",
                "trade_time": now - timedelta(minutes=2),
            },
            {
                "trade_id": "t-2",
                "ticker": "OTHER",
                "count": 1,
                "yes_price": 0.45,
                "taker_side": "no",
                "trade_time": now - timedelta(minutes=1),
            },
        ]
    )

    rows = data_store.query(
        """
        SELECT trade_id
        FROM trades
        WHERE ticker = $ticker AND trade_time <= $as_of
        ORDER BY trade_time
        """,
        {"ticker": "TEST", "as_of": now},
    )

    assert rows == [("t-1",)]
