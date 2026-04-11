from __future__ import annotations

from datetime import datetime

from moneygone.backtest.data_loader import EventType, HistoricalDataLoader
from moneygone.data.store import DataStore


def test_loader_uses_snapshot_time_for_market_states_and_respects_ticker_filter() -> None:
    store = DataStore(":memory:")
    store.initialize_schema()
    try:
        store.insert_market_states(
            [
                {
                    "ticker": "T1",
                    "event_ticker": "E1",
                    "title": "One",
                    "status": "open",
                    "yes_bid": 0.40,
                    "yes_ask": 0.60,
                    "last_price": 0.50,
                    "volume": 10,
                    "open_interest": 1,
                    "close_time": datetime(2026, 6, 1, 20, 0),
                    "snapshot_time": datetime(2026, 4, 9, 10, 0),
                    "result": None,
                    "category": "politics",
                },
                {
                    "ticker": "T2",
                    "event_ticker": "E2",
                    "title": "Two",
                    "status": "open",
                    "yes_bid": 0.45,
                    "yes_ask": 0.55,
                    "last_price": 0.50,
                    "volume": 10,
                    "open_interest": 1,
                    "close_time": datetime(2026, 6, 1, 20, 0),
                    "snapshot_time": datetime(2026, 4, 9, 10, 5),
                    "result": None,
                    "category": "politics",
                },
            ]
        )
        store.insert_orderbook_snapshots(
            [
                {
                    "ticker": "T1",
                    "yes_levels": [[0.40, 10]],
                    "no_levels": [[0.40, 10]],
                    "seq": 1,
                    "snapshot_time": datetime(2026, 4, 9, 10, 0),
                },
                {
                    "ticker": "T2",
                    "yes_levels": [[0.45, 10]],
                    "no_levels": [[0.35, 10]],
                    "seq": 2,
                    "snapshot_time": datetime(2026, 4, 9, 10, 5),
                },
            ]
        )

        loader = HistoricalDataLoader(store)
        events = loader.load(
            start_date=datetime(2026, 4, 9, 9, 0),
            end_date=datetime(2026, 4, 9, 11, 0),
            tickers=["T1"],
        )

        assert {event.ticker for event in events} == {"T1"}
        tick_events = [event for event in events if event.event_type is EventType.TICK]
        assert len(tick_events) == 1
        assert tick_events[0].timestamp == datetime(2026, 4, 9, 10, 0)
    finally:
        store.close()
