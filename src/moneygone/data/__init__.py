"""Data layer for the MoneyGone trading system.

Provides DuckDB-backed storage, market data recording, and data feeds for
weather, crypto, and general event sources.
"""

from moneygone.data.market_data import MarketDataRecorder
from moneygone.data.schemas import ALL_TABLES
from moneygone.data.store import DataStore

__all__ = [
    "ALL_TABLES",
    "DataStore",
    "MarketDataRecorder",
]
