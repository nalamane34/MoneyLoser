"""Backtesting framework: replay historical data through the live pipeline."""

from moneygone.backtest.data_loader import HistoricalDataLoader, HistoricalEvent
from moneygone.backtest.engine import BacktestEngine
from moneygone.backtest.guards import LeakageGuard, TimeFencedStore
from moneygone.backtest.results import BacktestResult
from moneygone.backtest.sim_exchange import SimulatedExchange, SimulatedPortfolio

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "HistoricalDataLoader",
    "HistoricalEvent",
    "LeakageGuard",
    "SimulatedExchange",
    "SimulatedPortfolio",
    "TimeFencedStore",
]
