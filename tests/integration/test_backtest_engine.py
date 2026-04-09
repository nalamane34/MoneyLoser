"""Integration tests for backtest engine guards and results."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from moneygone.backtest.guards import LeakageGuard, LookaheadError, TimeFencedStore
from moneygone.backtest.results import (
    BacktestResult,
    compute_max_drawdown,
    compute_sharpe,
    compute_win_rate,
)
from moneygone.data.store import DataStore
from moneygone.exchange.types import OrderbookLevel, OrderbookSnapshot
from moneygone.features.base import FeatureContext


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# LeakageGuard tests
# ---------------------------------------------------------------------------


class TestLeakageGuard:
    """Test that the leakage guard blocks future data access."""

    def test_leakage_guard_blocks_future_data(self) -> None:
        """Trading after settlement time should raise LookaheadError."""
        settlement_time = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
        guard = LeakageGuard(settlement_times={"SETTLED-TICKER": settlement_time})

        # Trading before settlement is fine
        guard.validate_no_label_access(
            "SETTLED-TICKER",
            datetime(2026, 4, 10, 11, 0, tzinfo=timezone.utc),
        )

        # Trading at or after settlement should raise
        with pytest.raises(LookaheadError):
            guard.validate_no_label_access(
                "SETTLED-TICKER",
                datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
            )

        with pytest.raises(LookaheadError):
            guard.validate_no_label_access(
                "SETTLED-TICKER",
                datetime(2026, 4, 10, 13, 0, tzinfo=timezone.utc),
            )

    def test_leakage_guard_validates_feature_context(self) -> None:
        """An orderbook timestamp in the future should raise LookaheadError."""
        guard = LeakageGuard(settlement_times={})
        obs_time = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)
        future_time = datetime(2026, 4, 9, 15, 0, tzinfo=timezone.utc)

        # Future orderbook timestamp
        ob = OrderbookSnapshot(
            ticker="TEST",
            yes_bids=(),
            no_bids=(),
            seq=1,
            timestamp=future_time,
        )
        ctx = FeatureContext(
            ticker="TEST",
            observation_time=obs_time,
            orderbook=ob,
        )

        with pytest.raises(LookaheadError):
            guard.validate_feature_context(ctx)


# ---------------------------------------------------------------------------
# TimeFencedStore tests
# ---------------------------------------------------------------------------


class TestTimeFencedStore:
    """Test time fencing on the DataStore wrapper."""

    def test_time_fenced_store(self, data_store: DataStore) -> None:
        """TimeFencedStore should clamp queries to the as_of ceiling."""
        t1 = datetime(2026, 4, 9, 10, 0)
        t2 = datetime(2026, 4, 9, 12, 0)
        fence_time = datetime(2026, 4, 9, 11, 0)

        # Insert two rows at different ingested_at times
        data_store._conn.execute(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["FENCE-TICKER", "EVT", "Test", "open", 0.50, 0.52, 0.51,
             1000, 500, datetime(2026, 6, 1), t1],
        )
        data_store._conn.execute(
            """
            INSERT INTO market_states
                (ticker, event_ticker, title, status, yes_bid, yes_ask,
                 last_price, volume, open_interest, close_time, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["FENCE-TICKER", "EVT", "Test", "open", 0.70, 0.72, 0.71,
             2000, 800, datetime(2026, 6, 1), t2],
        )

        fenced = TimeFencedStore(data_store, as_of=fence_time)

        # Even requesting data far in the future, it should be clamped
        result = fenced.get_market_state_at(
            "FENCE-TICKER",
            datetime(2026, 12, 31),
        )

        assert result is not None
        # Should only see data from t1, not t2
        assert result["yes_bid"] == pytest.approx(0.50)

    def test_time_fence_cannot_go_backward(self, data_store: DataStore) -> None:
        """TimeFencedStore.advance_time should reject backward movement."""
        t1 = datetime(2026, 4, 9, 12, 0)
        t_earlier = datetime(2026, 4, 9, 10, 0)

        fenced = TimeFencedStore(data_store, as_of=t1)

        with pytest.raises(ValueError):
            fenced.advance_time(t_earlier)


# ---------------------------------------------------------------------------
# BacktestResult metrics tests
# ---------------------------------------------------------------------------


class TestBacktestResultMetrics:
    """Test backtest result metric computations."""

    def test_backtest_result_metrics(self) -> None:
        """Verify Sharpe, drawdown, and win rate are computed correctly."""
        # Build a simple equity curve: starts at 10000, goes up, dips, recovers
        timestamps = pd.date_range("2026-01-01", periods=20, freq="D")
        equity_values = [
            10000, 10050, 10120, 10080, 10200,
            10300, 10250, 10150, 10000, 10100,
            10200, 10350, 10400, 10380, 10500,
            10600, 10550, 10700, 10800, 10900,
        ]
        equity_curve = pd.Series(equity_values, index=timestamps, name="equity")

        # Build trades with known PnL
        trades = [
            {"pnl": 50, "edge": 0.05, "contracts": 10, "ticker": "A"},
            {"pnl": -20, "edge": 0.03, "contracts": 5, "ticker": "B"},
            {"pnl": 80, "edge": 0.08, "contracts": 15, "ticker": "C"},
            {"pnl": -10, "edge": 0.02, "contracts": 3, "ticker": "D"},
            {"pnl": 100, "edge": 0.10, "contracts": 20, "ticker": "E"},
        ]

        # Sharpe ratio
        daily_eq = equity_curve.resample("D").last().dropna()
        daily_returns = daily_eq.pct_change().dropna()
        sharpe = compute_sharpe(daily_returns)
        # Should be positive (equity generally trends up)
        assert sharpe > 0

        # Max drawdown
        max_dd = compute_max_drawdown(equity_curve)
        # Peak was 10300 at index 5, trough 10000 at index 8 -> ~2.9%
        assert 0 < max_dd < 1

        # Win rate: 3 wins out of 5
        wr = compute_win_rate(trades)
        assert wr == pytest.approx(0.6, abs=1e-10)

    def test_from_trades_and_equity(self) -> None:
        """BacktestResult.from_trades_and_equity should populate all fields."""
        timestamps = pd.date_range("2026-01-01", periods=5, freq="D")
        equity_curve = pd.Series(
            [10000, 10100, 10050, 10200, 10300],
            index=timestamps,
            name="equity",
        )
        trades = [
            {"pnl": 100, "edge": 0.05, "contracts": 10, "ticker": "A"},
            {"pnl": -50, "edge": 0.03, "contracts": 5, "ticker": "B"},
        ]

        result = BacktestResult.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_curve,
            total_fees=Decimal("5.00"),
            predicted_probs=[0.7, 0.6],
            outcomes=[1, 0],
        )

        assert result.num_trades == 2
        assert result.total_fees == Decimal("5.00")
        assert result.win_rate == pytest.approx(0.5)
        assert result.brier_score < 1.0
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
