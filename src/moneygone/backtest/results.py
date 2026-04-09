"""Backtest result analytics and reporting.

Provides the :class:`BacktestResult` dataclass that holds all trade
records, equity curve, and computed performance metrics.  Helper
functions compute standard quant metrics (Sharpe, drawdown, Brier score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def compute_sharpe(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualised Sharpe ratio from daily returns.

    Parameters
    ----------
    daily_returns:
        Series of daily return fractions.
    risk_free_rate:
        Annual risk-free rate (default 0).
    periods_per_year:
        Number of trading periods per year.

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns 0.0 if insufficient data.
    """
    if len(daily_returns) < 2:
        return 0.0

    excess = daily_returns - risk_free_rate / periods_per_year
    mean = float(excess.mean())
    std = float(excess.std(ddof=1))

    if std == 0 or np.isnan(std):
        return 0.0

    return mean / std * np.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve.

    Parameters
    ----------
    equity_curve:
        Time series of portfolio equity values.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
    """
    if len(equity_curve) < 2:
        return 0.0

    values = equity_curve.values.astype(float)
    peak = np.maximum.accumulate(values)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peak > 0, (peak - values) / peak, 0.0)

    return float(np.nanmax(drawdowns))


def compute_win_rate(trades: list[dict[str, Any]]) -> float:
    """Compute the fraction of trades with positive PnL.

    Parameters
    ----------
    trades:
        List of trade dicts, each with a ``pnl`` key.

    Returns
    -------
    float
        Win rate as a fraction (0-1).
    """
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
    return wins / len(trades)


def compute_brier_score(
    predicted_probs: list[float],
    outcomes: list[int],
) -> float:
    """Compute Brier score for probability predictions.

    Parameters
    ----------
    predicted_probs:
        Model's predicted probabilities for YES outcome.
    outcomes:
        Actual outcomes (1 = YES, 0 = NO).

    Returns
    -------
    float
        Brier score (0 = perfect, 1 = worst possible).
    """
    if not predicted_probs or not outcomes:
        return 1.0

    if len(predicted_probs) != len(outcomes):
        raise ValueError("predicted_probs and outcomes must have same length")

    probs = np.array(predicted_probs, dtype=np.float64)
    actual = np.array(outcomes, dtype=np.float64)

    return float(np.mean((probs - actual) ** 2))


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    Holds all trade records, equity curve, and precomputed metrics.
    """

    trades: list[dict[str, Any]]
    """List of trade dicts with fill details and prediction context."""

    equity_curve: pd.Series
    """Portfolio equity indexed by timestamp."""

    daily_returns: pd.Series
    """Daily return fractions."""

    total_pnl: Decimal
    """Gross PnL before fees."""

    total_fees: Decimal
    """Total fees paid."""

    net_pnl: Decimal
    """Net PnL after fees."""

    sharpe_ratio: float
    """Annualised Sharpe ratio."""

    max_drawdown: float
    """Maximum drawdown as a fraction."""

    win_rate: float
    """Fraction of trades with positive PnL."""

    avg_edge_predicted: float
    """Average predicted edge across all trades."""

    avg_edge_realized: float
    """Average realized edge (actual PnL / contracts)."""

    brier_score: float
    """Brier score for model probability calibration."""

    num_trades: int
    """Total number of trades executed."""

    def summary(self) -> str:
        """Generate a human-readable performance summary.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"  Trades:             {self.num_trades}",
            f"  Total PnL:          ${self.total_pnl:,.2f}",
            f"  Total Fees:         ${self.total_fees:,.2f}",
            f"  Net PnL:            ${self.net_pnl:,.2f}",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.3f}",
            f"  Max Drawdown:       {self.max_drawdown:.2%}",
            f"  Win Rate:           {self.win_rate:.2%}",
            f"  Avg Edge Predicted: {self.avg_edge_predicted:.4f}",
            f"  Avg Edge Realized:  {self.avg_edge_realized:.4f}",
            f"  Brier Score:        {self.brier_score:.4f}",
            "-" * 60,
        ]

        if len(self.equity_curve) > 0:
            start_eq = float(self.equity_curve.iloc[0])
            end_eq = float(self.equity_curve.iloc[-1])
            total_return = (end_eq - start_eq) / start_eq if start_eq > 0 else 0
            lines.append(f"  Start Equity:       ${start_eq:,.2f}")
            lines.append(f"  End Equity:         ${end_eq:,.2f}")
            lines.append(f"  Total Return:       {total_return:.2%}")
            lines.append(f"  Period:             {self.equity_curve.index[0]} to {self.equity_curve.index[-1]}")

        if len(self.daily_returns) > 0:
            lines.append(f"  Avg Daily Return:   {float(self.daily_returns.mean()):.4%}")
            lines.append(f"  Daily Return StdDev:{float(self.daily_returns.std()):.4%}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trades to a DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            One row per trade with all recorded fields.
        """
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)

        # Ensure timestamp column exists and is properly typed
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    @classmethod
    def from_trades_and_equity(
        cls,
        trades: list[dict[str, Any]],
        equity_curve: pd.Series,
        total_fees: Decimal,
        predicted_probs: list[float] | None = None,
        outcomes: list[int] | None = None,
    ) -> BacktestResult:
        """Construct a BacktestResult and compute all metrics.

        Convenience factory that calculates all metrics from raw
        trade data and equity curve.

        Parameters
        ----------
        trades:
            List of trade dicts, each with at least ``pnl``, ``edge``,
            and ``contracts`` keys.
        equity_curve:
            Time series of equity values indexed by timestamp.
        total_fees:
            Total fees paid across all trades.
        predicted_probs:
            Model probabilities for Brier score computation.
        outcomes:
            Actual outcomes (0/1) for Brier score computation.
        """
        # Daily returns from equity curve
        if len(equity_curve) >= 2:
            equity_float = equity_curve.astype(float)
            daily_equity = equity_float.resample("D").last().dropna()
            if len(daily_equity) >= 2:
                daily_returns = daily_equity.pct_change().dropna()
            else:
                daily_returns = pd.Series(dtype=float)
        else:
            daily_returns = pd.Series(dtype=float)

        # PnL computation
        total_pnl = Decimal(str(sum(float(t.get("pnl", 0)) for t in trades)))
        net_pnl = total_pnl - total_fees

        # Metrics
        sharpe = compute_sharpe(daily_returns)
        max_dd = compute_max_drawdown(equity_curve)
        wr = compute_win_rate(trades)

        # Edge analysis
        edges_predicted = [float(t.get("edge", 0)) for t in trades]
        avg_edge_pred = sum(edges_predicted) / len(edges_predicted) if edges_predicted else 0.0

        # Realized edge = actual PnL per contract
        realized_edges: list[float] = []
        for t in trades:
            contracts = int(t.get("contracts", 1))
            pnl = float(t.get("pnl", 0))
            if contracts > 0:
                realized_edges.append(pnl / contracts)
        avg_edge_real = sum(realized_edges) / len(realized_edges) if realized_edges else 0.0

        # Brier score
        brier = 1.0
        if predicted_probs and outcomes:
            brier = compute_brier_score(predicted_probs, outcomes)

        return cls(
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            total_pnl=total_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=wr,
            avg_edge_predicted=avg_edge_pred,
            avg_edge_realized=avg_edge_real,
            brier_score=brier,
            num_trades=len(trades),
        )
