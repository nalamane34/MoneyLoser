"""Edge persistence and sensitivity analysis.

Provides tools to evaluate whether model edge is durable across time
periods, robust to fee assumptions, and stable across market regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from moneygone.data.store import DataStore
from moneygone.signals.edge import EdgeCalculator, EdgeResult

log = structlog.get_logger(__name__)


@dataclass
class EdgePersistenceReport:
    """Report on edge persistence across multiple time periods."""

    ticker: str
    periods: list[dict[str, Any]]
    """List of dicts with keys: start, end, mean_edge, median_edge,
    n_observations, pct_actionable."""
    overall_mean_edge: float
    is_persistent: bool
    """True if edge is positive and significant in all periods."""


@dataclass
class FeeSensitivityReport:
    """Report on how edge changes with fee assumptions."""

    base_fee_edge: float
    fee_scenarios: list[dict[str, Any]]
    """List of dicts: fee_multiplier, adjusted_edge, pct_actionable."""
    breakeven_fee_multiplier: float
    """Fee multiplier at which edge goes to zero."""


@dataclass
class RegimeSensitivityReport:
    """Report on edge variation across market regimes."""

    regimes: dict[str, dict[str, float]]
    """regime_name -> {mean_edge, std_edge, n_observations}."""
    best_regime: str
    worst_regime: str


class EdgeAnalyzer:
    """Analyzes the durability and sensitivity of model edge.

    Parameters
    ----------
    edge_calculator:
        Configured edge calculator for computing edge values.
    """

    def __init__(self, edge_calculator: EdgeCalculator) -> None:
        self._edge_calc = edge_calculator
        log.info("edge_analyzer.initialized")

    # ------------------------------------------------------------------
    # Edge persistence
    # ------------------------------------------------------------------

    async def analyze_edge_persistence(
        self,
        model: Any,
        store: DataStore,
        tickers: list[str],
        periods: list[tuple[datetime, datetime]],
    ) -> list[EdgePersistenceReport]:
        """Test whether edge persists across multiple time periods.

        Parameters
        ----------
        model:
            Model with a ``predict(features: dict) -> float`` method.
        store:
            Data store for fetching historical features and orderbooks.
        tickers:
            List of market tickers to analyze.
        periods:
            List of ``(start, end)`` datetime tuples defining periods.

        Returns
        -------
        list[EdgePersistenceReport]
            One report per ticker.
        """
        reports: list[EdgePersistenceReport] = []

        for ticker in tickers:
            period_results: list[dict[str, Any]] = []

            for start, end in periods:
                # Fetch features and market data for this period
                features = store.get_features_at(ticker, end)
                if not features:
                    period_results.append({
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "mean_edge": 0.0,
                        "median_edge": 0.0,
                        "n_observations": 0,
                        "pct_actionable": 0.0,
                    })
                    continue

                # Get trades in period for price data
                trades = store.get_trades_between(ticker, start, end)

                # Compute edges for each trade observation
                edges: list[float] = []
                actionable_count = 0

                for trade_data in trades:
                    ob_data = store.get_orderbook_at(ticker, trade_data["trade_time"])
                    if ob_data is None:
                        continue

                    # Reconstruct minimal orderbook for edge calculation
                    from moneygone.exchange.types import OrderbookLevel, OrderbookSnapshot
                    from moneygone.utils.time import parse_iso

                    yes_levels = tuple(
                        OrderbookLevel(
                            price=__import__("decimal").Decimal(str(lv[0])),
                            contracts=__import__("decimal").Decimal(str(lv[1])),
                        )
                        for lv in ob_data.get("yes_levels", [])
                    )
                    no_levels = tuple(
                        OrderbookLevel(
                            price=__import__("decimal").Decimal(str(lv[0])),
                            contracts=__import__("decimal").Decimal(str(lv[1])),
                        )
                        for lv in ob_data.get("no_levels", [])
                    )

                    snapshot_ts = ob_data.get("snapshot_time")
                    if isinstance(snapshot_ts, str):
                        snapshot_ts = parse_iso(snapshot_ts)
                    elif not isinstance(snapshot_ts, datetime):
                        from moneygone.utils.time import now_utc
                        snapshot_ts = now_utc()

                    ob = OrderbookSnapshot(
                        ticker=ticker,
                        yes_bids=yes_levels,
                        no_bids=no_levels,
                        seq=ob_data.get("seq", 0),
                        timestamp=snapshot_ts,
                    )

                    try:
                        prob = model.predict(features)
                        edge = self._edge_calc.compute_edge(prob, ob)
                        edges.append(edge.fee_adjusted_edge)
                        if edge.is_actionable:
                            actionable_count += 1
                    except Exception:
                        log.debug(
                            "edge_analyzer.compute_failed",
                            ticker=ticker,
                            exc_info=True,
                        )

                n_obs = len(edges)
                period_results.append({
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "mean_edge": round(float(np.mean(edges)), 6) if edges else 0.0,
                    "median_edge": round(float(np.median(edges)), 6) if edges else 0.0,
                    "n_observations": n_obs,
                    "pct_actionable": round(
                        actionable_count / n_obs * 100, 1
                    ) if n_obs > 0 else 0.0,
                })

            # Overall assessment
            all_means = [p["mean_edge"] for p in period_results if p["n_observations"] > 0]
            overall_mean = float(np.mean(all_means)) if all_means else 0.0
            is_persistent = all(m > 0 for m in all_means) and len(all_means) == len(periods)

            reports.append(
                EdgePersistenceReport(
                    ticker=ticker,
                    periods=period_results,
                    overall_mean_edge=round(overall_mean, 6),
                    is_persistent=is_persistent,
                )
            )

        log.info(
            "edge_persistence.complete",
            n_tickers=len(tickers),
            n_periods=len(periods),
        )
        return reports

    # ------------------------------------------------------------------
    # Fee sensitivity
    # ------------------------------------------------------------------

    def analyze_fee_sensitivity(
        self,
        edge_results: list[EdgeResult],
        fee_multipliers: list[float] | None = None,
    ) -> FeeSensitivityReport:
        """Analyze how edge changes under different fee assumptions.

        Parameters
        ----------
        edge_results:
            List of computed edge results.
        fee_multipliers:
            Fee multipliers to test (e.g. [0.5, 1.0, 1.5, 2.0]).

        Returns
        -------
        FeeSensitivityReport
        """
        if fee_multipliers is None:
            fee_multipliers = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

        if not edge_results:
            return FeeSensitivityReport(
                base_fee_edge=0.0,
                fee_scenarios=[],
                breakeven_fee_multiplier=0.0,
            )

        # Base case: fee_multiplier = 1.0
        base_edges = [e.fee_adjusted_edge for e in edge_results]
        base_raw = [e.raw_edge for e in edge_results]
        fee_impacts = [r - f for r, f in zip(base_raw, base_edges)]
        base_mean = float(np.mean(base_edges))

        scenarios: list[dict[str, Any]] = []
        breakeven_mult = float("inf")

        for mult in fee_multipliers:
            # Adjust edges by changing the fee component
            adjusted_edges = [
                raw - (fee * mult)
                for raw, fee in zip(base_raw, fee_impacts)
            ]
            mean_adjusted = float(np.mean(adjusted_edges))
            pct_actionable = sum(1 for e in adjusted_edges if e > 0) / len(adjusted_edges) * 100

            scenarios.append({
                "fee_multiplier": mult,
                "adjusted_edge": round(mean_adjusted, 6),
                "pct_actionable": round(pct_actionable, 1),
            })

            # Track breakeven
            if mean_adjusted <= 0 and breakeven_mult == float("inf"):
                # Interpolate to find exact breakeven
                if mult > 0:
                    prev_scenario = scenarios[-2] if len(scenarios) > 1 else None
                    if prev_scenario and prev_scenario["adjusted_edge"] > 0:
                        prev_mult = prev_scenario["fee_multiplier"]
                        prev_edge = prev_scenario["adjusted_edge"]
                        # Linear interpolation
                        breakeven_mult = prev_mult + prev_edge / (
                            prev_edge - mean_adjusted
                        ) * (mult - prev_mult)
                    else:
                        breakeven_mult = mult

        if breakeven_mult == float("inf"):
            breakeven_mult = fee_multipliers[-1] if fee_multipliers else 0.0

        return FeeSensitivityReport(
            base_fee_edge=round(base_mean, 6),
            fee_scenarios=scenarios,
            breakeven_fee_multiplier=round(breakeven_mult, 4),
        )

    # ------------------------------------------------------------------
    # Regime sensitivity
    # ------------------------------------------------------------------

    def analyze_regime_sensitivity(
        self,
        model: Any,
        features_by_regime: dict[str, list[dict[str, float]]],
    ) -> RegimeSensitivityReport:
        """Analyze edge by market regime.

        Parameters
        ----------
        model:
            Model with ``predict(features: dict) -> float``.
        features_by_regime:
            Mapping of regime name to list of feature dicts.

        Returns
        -------
        RegimeSensitivityReport
        """
        regime_stats: dict[str, dict[str, float]] = {}

        for regime_name, feature_sets in features_by_regime.items():
            if not feature_sets:
                regime_stats[regime_name] = {
                    "mean_edge": 0.0,
                    "std_edge": 0.0,
                    "n_observations": 0,
                }
                continue

            predictions: list[float] = []
            for features in feature_sets:
                try:
                    prob = model.predict(features)
                    predictions.append(prob)
                except Exception:
                    continue

            if predictions:
                # Edge approximation: distance from 0.5 (fair odds)
                edges = [abs(p - 0.5) for p in predictions]
                regime_stats[regime_name] = {
                    "mean_edge": round(float(np.mean(edges)), 6),
                    "std_edge": round(float(np.std(edges)), 6),
                    "n_observations": len(predictions),
                }
            else:
                regime_stats[regime_name] = {
                    "mean_edge": 0.0,
                    "std_edge": 0.0,
                    "n_observations": 0,
                }

        # Find best/worst regimes
        valid_regimes = {
            k: v
            for k, v in regime_stats.items()
            if v["n_observations"] > 0
        }
        if valid_regimes:
            best_regime = max(valid_regimes, key=lambda k: valid_regimes[k]["mean_edge"])
            worst_regime = min(valid_regimes, key=lambda k: valid_regimes[k]["mean_edge"])
        else:
            best_regime = "none"
            worst_regime = "none"

        log.info(
            "regime_sensitivity.complete",
            n_regimes=len(regime_stats),
            best_regime=best_regime,
            worst_regime=worst_regime,
        )

        return RegimeSensitivityReport(
            regimes=regime_stats,
            best_regime=best_regime,
            worst_regime=worst_regime,
        )
