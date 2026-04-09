"""Forecast processing: model comparison, exceedance probabilities, revision tracking.

This module operates on :class:`~moneygone.data.weather.noaa.ForecastEnsemble`
objects and the :class:`~moneygone.data.store.DataStore` to produce
actionable signals for prediction-market weather contracts.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from moneygone.data.store import DataStore
from moneygone.data.weather.noaa import ForecastEnsemble

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Disagreement metrics between two ensemble forecasts.

    Attributes
    ----------
    mean_diff:
        Average absolute difference in ensemble means across valid times.
    max_diff:
        Maximum absolute difference in ensemble means.
    spread_ratio:
        Ratio of ensemble standard deviations (model_a / model_b), averaged
        across valid times.  Values far from 1.0 indicate one model is much
        more uncertain.
    correlation:
        Pearson correlation between the two mean series.
    """

    mean_diff: float
    max_diff: float
    spread_ratio: float
    correlation: float


@dataclass(frozen=True, slots=True)
class RevisionMetrics:
    """Metrics for how a forecast has changed between successive cycles.

    Attributes
    ----------
    init_time_previous:
        Initialization time of the earlier forecast cycle.
    init_time_current:
        Initialization time of the later forecast cycle.
    magnitude:
        Mean absolute revision in ensemble mean across overlapping valid times.
    direction:
        Mean signed revision (positive = current forecast is higher).
    max_revision:
        Largest absolute single-timestep revision.
    """

    init_time_previous: datetime
    init_time_current: datetime
    magnitude: float
    direction: float
    max_revision: float


class ForecastProcessor:
    """Stateless utilities for comparing and analysing ensemble forecasts."""

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_ensembles(
        model_a: ForecastEnsemble,
        model_b: ForecastEnsemble,
    ) -> ModelComparison:
        """Compare two ensemble forecasts that cover overlapping valid times.

        The comparison is performed over the intersection of valid times.  If
        there is no overlap the metrics are all zero.
        """
        times_a = {t: i for i, t in enumerate(model_a.valid_times)}
        indices: list[tuple[int, int]] = []
        for j, t in enumerate(model_b.valid_times):
            if t in times_a:
                indices.append((times_a[t], j))

        if not indices:
            logger.warning("forecast_processor.no_overlap")
            return ModelComparison(
                mean_diff=0.0, max_diff=0.0, spread_ratio=1.0, correlation=0.0
            )

        diffs: list[float] = []
        a_means: list[float] = []
        b_means: list[float] = []
        spread_ratios: list[float] = []
        for i_a, i_b in indices:
            ma = model_a.ensemble_mean[i_a]
            mb = model_b.ensemble_mean[i_b]
            diffs.append(abs(ma - mb))
            a_means.append(ma)
            b_means.append(mb)
            sa = model_a.ensemble_std[i_a]
            sb = model_b.ensemble_std[i_b]
            if sb > 0:
                spread_ratios.append(sa / sb)

        mean_diff = statistics.mean(diffs)
        max_diff = max(diffs)
        spread_ratio = statistics.mean(spread_ratios) if spread_ratios else 1.0
        correlation = _pearson(a_means, b_means)

        return ModelComparison(
            mean_diff=mean_diff,
            max_diff=max_diff,
            spread_ratio=spread_ratio,
            correlation=correlation,
        )

    # ------------------------------------------------------------------
    # Exceedance probability
    # ------------------------------------------------------------------

    @staticmethod
    def compute_exceedance_prob(
        ensemble: ForecastEnsemble,
        threshold: float,
        *,
        timestep_index: int | None = None,
    ) -> float:
        """Fraction of ensemble members exceeding *threshold*.

        Parameters
        ----------
        ensemble:
            The ensemble forecast.
        threshold:
            The value to compare against.
        timestep_index:
            If given, compute exceedance at a single valid time.  Otherwise
            compute the average exceedance across all valid times.
        """
        if not ensemble.member_values:
            return 0.0

        if timestep_index is not None:
            count = sum(
                1
                for member in ensemble.member_values
                if timestep_index < len(member) and member[timestep_index] > threshold
            )
            return count / len(ensemble.member_values)

        # Average across all timesteps.
        total_exceed = 0
        total_checks = 0
        n_times = len(ensemble.valid_times)
        for t_idx in range(n_times):
            for member in ensemble.member_values:
                if t_idx < len(member):
                    total_checks += 1
                    if member[t_idx] > threshold:
                        total_exceed += 1
        return total_exceed / total_checks if total_checks > 0 else 0.0

    # ------------------------------------------------------------------
    # Revision tracking
    # ------------------------------------------------------------------

    @staticmethod
    def compute_revision(
        current: ForecastEnsemble,
        previous: ForecastEnsemble,
    ) -> RevisionMetrics:
        """Compute revision metrics between two consecutive forecast cycles.

        Comparison is done over overlapping valid times.
        """
        times_prev = {t: i for i, t in enumerate(previous.valid_times)}
        revisions: list[float] = []
        for j, t in enumerate(current.valid_times):
            if t in times_prev:
                i = times_prev[t]
                revisions.append(
                    current.ensemble_mean[j] - previous.ensemble_mean[i]
                )

        if not revisions:
            return RevisionMetrics(
                init_time_previous=previous.init_time,
                init_time_current=current.init_time,
                magnitude=0.0,
                direction=0.0,
                max_revision=0.0,
            )

        magnitude = statistics.mean([abs(r) for r in revisions])
        direction = statistics.mean(revisions)
        max_revision = max(abs(r) for r in revisions)

        return RevisionMetrics(
            init_time_previous=previous.init_time,
            init_time_current=current.init_time,
            magnitude=magnitude,
            direction=direction,
            max_revision=max_revision,
        )

    # ------------------------------------------------------------------
    # Historical revision series from DataStore
    # ------------------------------------------------------------------

    @staticmethod
    def track_revisions(
        store: DataStore,
        location: str,
        variable: str,
        *,
        n_cycles: int = 6,
        as_of: datetime | None = None,
    ) -> list[RevisionMetrics]:
        """Return revision metrics for the last *n_cycles* forecast cycles.

        Loads forecasts from the :class:`DataStore` and computes pairwise
        revisions between consecutive initialization times.
        """
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        all_rows = store.get_forecasts_at(location, variable, as_of)
        if not all_rows:
            return []

        # Group by init_time.
        by_init: dict[datetime, list[dict]] = {}
        for row in all_rows:
            init_t = row["init_time"]
            by_init.setdefault(init_t, []).append(row)

        sorted_inits = sorted(by_init.keys())[-n_cycles:]
        if len(sorted_inits) < 2:
            return []

        def _rows_to_ensemble(rows: list[dict]) -> ForecastEnsemble:
            rows = sorted(rows, key=lambda r: r["valid_time"])
            return ForecastEnsemble(
                location_name=rows[0]["location_name"],
                lat=rows[0]["lat"],
                lon=rows[0]["lon"],
                variable=rows[0]["variable"],
                init_time=rows[0]["init_time"],
                valid_times=[r["valid_time"] for r in rows],
                member_values=[r["member_values"] for r in rows],
                ensemble_mean=[r["ensemble_mean"] for r in rows],
                ensemble_std=[r["ensemble_std"] for r in rows],
            )

        revisions: list[RevisionMetrics] = []
        for i in range(1, len(sorted_inits)):
            prev_ensemble = _rows_to_ensemble(by_init[sorted_inits[i - 1]])
            curr_ensemble = _rows_to_ensemble(by_init[sorted_inits[i]])
            revisions.append(
                ForecastProcessor.compute_revision(curr_ensemble, prev_ensemble)
            )

        return revisions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two equal-length sequences."""
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)
