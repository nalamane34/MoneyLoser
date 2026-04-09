"""Feature computation pipeline with dependency resolution and caching."""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from moneygone.features.base import Feature, FeatureContext
from moneygone.features.registry import FeatureRegistry

log = structlog.get_logger()


class FeaturePipeline:
    """Orchestrates feature computation in dependency-safe order.

    The pipeline resolves the topological order of features from the
    registry and caches intermediate results within a single ``compute``
    call so that shared dependencies are only evaluated once.

    Parameters:
        features: List of :class:`Feature` instances to include.
        store: DataStore handle passed into every :class:`FeatureContext`.
    """

    def __init__(self, features: list[Feature], store: Any = None) -> None:
        self._store = store

        # Build a private registry and resolve order
        self._registry = FeatureRegistry()
        for feat in features:
            self._registry.register(feat)
        self._ordered: list[Feature] = self._registry.resolve_order()
        log.info(
            "pipeline_initialized",
            n_features=len(self._ordered),
            order=[f.name for f in self._ordered],
        )

    # ------------------------------------------------------------------
    # Single observation
    # ------------------------------------------------------------------

    def compute(self, context: FeatureContext) -> dict[str, float]:
        """Compute all features for a single observation.

        Returns a dict mapping feature name to value.  Features that
        return ``None`` (missing data) are excluded from the result.
        """
        if context.store is None and self._store is not None:
            context.store = self._store
        cache: dict[str, float | None] = {}
        errors = 0

        for feat in self._ordered:
            try:
                value = feat.compute(context)
                cache[feat.name] = value
            except Exception:
                errors += 1
                log.warning(
                    "feature_compute_error",
                    feature=feat.name,
                    ticker=context.ticker,
                    exc_info=True,
                )
                cache[feat.name] = None

        result = {k: v for k, v in cache.items() if v is not None}

        if errors > 0:
            log.warning(
                "feature_pipeline.errors_summary",
                ticker=context.ticker,
                errors=errors,
                total=len(self._ordered),
                succeeded=len(result),
            )
        if len(result) == 0 and len(self._ordered) > 0:
            log.error(
                "feature_pipeline.all_features_failed",
                ticker=context.ticker,
                total=len(self._ordered),
            )

        return result

    # ------------------------------------------------------------------
    # Batch (training / backtesting)
    # ------------------------------------------------------------------

    def compute_batch(self, contexts: list[FeatureContext]) -> pd.DataFrame:
        """Compute all features for a batch of observations.

        Returns a :class:`pandas.DataFrame` with one row per context and
        one column per feature.  Missing values are ``NaN``.
        """
        if not contexts:
            return pd.DataFrame(columns=[f.name for f in self._ordered])

        rows: list[dict[str, float | None]] = []
        for ctx in contexts:
            row = self._compute_single_cached(ctx)
            rows.append(row)

        df = pd.DataFrame(rows, columns=[f.name for f in self._ordered])
        log.info(
            "pipeline_batch_complete",
            n_contexts=len(contexts),
            n_features=len(self._ordered),
            missing_pct=round(df.isna().mean().mean() * 100, 2),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_single_cached(self, context: FeatureContext) -> dict[str, float | None]:
        """Compute all features for a single context, caching intermediates."""
        if context.store is None and self._store is not None:
            context.store = self._store
        cache: dict[str, float | None] = {}
        for feat in self._ordered:
            try:
                cache[feat.name] = feat.compute(context)
            except Exception:
                log.warning(
                    "feature_compute_error",
                    feature=feat.name,
                    ticker=context.ticker,
                    exc_info=True,
                )
                cache[feat.name] = None
        return cache

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names."""
        return [f.name for f in self._ordered]

    def __repr__(self) -> str:
        return f"<FeaturePipeline features={len(self._ordered)}>"
