"""Feature registry with topological dependency resolution."""

from __future__ import annotations

from collections import defaultdict, deque

import structlog

from moneygone.features.base import Feature

log = structlog.get_logger()


class CyclicDependencyError(Exception):
    """Raised when feature dependencies contain a cycle."""


class FeatureRegistry:
    """Central registry for all features.

    Supports registration, lookup, and topological ordering based on
    declared inter-feature dependencies.
    """

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, feature: Feature) -> None:
        """Register a feature instance.

        Raises ``ValueError`` if a feature with the same name is already
        registered.
        """
        if feature.name in self._features:
            raise ValueError(
                f"Feature '{feature.name}' is already registered. "
                "Use a unique name for each feature."
            )
        self._features[feature.name] = feature
        log.debug("feature_registered", name=feature.name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Feature:
        """Retrieve a registered feature by name.

        Raises ``KeyError`` if not found.
        """
        try:
            return self._features[name]
        except KeyError:
            raise KeyError(
                f"Feature '{name}' not found. "
                f"Registered: {list(self._features.keys())}"
            ) from None

    @property
    def all_features(self) -> dict[str, Feature]:
        """Return a copy of all registered features."""
        return dict(self._features)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def resolve_order(self) -> list[Feature]:
        """Return features in dependency-safe topological order.

        Uses Kahn's algorithm.  Raises :class:`CyclicDependencyError` if
        the dependency graph contains a cycle.
        """
        # Build adjacency list and in-degree map
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for name in self._features:
            in_degree.setdefault(name, 0)

        for name, feat in self._features.items():
            for dep in feat.dependencies:
                if dep not in self._features:
                    raise KeyError(
                        f"Feature '{name}' depends on '{dep}' which is not registered."
                    )
                dependents[dep].append(name)
                in_degree[name] += 1

        # Seed queue with features that have no dependencies
        queue: deque[str] = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)

        ordered: list[Feature] = []
        while queue:
            current = queue.popleft()
            ordered.append(self._features[current])
            for dep_name in dependents[current]:
                in_degree[dep_name] -= 1
                if in_degree[dep_name] == 0:
                    queue.append(dep_name)

        if len(ordered) != len(self._features):
            visited = {f.name for f in ordered}
            remaining = set(self._features.keys()) - visited
            raise CyclicDependencyError(
                f"Cyclic dependency detected among features: {remaining}"
            )

        return ordered

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features
