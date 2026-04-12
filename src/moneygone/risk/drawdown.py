"""Drawdown monitoring for circuit-breaker logic.

Tracks peak equity and current drawdown in real time.  Triggers a
circuit breaker when drawdown exceeds a configurable threshold,
halting all new trading activity until manually reset.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")


class DrawdownMonitor:
    """Tracks portfolio drawdown from peak equity.

    Thread-safe for single-writer usage (the main trading loop).
    """

    def __init__(self) -> None:
        self._peak: Decimal = _ZERO
        self._trough: Decimal = _ZERO
        self._current: Decimal = _ZERO
        self._max_drawdown_seen: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def peak_equity(self) -> Decimal:
        """Highest equity value observed since last reset."""
        return self._peak

    @property
    def trough_equity(self) -> Decimal:
        """Lowest equity value observed since peak."""
        return self._trough

    @property
    def current_equity(self) -> Decimal:
        """Most recently tracked equity value."""
        return self._current

    @property
    def max_drawdown_seen(self) -> float:
        """Largest drawdown percentage observed since last reset."""
        return self._max_drawdown_seen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, equity: Decimal) -> None:
        """Update with a new equity observation.

        Parameters
        ----------
        equity:
            Current total portfolio equity (cash + unrealized).
        """
        self._current = equity

        # Update peak
        if equity > self._peak:
            self._peak = equity
            self._trough = equity  # reset trough at new peak
        elif equity < self._trough or self._trough == _ZERO:
            self._trough = equity

        # Track worst drawdown
        dd = self.current_drawdown()
        if dd > self._max_drawdown_seen:
            self._max_drawdown_seen = dd
            logger.warning(
                "new_max_drawdown",
                drawdown_pct=round(dd * 100, 2),
                peak=str(self._peak),
                current=str(equity),
            )

    def current_drawdown(self) -> float:
        """Current drawdown as a fraction (0.0 to 1.0) from peak.

        Returns 0.0 if peak is zero or equity is at/above peak.
        """
        if self._peak <= _ZERO:
            return 0.0
        dd = (self._peak - self._current) / self._peak
        return max(0.0, float(dd))

    def is_circuit_breaker_triggered(self, max_drawdown_pct: float) -> bool:
        """Check if current drawdown exceeds the circuit breaker threshold.

        Parameters
        ----------
        max_drawdown_pct:
            Maximum allowed drawdown as a fraction (e.g. 0.15 for 15%).

        Returns
        -------
        bool
            True if trading should be halted.
        """
        triggered = self.current_drawdown() >= max_drawdown_pct
        if triggered:
            logger.critical(
                "circuit_breaker_triggered",
                drawdown_pct=round(self.current_drawdown() * 100, 2),
                threshold_pct=round(max_drawdown_pct * 100, 2),
                peak=str(self._peak),
                current=str(self._current),
            )
        return triggered

    def reset(self) -> None:
        """Reset all tracking state for a new session."""
        self._peak = _ZERO
        self._trough = _ZERO
        self._current = _ZERO
        self._max_drawdown_seen = 0.0
        logger.info("drawdown_monitor_reset")

    def reset_peak_to_current(self) -> None:
        """Reset the peak to the current equity level.

        Use after acknowledging losses to allow trading to resume
        without the circuit breaker being permanently triggered.
        """
        old_peak = self._peak
        self._peak = self._current
        self._trough = self._current
        self._max_drawdown_seen = 0.0
        logger.info(
            "drawdown_peak_reset",
            old_peak=str(old_peak),
            new_peak=str(self._current),
        )
