"""Async token-bucket rate limiter.

Provides non-blocking throttling to stay within Kalshi API rate limits.
"""

from __future__ import annotations

import asyncio
import time

import structlog

log = structlog.get_logger(__name__)


class AsyncRateLimiter:
    """Token-bucket rate limiter for async code.

    Tokens are refilled continuously based on elapsed wall-clock time.

    Parameters:
        rps: Maximum requests per second (sustained rate).
        burst: Maximum burst size.  Defaults to ``rps`` (one second of burst).
    """

    def __init__(self, rps: float, burst: float | None = None) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")

        self._rps = rps
        self._burst = burst if burst is not None else rps
        self._tokens = self._burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        log.debug(
            "rate_limiter.initialized", rps=self._rps, burst=self._burst
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rps)
        self._last_refill = now

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one.

        This method is safe for concurrent use from multiple coroutines.
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Calculate how long to wait for the next token.
                deficit = 1.0 - self._tokens
                wait_seconds = deficit / self._rps

            log.debug("rate_limiter.waiting", wait_seconds=round(wait_seconds, 4))
            await asyncio.sleep(wait_seconds)

    @property
    def tokens(self) -> float:
        """Current (approximate) number of tokens available."""
        self._refill()
        return self._tokens
