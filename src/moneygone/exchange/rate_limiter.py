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


class DualRateLimiter:
    """Pair of rate limiters separating order traffic from data reads.

    This prevents high-frequency data polling (orderbook snapshots, market
    discovery) from starving latency-sensitive order submission.

    Parameters:
        order_rps: Sustained rate for order endpoints (POST/DELETE /orders).
        data_rps: Sustained rate for data/read endpoints (GET requests, etc.).
        order_burst: Burst size for orders.  Defaults to ``order_rps``.
        data_burst: Burst size for data.  Defaults to ``data_rps``.
    """

    def __init__(
        self,
        order_rps: float = 10.0,
        data_rps: float = 30.0,
        order_burst: float | None = None,
        data_burst: float | None = None,
    ) -> None:
        self.orders = AsyncRateLimiter(order_rps, burst=order_burst)
        self.data = AsyncRateLimiter(data_rps, burst=data_burst)
        log.debug(
            "dual_rate_limiter.initialized",
            order_rps=order_rps,
            data_rps=data_rps,
        )

    async def acquire_order(self) -> None:
        """Acquire a token from the *order* bucket."""
        await self.orders.acquire()

    async def acquire_data(self) -> None:
        """Acquire a token from the *data* bucket."""
        await self.data.acquire()
