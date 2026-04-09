"""Async helper patterns: retry, concurrency limiter, periodic runner, timeout."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# retry_async — exponential back-off decorator
# ---------------------------------------------------------------------------


def retry_async(
    func: Callable[..., Awaitable[T]] | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Decorator that retries an async function with exponential back-off.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts (excluding the initial call).
    base_delay:
        Delay in seconds before the first retry.
    max_delay:
        Upper bound on the back-off delay.
    retryable_exceptions:
        Exception types that trigger a retry.  All others propagate
        immediately.

    Usage::

        @retry_async
        async def fetch_data():
            ...

        @retry_async(max_retries=5, base_delay=0.5)
        async def fetch_data():
            ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2**attempt), max_delay)
                    # Add jitter: 0-25% of the delay
                    jitter = delay * random.uniform(0, 0.25)
                    total_delay = delay + jitter
                    log.warning(
                        "retry_async.retrying",
                        func=fn.__qualname__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=round(total_delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(total_delay)

            raise last_exc  # type: ignore[misc]

        return wrapper

    # Support both @retry_async and @retry_async(...)
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# gather_with_concurrency — bounded asyncio.gather
# ---------------------------------------------------------------------------


async def gather_with_concurrency(
    n: int, *coros: Coroutine[Any, Any, T]
) -> list[T]:
    """Run coroutines concurrently, limiting to *n* running at once.

    Behaves like ``asyncio.gather`` but uses a semaphore to cap the number
    of concurrently executing coroutines, preventing resource exhaustion.

    Parameters
    ----------
    n:
        Maximum number of coroutines to run simultaneously.
    *coros:
        Coroutines to execute.

    Returns
    -------
    list[T]
        Results in the same order as the input coroutines.
    """
    semaphore = asyncio.Semaphore(n)

    async def limited(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(limited(c) for c in coros)))


# ---------------------------------------------------------------------------
# periodic — recurring task decorator
# ---------------------------------------------------------------------------


def periodic(interval_seconds: float) -> Callable[
    [Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]
]:
    """Decorator that runs an async function repeatedly at a fixed interval.

    The decorated function becomes a long-running coroutine.  Call it once
    and it will loop forever (or until cancelled).  The interval is measured
    from the *start* of one invocation to the start of the next; if an
    invocation takes longer than *interval_seconds* the next call begins
    immediately.

    Usage::

        @periodic(interval_seconds=60)
        async def poll_markets():
            ...

        # Start the loop (runs until task is cancelled)
        task = asyncio.create_task(poll_markets())
    """

    def decorator(fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            while True:
                try:
                    await fn(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "periodic.error",
                        func=fn.__qualname__,
                        interval=interval_seconds,
                    )
                await asyncio.sleep(interval_seconds)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# timeout_async — async context manager
# ---------------------------------------------------------------------------


class timeout_async:
    """Async context manager that raises ``asyncio.TimeoutError`` if the body
    takes longer than *seconds*.

    Usage::

        async with timeout_async(10.0):
            result = await long_running_operation()
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> timeout_async:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass

    def __call__(self, coro: Coroutine[Any, Any, T]) -> Awaitable[T]:
        """Alternative usage: ``result = await timeout_async(5.0)(coro())``."""
        return asyncio.wait_for(coro, timeout=self._seconds)

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run *coro* with the configured timeout."""
        return await asyncio.wait_for(coro, timeout=self._seconds)
