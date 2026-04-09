"""Custom exceptions for the Kalshi exchange layer."""

from __future__ import annotations


class KalshiAPIError(Exception):
    """Base exception for all Kalshi API errors.

    Attributes:
        status_code: HTTP status code from the API response, if available.
        response_body: Raw response body, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class AuthError(KalshiAPIError):
    """Raised when authentication fails (401/403) or key loading errors occur."""


class RateLimitError(KalshiAPIError):
    """Raised when the API rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        status_code: int | None = 429,
        response_body: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message, status_code=status_code, response_body=response_body
        )


class OrderError(KalshiAPIError):
    """Raised when an order operation fails (create, cancel, batch cancel)."""

    def __init__(
        self,
        message: str,
        *,
        order_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.order_id = order_id
        super().__init__(
            message, status_code=status_code, response_body=response_body
        )


class WebSocketError(KalshiAPIError):
    """Raised when a WebSocket connection or message handling error occurs."""
