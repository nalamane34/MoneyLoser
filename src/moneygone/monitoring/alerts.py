"""Alert manager with rate-limiting and optional webhook delivery.

All alerts are logged via structlog.  Optionally, alerts above a severity
threshold can be forwarded to an external webhook (Slack, PagerDuty, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
import structlog

from moneygone.config import MonitoringConfig

log = structlog.get_logger(__name__)

# Alert types recognized by the system
AlertType = Literal[
    "drift_detected",
    "calibration_degraded",
    "circuit_breaker",
    "large_loss",
    "regime_change",
    "generic",
]


@dataclass
class _AlertRecord:
    """Internal record for rate-limiting identical alerts."""

    alert_type: str
    last_sent: float  # monotonic timestamp
    count: int = 0


class AlertManager:
    """Manages alert emission with rate-limiting and optional webhook.

    Parameters
    ----------
    config:
        Monitoring configuration (contains webhook URL).
    rate_limit_seconds:
        Minimum seconds between alerts of the same type/message.
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        rate_limit_seconds: float = 300.0,
    ) -> None:
        self._webhook_url = config.alert_webhook_url if config else None
        self._rate_limit = rate_limit_seconds
        self._recent_alerts: dict[str, _AlertRecord] = {}
        self._http_client: httpx.AsyncClient | None = None

        log.info(
            "alert_manager.initialized",
            webhook_configured=self._webhook_url is not None,
            rate_limit_seconds=rate_limit_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        level: Literal["info", "warning", "critical"],
        message: str,
        data: dict[str, Any] | None = None,
        alert_type: AlertType = "generic",
    ) -> None:
        """Emit an alert.

        The alert is always logged.  If a webhook URL is configured and
        the rate limiter permits, it is also sent to the webhook.

        Parameters
        ----------
        level:
            Severity level.
        message:
            Human-readable alert message.
        data:
            Optional structured data payload.
        alert_type:
            Categorization for rate-limiting.
        """
        data = data or {}

        # Rate limiting: suppress duplicate alerts within the window
        alert_key = f"{alert_type}:{message}"
        if self._is_rate_limited(alert_key):
            log.debug(
                "alert.rate_limited",
                alert_type=alert_type,
                message=message,
            )
            return

        # Always log
        log_method = getattr(log, level, log.info)
        log_method(
            "alert.fired",
            alert_type=alert_type,
            message=message,
            **data,
        )

        # Track for rate limiting
        self._recent_alerts[alert_key] = _AlertRecord(
            alert_type=alert_type,
            last_sent=time.monotonic(),
            count=self._recent_alerts.get(alert_key, _AlertRecord(alert_type, 0.0)).count + 1,
        )

        # Optionally send to webhook
        if self._webhook_url:
            await self._send_webhook(level, message, data, alert_type)

    async def close(self) -> None:
        """Close the HTTP client used for webhook delivery."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def alert_drift_detected(
        self,
        metric_name: str,
        metric_value: float,
        severity: str,
    ) -> None:
        """Send a drift detection alert."""
        await self.send_alert(
            level="critical" if severity == "critical" else "warning",
            message=f"Model drift detected: {metric_name}={metric_value:.4f}",
            data={"metric_name": metric_name, "metric_value": metric_value},
            alert_type="drift_detected",
        )

    async def alert_calibration_degraded(
        self,
        ece: float,
        threshold: float,
    ) -> None:
        """Send a calibration degradation alert."""
        await self.send_alert(
            level="warning",
            message=f"Calibration degraded: ECE={ece:.4f} > {threshold:.4f}",
            data={"ece": ece, "threshold": threshold},
            alert_type="calibration_degraded",
        )

    async def alert_circuit_breaker(
        self,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send a circuit breaker activation alert."""
        await self.send_alert(
            level="critical",
            message=f"Circuit breaker triggered: {reason}",
            data=details or {},
            alert_type="circuit_breaker",
        )

    async def alert_large_loss(
        self,
        loss_amount: float,
        ticker: str,
    ) -> None:
        """Send a large loss alert."""
        await self.send_alert(
            level="critical",
            message=f"Large loss on {ticker}: ${loss_amount:.2f}",
            data={"loss_amount": loss_amount, "ticker": ticker},
            alert_type="large_loss",
        )

    async def alert_regime_change(
        self,
        from_regime: str,
        to_regime: str,
        volatility: float,
    ) -> None:
        """Send a regime change alert."""
        level: Literal["info", "warning", "critical"] = "info"
        if to_regime == "crisis":
            level = "critical"
        elif to_regime == "high_vol":
            level = "warning"

        await self.send_alert(
            level=level,
            message=f"Regime change: {from_regime} -> {to_regime}",
            data={
                "from_regime": from_regime,
                "to_regime": to_regime,
                "volatility": volatility,
            },
            alert_type="regime_change",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_rate_limited(self, alert_key: str) -> bool:
        """Check if an alert with this key was sent recently."""
        record = self._recent_alerts.get(alert_key)
        if record is None:
            return False
        elapsed = time.monotonic() - record.last_sent
        return elapsed < self._rate_limit

    async def _send_webhook(
        self,
        level: str,
        message: str,
        data: dict[str, Any],
        alert_type: str,
    ) -> None:
        """Send alert payload to the configured webhook URL."""
        if not self._webhook_url:
            return

        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=10.0)

        payload = {
            "level": level,
            "alert_type": alert_type,
            "message": message,
            "data": data,
        }

        try:
            response = await self._http_client.post(
                self._webhook_url, json=payload
            )
            if response.status_code >= 400:
                log.warning(
                    "alert.webhook_error",
                    status=response.status_code,
                    response=response.text[:200],
                )
        except httpx.HTTPError as exc:
            log.warning("alert.webhook_failed", error=str(exc))
