"""RSA-PSS authentication for the Kalshi API.

Signs each request with: ``timestamp_ms + method + path`` using RSA-PSS
(SHA-256, MGF1-SHA-256, salt_length=DIGEST_LENGTH).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from moneygone.exchange.errors import AuthError

log = structlog.get_logger(__name__)

_HASH_ALGORITHM = hashes.SHA256()
_DIGEST_LENGTH = _HASH_ALGORITHM.digest_size  # 32 bytes


class KalshiAuth:
    """Generates per-request authentication headers for the Kalshi API.

    Parameters:
        api_key_id: The public API key identifier.
        private_key_path: Path to the PEM-encoded RSA private key file.
    """

    def __init__(self, api_key_id: str, private_key_path: Path) -> None:
        self._api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)
        log.info("kalshi_auth.initialized", api_key_id=api_key_id)

    # ------------------------------------------------------------------
    # Private key loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_private_key(path: Path) -> rsa.RSAPrivateKey:
        """Load an RSA private key from a PEM file."""
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise AuthError(f"Private key file not found: {resolved}")

        try:
            pem_data = resolved.read_bytes()
            key = serialization.load_pem_private_key(pem_data, password=None)
        except Exception as exc:
            raise AuthError(f"Failed to load private key from {resolved}: {exc}") from exc

        if not isinstance(key, rsa.RSAPrivateKey):
            raise AuthError("Private key is not an RSA key")

        return key

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _sign(self, message: str) -> str:
        """Sign *message* using RSA-PSS and return the hex-encoded signature."""
        sig_bytes = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(_HASH_ALGORITHM),
                salt_length=_DIGEST_LENGTH,
            ),
            _HASH_ALGORITHM,
        )
        return base64.b64encode(sig_bytes).decode("ascii")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Return authentication headers for a single API request.

        Parameters:
            method: HTTP method in uppercase (``GET``, ``POST``, etc.).
            path: Request path **without** the scheme/host, e.g.
                ``/trade-api/v2/markets``.

        Returns:
            A dict with ``KALSHI-ACCESS-KEY``, ``KALSHI-ACCESS-SIGNATURE``,
            and ``KALSHI-ACCESS-TIMESTAMP`` headers.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        signature = self._sign(message)

        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def get_ws_query_params(self) -> dict[str, str]:
        """Return authentication query parameters for WebSocket connections.

        Uses ``GET`` and ``/trade-api/ws/v2`` as the canonical method/path.
        """
        return self.get_headers("GET", "/trade-api/ws/v2")
