"""Crypto signal processor: z-scores, OI changes, whale detection, basis spread.

Operates on the dataclasses produced by
:class:`~moneygone.data.crypto.ccxt_feed.CryptoDataFeed` to derive
features useful for prediction-market models.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import structlog

from moneygone.data.crypto.ccxt_feed import (
    CryptoTrade,
    FundingRate,
    OpenInterestSnapshot,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WhaleTradeAlert:
    """A trade whose notional value exceeds the whale threshold."""

    trade: CryptoTrade
    notional_usd: float


class CryptoProcessor:
    """Stateless helper for deriving crypto-market signals."""

    # ------------------------------------------------------------------
    # Funding rate z-score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_funding_zscore(
        rates: list[FundingRate],
        lookback: int | None = None,
    ) -> float:
        """Compute the z-score of the most recent funding rate.

        Parameters
        ----------
        rates:
            Chronologically ordered funding-rate observations for a single
            symbol.  The last entry is treated as the current rate.
        lookback:
            Number of historical rates to use for the rolling mean/std.
            ``None`` means use the entire series.

        Returns
        -------
        float
            Z-score of the latest rate relative to the lookback window.
            Returns ``0.0`` if insufficient data.
        """
        if len(rates) < 2:
            return 0.0

        window = rates if lookback is None else rates[-lookback:]
        values = [r.rate for r in window]
        current = values[-1]
        historical = values[:-1]

        if not historical:
            return 0.0

        mu = statistics.mean(historical)
        sigma = statistics.stdev(historical) if len(historical) > 1 else 0.0

        if sigma == 0.0:
            return 0.0

        return (current - mu) / sigma

    # ------------------------------------------------------------------
    # Open-interest change
    # ------------------------------------------------------------------

    @staticmethod
    def compute_oi_change(
        snapshots: list[OpenInterestSnapshot],
        lookback: int | None = None,
    ) -> float:
        """Compute relative change in open interest.

        Parameters
        ----------
        snapshots:
            Chronologically ordered OI snapshots.
        lookback:
            If given, compare the latest value to the value *lookback*
            entries ago.  Otherwise compare to the first entry.

        Returns
        -------
        float
            Fractional change (e.g. ``0.05`` = 5% increase).  Returns
            ``0.0`` if insufficient data.
        """
        if len(snapshots) < 2:
            return 0.0

        current = snapshots[-1].value
        if lookback is not None and lookback < len(snapshots):
            baseline = snapshots[-lookback].value
        else:
            baseline = snapshots[0].value

        if baseline == 0.0:
            return 0.0

        return (current - baseline) / baseline

    # ------------------------------------------------------------------
    # Whale trade detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_whale_trades(
        trades: list[CryptoTrade],
        threshold_usd: float = 100_000.0,
    ) -> list[WhaleTradeAlert]:
        """Return trades whose notional value exceeds *threshold_usd*.

        Notional is computed as ``price * amount``.
        """
        alerts: list[WhaleTradeAlert] = []
        for t in trades:
            notional = t.price * t.amount
            if notional >= threshold_usd:
                alerts.append(WhaleTradeAlert(trade=t, notional_usd=notional))
        if alerts:
            logger.info(
                "crypto_processor.whale_trades_detected",
                count=len(alerts),
                threshold_usd=threshold_usd,
            )
        return alerts

    # ------------------------------------------------------------------
    # Basis spread
    # ------------------------------------------------------------------

    @staticmethod
    def compute_basis_spread(
        spot_price: float,
        futures_price: float,
    ) -> float:
        """Compute annualised basis spread between spot and futures.

        A positive value means futures trade at a premium (contango).
        A negative value means backwardation.

        The returned value is a simple ratio ``(futures - spot) / spot``,
        *not* annualised (caller should annualise based on time to expiry).

        Returns ``0.0`` if spot price is zero.
        """
        if spot_price == 0.0:
            return 0.0
        return (futures_price - spot_price) / spot_price

    # ------------------------------------------------------------------
    # Aggregated sentiment score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sentiment_score(
        funding_zscore: float,
        oi_change: float,
        basis_spread: float,
    ) -> float:
        """Combine individual signals into a single sentiment score in [-1, 1].

        Positive values indicate bullish sentiment, negative bearish.
        The score is a weighted average passed through ``tanh`` to bound it.
        """
        raw = 0.4 * funding_zscore + 0.3 * (oi_change * 10) + 0.3 * (basis_spread * 100)
        return math.tanh(raw)
