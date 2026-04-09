"""Crypto data sub-package: exchange feeds and signal processing."""

from moneygone.data.crypto.ccxt_feed import (
    CryptoDataFeed,
    CryptoOrderbook,
    CryptoTrade,
    FundingRate,
    OpenInterestSnapshot,
)
from moneygone.data.crypto.processor import CryptoProcessor, WhaleTradeAlert

__all__ = [
    "CryptoDataFeed",
    "CryptoOrderbook",
    "CryptoProcessor",
    "CryptoTrade",
    "FundingRate",
    "OpenInterestSnapshot",
    "WhaleTradeAlert",
]
