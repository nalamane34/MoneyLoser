"""Kill switch for live closer strategies (resolution sniper + live event edge).

Tracks consecutive losses across both strategies.  If the loss streak
reaches a configured threshold (default: 4), both strategies are paused
for a configurable cooldown period (default: 12 hours).

The kill switch is shared between ResolutionSniper and LiveEventEdge so
that losses from either strategy count toward the same streak.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from moneygone.utils.time import now_utc

log = structlog.get_logger(__name__)


@dataclass
class KillSwitchConfig:
    """Configuration for the closer kill switch."""

    max_consecutive_losses: int = 4
    cooldown_hours: float = 12.0


@dataclass
class CloserTradeRecord:
    """Record of a closer trade for kill-switch tracking."""

    ticker: str
    strategy: str  # "sniper" or "live_edge"
    entry_price: float
    side: str
    contracts: int
    executed_at: datetime
    resolved: bool = False
    won: bool | None = None


class CloserKillSwitch:
    """Monitors closer strategy losses and pauses trading on loss streaks.

    Thread-safe: uses an asyncio lock for state mutation.
    """

    def __init__(self, config: KillSwitchConfig | None = None) -> None:
        self._config = config or KillSwitchConfig()
        self._consecutive_losses = 0
        self._total_wins = 0
        self._total_losses = 0
        self._paused_until: datetime | None = None
        self._trade_log: list[CloserTradeRecord] = []
        self._lock = asyncio.Lock()

    @property
    def is_active(self) -> bool:
        """True if strategies are allowed to trade (not paused)."""
        if self._paused_until is None:
            return True
        now = now_utc()
        if now >= self._paused_until:
            # Cooldown expired — resume
            self._paused_until = None
            self._consecutive_losses = 0
            log.info(
                "kill_switch.cooldown_expired",
                msg="Closer strategies resumed after cooldown",
            )
            return True
        return False

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def paused_until(self) -> datetime | None:
        return self._paused_until

    def record_trade(
        self,
        ticker: str,
        strategy: str,
        entry_price: float,
        side: str,
        contracts: int,
    ) -> CloserTradeRecord:
        """Record a new closer trade.  Resolution is marked later."""
        record = CloserTradeRecord(
            ticker=ticker,
            strategy=strategy,
            entry_price=entry_price,
            side=side,
            contracts=contracts,
            executed_at=now_utc(),
        )
        self._trade_log.append(record)
        return record

    async def mark_resolution(self, ticker: str, won: bool) -> None:
        """Mark a trade's outcome and update the loss streak.

        Parameters
        ----------
        ticker:
            The Kalshi ticker that settled.
        won:
            True if the trade was profitable.
        """
        async with self._lock:
            # Find the most recent unresolved trade for this ticker
            for record in reversed(self._trade_log):
                if record.ticker == ticker and not record.resolved:
                    record.resolved = True
                    record.won = won
                    break

            if won:
                self._consecutive_losses = 0
                self._total_wins += 1
                log.info(
                    "kill_switch.win",
                    ticker=ticker,
                    streak=0,
                    total_wins=self._total_wins,
                    total_losses=self._total_losses,
                )
            else:
                self._consecutive_losses += 1
                self._total_losses += 1
                log.warning(
                    "kill_switch.loss",
                    ticker=ticker,
                    streak=self._consecutive_losses,
                    max_allowed=self._config.max_consecutive_losses,
                    total_wins=self._total_wins,
                    total_losses=self._total_losses,
                )

                if self._consecutive_losses >= self._config.max_consecutive_losses:
                    self._paused_until = now_utc() + timedelta(
                        hours=self._config.cooldown_hours
                    )
                    log.error(
                        "kill_switch.TRIGGERED",
                        consecutive_losses=self._consecutive_losses,
                        paused_until=self._paused_until.isoformat(),
                        cooldown_hours=self._config.cooldown_hours,
                        msg=(
                            f"Closer strategies PAUSED for "
                            f"{self._config.cooldown_hours}h after "
                            f"{self._consecutive_losses} consecutive losses"
                        ),
                    )

    def get_stats(self) -> dict:
        """Return current kill switch statistics."""
        return {
            "active": self.is_active,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self._config.max_consecutive_losses,
            "total_wins": self._total_wins,
            "total_losses": self._total_losses,
            "win_rate": (
                self._total_wins / (self._total_wins + self._total_losses)
                if (self._total_wins + self._total_losses) > 0
                else 0.0
            ),
            "paused_until": (
                self._paused_until.isoformat() if self._paused_until else None
            ),
            "total_trades": len(self._trade_log),
        }


def tiered_min_confidence(market_price: float) -> float:
    """Return the minimum confidence required based on contract price.

    Lower-priced contracts have better risk/reward (fewer wins needed
    to cover a loss), so we can trade with lower confidence.

    Price Range  | Min Confidence | Wins to Cover 1 Loss
    -------------|----------------|---------------------
    75-84c       | 90%            | 4-6
    85-89c       | 95%            | 7-9
    90-95c       | 97%            | 10-21
    """
    if market_price < 0.85:
        return 0.90
    elif market_price < 0.90:
        return 0.95
    else:
        return 0.97
