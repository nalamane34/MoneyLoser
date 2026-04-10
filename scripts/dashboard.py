#!/usr/bin/env python3
"""MoneyGone Live Trading Dashboard — Matrix/Cyberpunk terminal UI.

Real-time dashboard showing portfolio state, live trade activity,
model inference, and market coverage from the MoneyGone trading system.

Usage::

    python scripts/dashboard.py
    python scripts/dashboard.py --config config/default.yaml --overlay config/live.yaml
    python scripts/dashboard.py --remote ubuntu@3.231.150.172  # SSH mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from moneygone.config import load_config
from moneygone.exchange.rest_client import KalshiRestClient

# ---------------------------------------------------------------------------
# Color palette — matrix/cyberpunk
# ---------------------------------------------------------------------------

C_HEADER = "bold bright_green"
C_ACCENT = "bright_cyan"
C_PROFIT = "bright_green"
C_LOSS = "bright_red"
C_DIM = "dim green"
C_BORDER = "green"
C_VALUE = "bright_white"
C_WARN = "bright_yellow"
C_LABEL = "green"
C_TICKER = "bright_cyan"
C_BUY = "bright_green"
C_SELL = "bright_red"
C_NEUTRAL = "bright_white"


def _pnl_color(value: float) -> str:
    if value > 0:
        return C_PROFIT
    elif value < 0:
        return C_LOSS
    return C_NEUTRAL


def _format_money(value: Decimal | float, prefix: str = "$") -> str:
    v = float(value) if isinstance(value, Decimal) else value
    if abs(v) >= 1000:
        return f"{prefix}{v:,.2f}"
    return f"{prefix}{v:.2f}"


def _format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------


class DashboardState:
    """Holds all dashboard data, refreshed periodically."""

    def __init__(self) -> None:
        self.balance_available = Decimal(0)
        self.balance_total = Decimal(0)
        self.initial_balance: Decimal | None = None
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self.recent_fills: list[dict] = []
        self.settlements: list[dict] = []
        self.activity_log: deque[dict] = deque(maxlen=20)
        self.discovery_stats: dict = {}
        self.sports_stats: dict = {}
        self.last_refresh: datetime | None = None
        self.last_inference: dict | None = None
        self.error: str | None = None
        self.refresh_count = 0
        self.start_time = datetime.now(timezone.utc)

    @property
    def pnl(self) -> Decimal:
        if self.initial_balance is None:
            return Decimal(0)
        return self.balance_total - self.initial_balance

    @property
    def pnl_pct(self) -> float:
        if self.initial_balance is None or self.initial_balance == 0:
            return 0.0
        return float(self.pnl / self.initial_balance) * 100

    @property
    def uptime(self) -> str:
        delta = datetime.now(timezone.utc) - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


async def refresh_state(
    state: DashboardState,
    rest_client: KalshiRestClient,
    data_dir: Path,
) -> None:
    """Fetch latest data from Kalshi API and local files."""
    try:
        # API calls
        balance = await rest_client.get_balance()
        state.balance_available = balance.available
        state.balance_total = balance.total
        if state.initial_balance is None:
            state.initial_balance = balance.total

        # Positions
        positions = await rest_client.get_positions()
        state.positions = [
            {
                "ticker": p.ticker,
                "event_ticker": p.event_ticker,
                "yes_count": p.yes_count,
                "no_count": p.no_count,
                "realized_pnl": p.realized_pnl,
                "market_exposure": p.market_exposure,
            }
            for p in positions
            if p.position != 0
        ]

        # Recent fills
        fills = await rest_client.get_fills(limit=20)
        state.recent_fills = [
            {
                "ticker": f.ticker,
                "side": f.side.value,
                "action": f.action.value,
                "count": f.count,
                "price": f.price,
                "is_taker": f.is_taker,
                "time": f.created_time,
            }
            for f in fills
        ]

        # Open orders
        orders = await rest_client.get_orders(status="resting")
        state.open_orders = [
            {
                "ticker": o.ticker,
                "side": o.side.value,
                "action": o.action.value,
                "count": o.remaining_count,
                "price": o.price,
                "time": o.created_time,
            }
            for o in orders
        ]

        # Settlements
        settlements = await rest_client.get_settlements(limit=20)
        state.settlements = [
            {
                "ticker": s.ticker,
                "result": s.market_result.value,
                "revenue": s.revenue,
                "fee_cost": s.fee_cost,
                "time": s.settled_time,
            }
            for s in settlements
        ]

        # Discovery cache
        cache_path = data_dir / "discovered_markets.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                markets = cache.get("markets", [])
                cats: dict[str, int] = {}
                for m in markets:
                    cat = m.get("market_category", "unknown")
                    cats[cat] = cats.get(cat, 0) + 1
                state.discovery_stats = {
                    "total": len(markets),
                    "refreshed_at": cache.get("refreshed_at", ""),
                    "categories": cats,
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse supervisor log for latest inference/trade decisions
        _parse_log_for_activity(state, data_dir)

        state.last_refresh = datetime.now(timezone.utc)
        state.refresh_count += 1
        state.error = None

    except Exception as exc:
        state.error = str(exc)[:80]


def _parse_log_for_activity(state: DashboardState, data_dir: Path) -> None:
    """Scan the supervisor log for recent trade decisions and activity."""
    log_paths = [
        Path("/tmp/supervisor.log"),
        data_dir / ".." / "logs" / "workers.log",
    ]
    log_path = None
    for p in log_paths:
        if p.exists():
            log_path = p
            break
    if log_path is None:
        return

    try:
        # Read last 100KB of log
        size = log_path.stat().st_size
        read_start = max(0, size - 100_000)
        with open(log_path, "r") as f:
            if read_start > 0:
                f.seek(read_start)
                f.readline()  # skip partial line
            lines = f.readlines()
    except Exception:
        return

    activity: list[dict] = []
    latest_inference = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip worker prefix if present
        if line.startswith("["):
            bracket_end = line.find("]")
            if bracket_end > 0:
                line = line[bracket_end + 1:].strip()

        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        event = entry.get("event", "")

        if event == "engine.trade_decision":
            latest_inference = {
                "ticker": entry.get("ticker", ""),
                "side": entry.get("side", ""),
                "contracts": entry.get("contracts", 0),
                "net_edge": entry.get("net_edge", 0),
                "model_prob": entry.get("model_prob", 0),
                "kelly": entry.get("kelly", 0),
                "time": entry.get("timestamp", ""),
            }
            activity.append({
                "type": "TRADE",
                "ticker": entry.get("ticker", "")[:25],
                "action": f"{'BUY' if entry.get('side') == 'yes' else 'SELL'}",
                "size": f"${entry.get('contracts', 0) * 1:.0f}",
                "confidence": f"{entry.get('net_edge', 0):.2f}",
                "time": entry.get("timestamp", ""),
            })

        elif event == "order_manager.submitted":
            activity.append({
                "type": "ORDER",
                "ticker": entry.get("ticker", "")[:25],
                "action": entry.get("status", "resting"),
                "size": "",
                "confidence": "",
                "time": entry.get("timestamp", ""),
            })

        elif event == "passive.timeout_cancel":
            activity.append({
                "type": "CANCEL",
                "ticker": entry.get("order_id", "")[:12],
                "action": "TIMEOUT",
                "size": f"{entry.get('timeout', 30):.0f}s",
                "confidence": "",
                "time": entry.get("timestamp", ""),
            })

        elif event == "passive.post_only_cross":
            activity.append({
                "type": "REJECT",
                "ticker": entry.get("ticker", "")[:25],
                "action": "CROSS",
                "size": "",
                "confidence": "",
                "time": entry.get("timestamp", ""),
            })

        elif event == "sports_snapshots.refreshed":
            state.sports_stats = {
                "matched": entry.get("matched", 0),
                "no_match": entry.get("no_match", 0),
                "unoriented": entry.get("unoriented", 0),
                "stale_line": entry.get("stale_line", 0),
                "leagues": entry.get("leagues", []),
            }

    if latest_inference:
        state.last_inference = latest_inference

    # Keep only latest 20 activities
    for act in activity[-20:]:
        state.activity_log.append(act)


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def _make_header(state: DashboardState) -> Panel:
    """Top banner with title and key metrics."""
    total = _format_money(state.balance_total)
    pnl_val = float(state.pnl)
    pnl_str = _format_money(pnl_val, prefix="+$" if pnl_val >= 0 else "-$")
    if pnl_val < 0:
        pnl_str = _format_money(abs(pnl_val), prefix="-$")
    else:
        pnl_str = _format_money(pnl_val, prefix="+$")
    pnl_pct = _format_pct(state.pnl_pct)
    pnl_c = _pnl_color(pnl_val)

    n_positions = len(state.positions)
    n_watched = state.discovery_stats.get("total", 0)

    header = Text()
    header.append("  TOTAL_CAPITAL: ", style=C_LABEL)
    header.append(total, style=C_VALUE)
    header.append("    REAL_TIME_PNL: ", style=C_LABEL)
    header.append(f"{pnl_str} ({pnl_pct})", style=pnl_c)
    header.append("    POSITIONS: ", style=C_LABEL)
    header.append(str(n_positions), style=C_VALUE)
    header.append("    MARKETS: ", style=C_LABEL)
    header.append(str(n_watched), style=C_VALUE)
    header.append("    UPTIME: ", style=C_LABEL)
    header.append(state.uptime, style=C_DIM)

    return Panel(
        header,
        title="[bold bright_green]═══ MONEYGONE // KALSHI PREDICTION MARKETS ═══[/]",
        border_style=C_BORDER,
        padding=(0, 1),
    )


def _make_activity_table(state: DashboardState) -> Panel:
    """Live L2 activity panel showing recent trade actions."""
    table = Table(
        show_header=True,
        header_style=C_ACCENT,
        border_style=C_BORDER,
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("TIME", style=C_DIM, width=8, no_wrap=True)
    table.add_column("MARKET", style=C_TICKER, ratio=2, no_wrap=True)
    table.add_column("AC.", style=C_NEUTRAL, width=7, justify="center")
    table.add_column("SIZ.", style=C_VALUE, width=6, justify="right")
    table.add_column("CONF.", style=C_VALUE, width=6, justify="right")

    # Show most recent activity first
    items = list(state.activity_log)
    items.reverse()

    for item in items[:16]:
        ts = item.get("time", "")
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                time_str = t.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                time_str = ts[-8:]
        else:
            time_str = ""

        action = item.get("action", "")
        action_style = C_BUY if action in ("BUY", "resting") else C_SELL if action in ("SELL", "TIMEOUT", "CROSS") else C_NEUTRAL

        table.add_row(
            time_str,
            item.get("ticker", ""),
            Text(action, style=action_style),
            item.get("size", ""),
            item.get("confidence", ""),
        )

    return Panel(
        table,
        title=f"[{C_ACCENT}]LIVE TRADE ACTIVITY[/]",
        border_style=C_BORDER,
    )


def _make_inference_panel(state: DashboardState) -> Panel:
    """Model inference panel showing the latest trade decision."""
    if state.last_inference is None:
        content = Text("  Waiting for model inference...", style=C_DIM)
    else:
        inf = state.last_inference
        ticker = inf.get("ticker", "UNKNOWN")
        side = inf.get("side", "")
        edge = inf.get("net_edge", 0)
        prob = inf.get("model_prob", 0)
        kelly = inf.get("kelly", 0)
        contracts = inf.get("contracts", 0)

        action = "BUY_YES" if side == "yes" else "BUY_NO"
        action_color = C_BUY if side == "yes" else C_SELL

        content = Text()
        content.append(f"  {ticker}\n\n", style=C_TICKER)
        content.append("  {\n", style=C_DIM)
        content.append('    "action": ', style=C_LABEL)
        content.append(f'"{action}"', style=action_color)
        content.append(",\n")
        content.append('    "confidence": ', style=C_LABEL)
        conf_val = edge * 10  # scale for display
        content.append(f"{min(conf_val, 0.99):.2f}", style=C_VALUE)
        content.append(",\n")
        content.append('    "edge_pct": ', style=C_LABEL)
        content.append(f"{edge:.1%}", style=C_PROFIT if edge > 0.05 else C_WARN)
        content.append(",\n")
        content.append('    "model_prob": ', style=C_LABEL)
        content.append(f"{prob:.1%}", style=C_VALUE)
        content.append(",\n")
        content.append('    "kelly_size": ', style=C_LABEL)
        content.append(f"{kelly:.1%}", style=C_VALUE)
        content.append(",\n")
        content.append('    "contracts": ', style=C_LABEL)
        content.append(str(contracts), style=C_VALUE)
        content.append(",\n")
        content.append('    "logic": ', style=C_LABEL)
        content.append('"Sharp Sportsbook +\n', style=C_ACCENT)
        content.append('     Pinnacle Edge Gap"', style=C_ACCENT)
        content.append("\n  }", style=C_DIM)

    return Panel(
        content,
        title=f"[{C_ACCENT}]MODEL INFERENCE[/]",
        border_style=C_BORDER,
    )


def _make_positions_table(state: DashboardState) -> Panel:
    """Current positions panel."""
    table = Table(
        show_header=True,
        header_style=C_ACCENT,
        border_style=C_BORDER,
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("MARKET", style=C_TICKER, ratio=2, no_wrap=True)
    table.add_column("SIDE", width=4, justify="center")
    table.add_column("QTY", style=C_VALUE, width=4, justify="right")
    table.add_column("P&L", width=8, justify="right")

    for pos in state.positions[:12]:
        ticker = pos["ticker"]
        if pos["yes_count"] > 0:
            side_text = Text("YES", style=C_BUY)
            qty = pos["yes_count"]
        elif pos["no_count"] > 0:
            side_text = Text("NO", style=C_SELL)
            qty = pos["no_count"]
        else:
            continue

        pnl = float(pos["realized_pnl"])
        pnl_text = Text(_format_money(pnl), style=_pnl_color(pnl))

        table.add_row(ticker[:28], side_text, str(qty), pnl_text)

    if not state.positions:
        table.add_row("No open positions", "", "", "")

    return Panel(
        table,
        title=f"[{C_ACCENT}]OPEN POSITIONS[/]",
        border_style=C_BORDER,
    )


def _make_fills_table(state: DashboardState) -> Panel:
    """Recent fills panel."""
    table = Table(
        show_header=True,
        header_style=C_ACCENT,
        border_style=C_BORDER,
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("TIME", style=C_DIM, width=8, no_wrap=True)
    table.add_column("MARKET", style=C_TICKER, ratio=2, no_wrap=True)
    table.add_column("SIDE", width=4, justify="center")
    table.add_column("QTY", style=C_VALUE, width=3, justify="right")
    table.add_column("PRICE", style=C_VALUE, width=6, justify="right")
    table.add_column("TYPE", style=C_DIM, width=5, justify="center")

    for fill in state.recent_fills[:10]:
        ts = fill.get("time")
        time_str = ts.strftime("%H:%M:%S") if ts else ""
        side = fill.get("side", "")
        side_text = Text(side.upper()[:3], style=C_BUY if side == "yes" else C_SELL)
        fill_type = "TAKER" if fill.get("is_taker") else "MAKER"
        type_style = C_WARN if fill.get("is_taker") else C_PROFIT

        table.add_row(
            time_str,
            fill["ticker"][:25],
            side_text,
            str(fill["count"]),
            f"${float(fill['price']):.2f}",
            Text(fill_type, style=type_style),
        )

    if not state.recent_fills:
        table.add_row("No recent fills", "", "", "", "", "")

    return Panel(
        table,
        title=f"[{C_ACCENT}]RECENT FILLS[/]",
        border_style=C_BORDER,
    )


def _make_market_radar(state: DashboardState) -> Panel:
    """Market category coverage radar."""
    table = Table(
        show_header=True,
        header_style=C_ACCENT,
        border_style=C_BORDER,
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("CATEGORY", style=C_LABEL, ratio=2)
    table.add_column("MKT.", style=C_VALUE, width=6, justify="right")
    table.add_column("STATUS", width=8, justify="center")

    cats = state.discovery_stats.get("categories", {})
    order = ["sports", "crypto", "weather", "financials", "politics", "economics", "companies", "unknown"]

    for cat in order:
        count = cats.get(cat, 0)
        if count == 0:
            status = Text("OFFLINE", style=C_DIM)
        elif count < 10:
            status = Text("LOW", style=C_WARN)
        else:
            status = Text("ACTIVE", style=C_PROFIT)

        table.add_row(cat.upper(), f"{count:,}", status)

    # Sports matching stats
    if state.sports_stats:
        table.add_row("", "", "")
        matched = state.sports_stats.get("matched", 0)
        table.add_row(
            Text("SPORTS MATCHED", style=C_ACCENT),
            str(matched),
            Text("LIVE" if matched > 0 else "WAIT", style=C_PROFIT if matched > 0 else C_WARN),
        )

    return Panel(
        table,
        title=f"[{C_ACCENT}]MARKET RADAR[/]",
        border_style=C_BORDER,
    )


def _make_settlements_table(state: DashboardState) -> Panel:
    """Recent settlements panel."""
    table = Table(
        show_header=True,
        header_style=C_ACCENT,
        border_style=C_BORDER,
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("TIME", style=C_DIM, width=8, no_wrap=True)
    table.add_column("MARKET", style=C_TICKER, ratio=2, no_wrap=True)
    table.add_column("RESULT", width=4, justify="center")
    table.add_column("P&L", width=8, justify="right")

    for s in state.settlements[:8]:
        ts = s.get("time")
        time_str = ts.strftime("%H:%M:%S") if ts else ""
        result = s.get("result", "")
        pnl = float(s.get("revenue", 0)) / 100  # revenue is in cents
        result_text = Text(
            result.upper()[:3],
            style=C_BUY if result in ("yes", "all_yes") else C_SELL,
        )
        pnl_text = Text(_format_money(pnl), style=_pnl_color(pnl))

        table.add_row(time_str, s["ticker"][:25], result_text, pnl_text)

    if not state.settlements:
        table.add_row("No settlements yet", "", "", "")

    return Panel(
        table,
        title=f"[{C_ACCENT}]SETTLEMENTS[/]",
        border_style=C_BORDER,
    )


def _make_footer(state: DashboardState) -> Text:
    """Bottom status bar."""
    footer = Text()
    footer.append("  CTRL+C to terminate", style=C_DIM)
    footer.append("  │  ", style=C_DIM)
    footer.append("MoneyGone live dashboard", style=C_LABEL)
    footer.append("  │  ", style=C_DIM)
    footer.append("Matrix/Cyberpunk", style=C_DIM)
    footer.append("  │  ", style=C_DIM)
    if state.error:
        footer.append(f"ERR: {state.error}", style=C_LOSS)
    else:
        footer.append(
            f"Last refresh: {state.last_refresh.strftime('%H:%M:%S') if state.last_refresh else 'never'}",
            style=C_DIM,
        )
    return footer


def build_layout(state: DashboardState) -> Layout:
    """Compose the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )

    # Body: left (activity + fills) | right (inference + radar + positions)
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )

    layout["left"].split_column(
        Layout(name="activity", ratio=2),
        Layout(name="fills", ratio=1),
    )

    layout["right"].split_column(
        Layout(name="inference", size=16),
        Layout(name="radar_and_positions", ratio=1),
    )

    layout["radar_and_positions"].split_row(
        Layout(name="radar"),
        Layout(name="positions"),
    )

    # Render panels
    layout["header"].update(_make_header(state))
    layout["activity"].update(_make_activity_table(state))
    layout["fills"].update(_make_fills_table(state))
    layout["inference"].update(_make_inference_panel(state))
    layout["radar"].update(_make_market_radar(state))
    layout["positions"].update(_make_positions_table(state))
    layout["footer"].update(_make_footer(state))

    return layout


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_dashboard(config_path: str, overlay_path: str) -> None:
    config = load_config(
        base_path=Path(config_path),
        overlay_path=Path(overlay_path),
    )

    data_dir = Path(config.data_dir)
    rest_client = KalshiRestClient(config.exchange)

    state = DashboardState()
    console = Console()

    # Initial fetch
    await refresh_state(state, rest_client, data_dir)

    with Live(
        build_layout(state),
        console=console,
        refresh_per_second=2,
        screen=True,
    ) as live:
        try:
            while True:
                await refresh_state(state, rest_client, data_dir)
                live.update(build_layout(state))
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            await rest_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MoneyGone Live Dashboard")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default="config/live.yaml")
    args = parser.parse_args()

    asyncio.run(run_dashboard(args.config, args.overlay))


if __name__ == "__main__":
    main()
