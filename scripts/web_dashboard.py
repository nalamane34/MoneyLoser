#!/usr/bin/env python3
"""MoneyGone Web Dashboard — Matrix/Cyberpunk themed.

Serves the trading dashboard as a web page with auto-refresh.
Designed to sit behind a Cloudflare Tunnel with basic auth.

Usage::

    python scripts/web_dashboard.py
    python scripts/web_dashboard.py --port 8050 --config config/live.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiohttp import web

from moneygone.config import load_config
from moneygone.exchange.rest_client import KalshiRestClient

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

AUTH_USER = os.environ.get("DASH_USER", "jordan")
AUTH_PASS = os.environ.get("DASH_PASS", "green")


def _check_auth(request: web.Request) -> bool:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode()
        user, password = decoded.split(":", 1)
        return user == AUTH_USER and password == AUTH_PASS
    except Exception:
        return False


def _auth_required() -> web.Response:
    return web.Response(
        status=401,
        text="Unauthorized",
        headers={"WWW-Authenticate": 'Basic realm="MoneyGone"'},
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DashState:
    def __init__(self) -> None:
        self.balance_available = Decimal(0)
        self.balance_total = Decimal(0)
        self.initial_balance: Decimal | None = None
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self.recent_fills: list[dict] = []
        self.settlements: list[dict] = []
        self.discovery_stats: dict = {}
        self.activity: list[dict] = []
        self.last_refresh: datetime | None = None
        self.error: str | None = None
        self.start_time = datetime.now(timezone.utc)

    @property
    def pnl(self) -> Decimal:
        if self.initial_balance is None:
            return Decimal(0)
        return self.balance_total - self.initial_balance

    @property
    def pnl_pct(self) -> float:
        if not self.initial_balance or self.initial_balance == 0:
            return 0.0
        return float(self.pnl / self.initial_balance) * 100

    @property
    def uptime(self) -> str:
        delta = datetime.now(timezone.utc) - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


state = DashState()
rest_client: KalshiRestClient | None = None
data_dir: Path = Path("data")


# ---------------------------------------------------------------------------
# Data refresh
# ---------------------------------------------------------------------------


async def refresh_loop():
    global rest_client
    while True:
        if rest_client is None:
            await asyncio.sleep(5)
            continue
        try:
            bal = await rest_client.get_balance()
            state.balance_available = bal.available
            state.balance_total = bal.total
            if state.initial_balance is None:
                state.initial_balance = bal.total

            positions = await rest_client.get_positions()
            state.positions = [
                {
                    "ticker": p.ticker,
                    "event": p.event_ticker,
                    "side": "YES" if p.yes_count > 0 else "NO",
                    "qty": p.yes_count if p.yes_count > 0 else p.no_count,
                    "pnl": float(p.realized_pnl),
                    "exposure": float(p.market_exposure),
                }
                for p in positions
                if p.position != 0
            ]

            fills = await rest_client.get_fills(limit=25)
            state.recent_fills = [
                {
                    "ticker": f.ticker,
                    "side": f.side.value.upper(),
                    "action": f.action.value.upper(),
                    "qty": f.count,
                    "price": float(f.price),
                    "fee": float(f.fee_cost),
                    "taker": f.is_taker,
                    "time": f.created_time.strftime("%H:%M:%S") if f.created_time.year > 2000 else "",
                }
                for f in fills
            ]

            orders = await rest_client.get_orders(status="resting")
            state.open_orders = [
                {
                    "ticker": o.ticker,
                    "side": o.side.value.upper(),
                    "action": o.action.value.upper(),
                    "qty": o.remaining_count,
                    "price": float(o.price),
                    "time": o.created_time.strftime("%H:%M:%S") if o.created_time.year > 2000 else "",
                }
                for o in orders
            ]

            settlements = await rest_client.get_settlements(limit=15)
            state.settlements = [
                {
                    "ticker": s.ticker,
                    "result": s.market_result.value.upper(),
                    "revenue_cents": int(s.revenue),
                    "fee": float(s.fee_cost),
                    "time": s.settled_time.strftime("%H:%M:%S") if s.settled_time.year > 2000 else "",
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
                        "categories": cats,
                    }
                except Exception:
                    pass

            # Parse log
            _parse_log(state)

            state.last_refresh = datetime.now(timezone.utc)
            state.error = None
        except Exception as exc:
            state.error = str(exc)[:120]

        await asyncio.sleep(15)


def _parse_log(st: DashState) -> None:
    # Try multiple log locations
    log_candidates = [
        Path("/tmp/supervisor.log"),
    ]
    # Also check MoneyGone logs dir for execution logs
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    if logs_dir.exists():
        exec_logs = sorted(logs_dir.glob("execution-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if exec_logs:
            log_candidates.insert(0, exec_logs[0])  # Most recent execution log

    log_path = None
    for candidate in log_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            log_path = candidate
            break
    if log_path is None:
        return

    try:
        size = log_path.stat().st_size
        with open(log_path) as f:
            f.seek(max(0, size - 120_000))
            if size > 120_000:
                f.readline()
            lines = f.readlines()
    except Exception:
        return

    activity = []
    keywords = (
        "order_placed", "fill", "sniper", "live_edge", "outcome_detected",
        "kill_switch", "TRADE", "edge=", "SIGNAL", "engine.submitted",
        "engine.order_filled", "settlement",
    )
    for line in lines:
        if any(kw in line for kw in keywords):
            # Try to parse JSON log line for cleaner display
            try:
                obj = json.loads(line)
                ts = obj.get("timestamp", "")[:19]
                event = obj.get("event", "")
                ticker = obj.get("ticker", "")
                msg = f"{event} {ticker}".strip()
                if "outcome" in event:
                    msg += f" → {obj.get('outcome', '')}"
                elif "fill" in event:
                    msg += f" {obj.get('side', '')} {obj.get('count', '')}@{obj.get('price', '')}"
                elif "kill_switch" in event:
                    msg += f" streak={obj.get('streak', '')} {obj.get('msg', '')}"
                activity.append({"time": ts, "msg": msg[:140]})
            except (json.JSONDecodeError, KeyError):
                ts = line[:19] if len(line) > 19 else ""
                activity.append({"time": ts, "msg": line.strip()[:140]})
    st.activity = activity[-20:]


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


def _json_state() -> dict:
    return {
        "balance": float(state.balance_available),
        "total": float(state.balance_total),
        "pnl": float(state.pnl),
        "pnl_pct": round(state.pnl_pct, 2),
        "uptime": state.uptime,
        "positions": state.positions,
        "fills": state.recent_fills,
        "orders": state.open_orders,
        "settlements": state.settlements,
        "discovery": state.discovery_stats,
        "activity": state.activity,
        "error": state.error,
        "last_refresh": state.last_refresh.isoformat() if state.last_refresh else None,
    }


async def handle_api(request: web.Request) -> web.Response:
    return web.json_response(_json_state())


HEALTH_FILE = Path("/tmp/moneygone_health.json")


async def handle_health(request: web.Request) -> web.Response:
    """Return execution engine health status from shared JSON file."""
    if not HEALTH_FILE.exists():
        return web.json_response(
            {"error": "health file not found", "engine_status": "unknown"},
            status=503,
        )
    try:
        data = json.loads(HEALTH_FILE.read_text())
        return web.json_response(data)
    except (json.JSONDecodeError, OSError) as exc:
        return web.json_response(
            {"error": str(exc)[:120], "engine_status": "unknown"},
            status=503,
        )


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_PAGE, content_type="text/html")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MONEYGONE // LIVE</title>
<style>
:root {
  --bg: #0a0a0a;
  --panel: #0d1117;
  --border: #0f0;
  --green: #00ff41;
  --cyan: #00e5ff;
  --red: #ff1744;
  --yellow: #ffd600;
  --dim: #2a5a2a;
  --text: #c0ffc0;
  --white: #e0ffe0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'SF Mono', monospace;
  font-size: 13px;
  line-height: 1.4;
  min-height: 100vh;
}
.header {
  text-align: center;
  padding: 12px 0 6px;
  border-bottom: 1px solid var(--dim);
}
.header h1 {
  color: var(--green);
  font-size: 20px;
  letter-spacing: 6px;
  text-shadow: 0 0 10px rgba(0,255,65,0.3);
}
.header .sub {
  color: var(--dim);
  font-size: 11px;
  margin-top: 2px;
}
.stats-bar {
  display: flex;
  justify-content: center;
  gap: 30px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--dim);
  flex-wrap: wrap;
}
.stat { text-align: center; }
.stat .label { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 2px; }
.stat .value { color: var(--green); font-size: 18px; font-weight: bold; }
.stat .value.loss { color: var(--red); }
.stat .value.cyan { color: var(--cyan); }
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  padding: 8px;
  max-width: 1400px;
  margin: 0 auto;
}
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.panel {
  background: var(--panel);
  border: 1px solid var(--dim);
  border-radius: 4px;
  overflow: hidden;
}
.panel-title {
  background: rgba(0,255,65,0.08);
  color: var(--cyan);
  font-size: 11px;
  font-weight: bold;
  letter-spacing: 2px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--dim);
}
.panel-body { padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
}
th {
  color: var(--dim);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 4px 8px;
  text-align: left;
  border-bottom: 1px solid rgba(0,255,65,0.1);
}
td {
  padding: 3px 8px;
  font-size: 12px;
  border-bottom: 1px solid rgba(0,255,65,0.05);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
}
tr:hover { background: rgba(0,255,65,0.03); }
.yes { color: var(--green); }
.no, .sell { color: var(--red); }
.buy { color: var(--green); }
.profit { color: var(--green); }
.loss { color: var(--red); }
.ticker { color: var(--cyan); }
.dim { color: var(--dim); }
.activity-line {
  padding: 2px 10px;
  font-size: 11px;
  border-bottom: 1px solid rgba(0,255,65,0.03);
  color: var(--text);
}
.activity-line .ts { color: var(--dim); margin-right: 8px; }
.status-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  margin-right: 6px;
  animation: pulse 2s infinite;
}
.status-dot.err { background: var(--red); animation: none; }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.footer {
  text-align: center;
  padding: 8px;
  color: var(--dim);
  font-size: 10px;
  border-top: 1px solid var(--dim);
}
.full-width { grid-column: 1 / -1; }
.cats { display: flex; gap: 12px; padding: 8px 10px; flex-wrap: wrap; }
.cat-badge {
  background: rgba(0,255,65,0.1);
  border: 1px solid var(--dim);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 11px;
  color: var(--cyan);
}
.cat-badge span { color: var(--green); font-weight: bold; }
.scanline {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  animation: scan 4s linear infinite;
  pointer-events: none;
  opacity: 0.3;
  z-index: 999;
}
@keyframes scan { 0% { transform: translateY(0); } 100% { transform: translateY(100vh); } }
</style>
</head>
<body>
<div class="scanline"></div>

<div class="header">
  <h1>M O N E Y G O N E</h1>
  <div class="sub">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">CONNECTING...</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;UPTIME <span id="uptime">--:--:--</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;REFRESH <span id="refreshTime">--</span>
  </div>
</div>

<div class="stats-bar">
  <div class="stat">
    <div class="label">Balance</div>
    <div class="value cyan" id="balance">--</div>
  </div>
  <div class="stat">
    <div class="label">Total</div>
    <div class="value cyan" id="total">--</div>
  </div>
  <div class="stat">
    <div class="label">Session P&L</div>
    <div class="value" id="pnl">--</div>
  </div>
  <div class="stat">
    <div class="label">P&L %</div>
    <div class="value" id="pnlPct">--</div>
  </div>
  <div class="stat">
    <div class="label">Positions</div>
    <div class="value cyan" id="posCount">0</div>
  </div>
  <div class="stat">
    <div class="label">Orders</div>
    <div class="value cyan" id="orderCount">0</div>
  </div>
</div>

<div class="grid">
  <!-- Positions -->
  <div class="panel">
    <div class="panel-title">OPEN POSITIONS</div>
    <div class="panel-body"><table>
      <thead><tr><th>Market</th><th>Side</th><th>Qty</th><th>Exposure</th><th>P&L</th></tr></thead>
      <tbody id="positionsBody"></tbody>
    </table></div>
  </div>

  <!-- Open Orders -->
  <div class="panel">
    <div class="panel-title">RESTING ORDERS</div>
    <div class="panel-body"><table>
      <thead><tr><th>Market</th><th>Side</th><th>Act</th><th>Qty</th><th>Price</th></tr></thead>
      <tbody id="ordersBody"></tbody>
    </table></div>
  </div>

  <!-- Fills -->
  <div class="panel">
    <div class="panel-title">RECENT FILLS</div>
    <div class="panel-body"><table>
      <thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Act</th><th>Qty</th><th>Price</th><th>Fee</th></tr></thead>
      <tbody id="fillsBody"></tbody>
    </table></div>
  </div>

  <!-- Settlements -->
  <div class="panel">
    <div class="panel-title">SETTLEMENTS</div>
    <div class="panel-body"><table>
      <thead><tr><th>Time</th><th>Market</th><th>Result</th><th>Revenue</th></tr></thead>
      <tbody id="settlementsBody"></tbody>
    </table></div>
  </div>

  <!-- Discovery -->
  <div class="panel">
    <div class="panel-title">MARKET DISCOVERY</div>
    <div class="panel-body">
      <div style="padding:8px 10px;font-size:14px;">
        Tracked markets: <span class="yes" id="discoveryTotal">--</span>
      </div>
      <div class="cats" id="discoveryCats"></div>
    </div>
  </div>

  <!-- Activity Log -->
  <div class="panel">
    <div class="panel-title">ACTIVITY LOG</div>
    <div class="panel-body" id="activityBody" style="max-height:240px;overflow-y:auto;"></div>
  </div>
</div>

<div class="footer">
  MONEYGONE AUTOMATED TRADING SYSTEM &mdash; KALSHI PREDICTION MARKETS
</div>

<script>
const $ = id => document.getElementById(id);
const money = v => v >= 1000 ? '$' + v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) : '$' + v.toFixed(2);

function pnlClass(v) { return v > 0 ? 'profit' : v < 0 ? 'loss' : ''; }
function sideClass(s) { return s === 'YES' ? 'yes' : 'no'; }

async function refresh() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) { throw new Error('HTTP ' + r.status); }
    const d = await r.json();

    $('balance').textContent = money(d.balance);
    $('total').textContent = money(d.total);

    const pnlEl = $('pnl');
    pnlEl.textContent = (d.pnl >= 0 ? '+' : '') + money(d.pnl);
    pnlEl.className = 'value ' + pnlClass(d.pnl);

    const pctEl = $('pnlPct');
    pctEl.textContent = (d.pnl_pct >= 0 ? '+' : '') + d.pnl_pct.toFixed(2) + '%';
    pctEl.className = 'value ' + pnlClass(d.pnl_pct);

    $('uptime').textContent = d.uptime;
    $('posCount').textContent = d.positions.length;
    $('orderCount').textContent = d.orders.length;

    const dot = $('statusDot');
    const stxt = $('statusText');
    if (d.error) {
      dot.className = 'status-dot err';
      stxt.textContent = 'ERROR: ' + d.error;
    } else {
      dot.className = 'status-dot';
      stxt.textContent = 'LIVE';
    }

    if (d.last_refresh) {
      const ago = Math.round((Date.now() - new Date(d.last_refresh).getTime()) / 1000);
      $('refreshTime').textContent = ago + 's ago';
    }

    // Positions
    $('positionsBody').innerHTML = d.positions.length ? d.positions.map(p =>
      `<tr>
        <td class="ticker">${p.ticker}</td>
        <td class="${sideClass(p.side)}">${p.side}</td>
        <td>${p.qty}</td>
        <td>${money(p.exposure)}</td>
        <td class="${pnlClass(p.pnl)}">${money(p.pnl)}</td>
      </tr>`
    ).join('') : '<tr><td colspan="5" class="dim">No open positions</td></tr>';

    // Orders
    $('ordersBody').innerHTML = d.orders.length ? d.orders.map(o =>
      `<tr>
        <td class="ticker">${o.ticker}</td>
        <td class="${sideClass(o.side)}">${o.side}</td>
        <td class="${o.action === 'BUY' ? 'buy' : 'sell'}">${o.action}</td>
        <td>${o.qty}</td>
        <td>${o.price.toFixed(2)}</td>
      </tr>`
    ).join('') : '<tr><td colspan="5" class="dim">No resting orders</td></tr>';

    // Fills
    $('fillsBody').innerHTML = d.fills.length ? d.fills.slice(0, 15).map(f =>
      `<tr>
        <td class="dim">${f.time}</td>
        <td class="ticker">${f.ticker}</td>
        <td class="${sideClass(f.side)}">${f.side}</td>
        <td class="${f.action === 'BUY' ? 'buy' : 'sell'}">${f.action}</td>
        <td>${f.qty}</td>
        <td>${f.price.toFixed(2)}</td>
        <td class="dim">${f.fee.toFixed(3)}</td>
      </tr>`
    ).join('') : '<tr><td colspan="7" class="dim">No fills</td></tr>';

    // Settlements
    $('settlementsBody').innerHTML = d.settlements.length ? d.settlements.slice(0, 10).map(s =>
      `<tr>
        <td class="dim">${s.time}</td>
        <td class="ticker">${s.ticker}</td>
        <td class="${s.result.includes('YES') ? 'yes' : 'no'}">${s.result}</td>
        <td>${s.revenue_cents}c</td>
      </tr>`
    ).join('') : '<tr><td colspan="4" class="dim">No settlements</td></tr>';

    // Discovery
    $('discoveryTotal').textContent = d.discovery.total || '--';
    const cats = d.discovery.categories || {};
    $('discoveryCats').innerHTML = Object.entries(cats)
      .sort((a,b) => b[1] - a[1])
      .map(([k,v]) => `<div class="cat-badge">${k} <span>${v}</span></div>`)
      .join('');

    // Activity
    $('activityBody').innerHTML = d.activity.length ? d.activity.map(a =>
      `<div class="activity-line"><span class="ts">${a.time}</span>${a.msg}</div>`
    ).join('') : '<div class="activity-line dim">No recent activity</div>';

  } catch(e) {
    $('statusDot').className = 'status-dot err';
    $('statusText').textContent = 'FETCH ERROR';
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


async def on_startup(app: web.Application) -> None:
    global rest_client
    cfg = app["cfg"]
    rest_client = KalshiRestClient(cfg.exchange)
    await rest_client._ensure_client()
    app["refresh_task"] = asyncio.create_task(refresh_loop())


async def on_cleanup(app: web.Application) -> None:
    task = app.get("refresh_task")
    if task:
        task.cancel()
    if rest_client:
        await rest_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MoneyGone Web Dashboard")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    overlay = Path(args.overlay) if args.overlay else None
    cfg = load_config(Path(args.config), overlay)

    global data_dir
    data_dir = Path(cfg.data_dir)

    app = web.Application()
    app["cfg"] = cfg
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_api)
    app.router.add_get("/api/health", handle_health)

    print(f"Starting MoneyGone dashboard on http://{args.host}:{args.port}")
    print(f"Auth: {AUTH_USER} / {'*' * len(AUTH_PASS)}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
