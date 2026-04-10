#!/usr/bin/env python3
"""Analyze dry_run.would_trade log entries from stress test."""
import sys
import json

trades = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        trades.append(d)
    except json.JSONDecodeError:
        continue

# Group by category
weather = [t for t in trades if any(p in t['ticker'] for p in ['KXLOWT', 'KXHIGHT', 'KXRAIN', 'KXSNOW', 'KXWIND', 'KXPRECIP'])]
sports = [t for t in trades if t not in weather]

print(f"=== TOTAL WOULD_TRADE: {len(trades)} ===")
print(f"  Weather: {len(weather)}")
print(f"  Sports: {len(sports)}")
print()

print("=== WEATHER TRADES ===")
for t in weather:
    side_dir = "BUY YES" if t['side'] == 'yes' else "BUY NO"
    tp = float(t['target_price'])
    cost = tp if t['side'] == 'yes' else 1.0 - tp
    ticker = t['ticker']
    edge = t['edge']
    model = t['model_prob']
    implied = t['implied_prob']
    qty = t['contracts']
    verdict = ""
    # Flag suspicious trades
    if model > 0.95 and edge > 0.50:
        verdict = " *** EXTREME - model 99% certain"
    elif edge > 0.30:
        verdict = " ** HIGH EDGE"
    elif edge < 0.02:
        verdict = " * MARGINAL"
    print(f"  {ticker:40s} {side_dir:8s} @${cost:.2f}  edge={edge:.2f}  model={model:.2f}  mkt={implied:.2f}  qty={qty}{verdict}")

print()
print("=== SPORTS TRADES ===")
for t in sports:
    side_dir = "BUY YES" if t['side'] == 'yes' else "BUY NO"
    tp = float(t['target_price'])
    cost = tp if t['side'] == 'yes' else 1.0 - tp
    ticker = t['ticker']
    edge = t['edge']
    model = t['model_prob']
    implied = t['implied_prob']
    qty = t['contracts']
    verdict = ""
    if edge > 0.15:
        verdict = " ** HIGH EDGE"
    elif edge < 0.02:
        verdict = " * MARGINAL"
    print(f"  {ticker:50s} {side_dir:8s} @${cost:.2f}  edge={edge:.2f}  model={model:.2f}  mkt={implied:.2f}  qty={qty}{verdict}")

# Summary stats
print()
print("=== EDGE DISTRIBUTION ===")
edges = [t['edge'] for t in trades]
weather_edges = [t['edge'] for t in weather]
sports_edges = [t['edge'] for t in sports]
if weather_edges:
    print(f"  Weather: min={min(weather_edges):.2f}  max={max(weather_edges):.2f}  avg={sum(weather_edges)/len(weather_edges):.2f}")
if sports_edges:
    print(f"  Sports:  min={min(sports_edges):.2f}  max={max(sports_edges):.2f}  avg={sum(sports_edges)/len(sports_edges):.2f}")

# Weather: check for model always 0.99 or 0.01
print()
print("=== MODEL PROBABILITY DISTRIBUTION ===")
weather_probs = [t['model_prob'] for t in weather]
if weather_probs:
    at_99 = sum(1 for p in weather_probs if p >= 0.99)
    at_01 = sum(1 for p in weather_probs if p <= 0.01)
    between = sum(1 for p in weather_probs if 0.01 < p < 0.99)
    print(f"  Weather model_prob: {at_99} at 0.99, {at_01} at 0.01, {between} in between")
    print(f"  THIS IS A CALIBRATION PROBLEM if everything is at 0.99/0.01!")
