#!/usr/bin/env python3
"""Stress test analyzer for engine.candidate log entries.

Parses structured JSON logs emitted by the execution engine and computes
the metrics from the 10-point stress testing framework:

  1. Adversarial market test — flag extreme/suspicious edges
  2. Counterfactual ranking — rank all candidates by edge quality
  3. Best-opportunity test — compare selected vs rejected
  4. Human audit — formatted output for manual review
  5. No-trade discipline — percentage of cycles with zero selections
  6. Calibration-by-confidence — bucket by model confidence
  7. Edge distribution — sanity check on edge magnitudes

Usage:
    # From JSON logs:
    cat /var/log/moneygone/engine.log | python scripts/analyze_trades.py

    # From journald:
    journalctl -u moneygone --since today -o cat | python scripts/analyze_trades.py

    # With jq preprocessing (if logs are wrapped):
    cat log.jsonl | jq -c 'select(.event)' | python scripts/analyze_trades.py
"""
import sys
import json
from collections import defaultdict
from datetime import datetime


def parse_logs(stream):
    """Parse engine.candidate and engine.cycle_summary entries from JSON lines."""
    candidates = []
    cycles = []
    dry_runs = []

    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = d.get("event", "")
        if event == "engine.candidate":
            candidates.append(d)
        elif event == "engine.cycle_summary":
            cycles.append(d)
        elif event == "dry_run.would_trade":
            dry_runs.append(d)

    return candidates, cycles, dry_runs


def analyze_candidates(candidates):
    """Analyze all candidate evaluations."""
    if not candidates:
        print("No engine.candidate entries found.")
        print("(Looking for dry_run.would_trade entries instead...)")
        return

    selected = [c for c in candidates if c.get("status") == "selected"]
    rejected = [c for c in candidates if c.get("status") == "rejected"]

    # Group by rejection reason
    reject_reasons = defaultdict(list)
    for c in rejected:
        reject_reasons[c.get("reject_reason", "unknown")].append(c)

    # Group by category
    by_category = defaultdict(lambda: {"selected": [], "rejected": []})
    for c in candidates:
        cat = c.get("category", "unknown")
        by_category[cat][c["status"]].append(c)

    # ── Overall Summary ──
    print("=" * 70)
    print(f"  STRESS TEST REPORT — {len(candidates)} candidates evaluated")
    print(f"  Selected: {len(selected)}  |  Rejected: {len(rejected)}")
    print(f"  Selection rate: {len(selected)/len(candidates)*100:.1f}%")
    print("=" * 70)
    print()

    # ── Category Breakdown ──
    print("── Category Breakdown ──")
    for cat, data in sorted(by_category.items()):
        total = len(data["selected"]) + len(data["rejected"])
        sel = len(data["selected"])
        print(f"  {cat:12s}: {total:4d} total, {sel:3d} selected ({sel/total*100:.0f}%)")
    print()

    # ── Rejection Funnel ──
    print("── Rejection Funnel ──")
    for reason, entries in sorted(reject_reasons.items(), key=lambda x: -len(x[1])):
        print(f"  {reason:35s}: {len(entries):4d}")
    print()

    # ── Test #1: Adversarial Market Test — Flag Extreme Edges ──
    print("── #1 Adversarial Market Test ──")
    extreme = [c for c in selected if c.get("fee_adjusted_edge", 0) > 0.30]
    suspicious = [c for c in selected if c.get("model_prob", 0.5) > 0.96 or c.get("model_prob", 0.5) < 0.04]
    if extreme:
        print(f"  ⚠  {len(extreme)} trades with edge > 30% — likely mispricing:")
        for c in extreme[:10]:
            print(f"     {c['ticker']:35s} edge={c.get('fee_adjusted_edge',0):.2f}  model={c.get('model_prob',0):.2f}  mkt={c.get('market_prob',0):.2f}")
    else:
        print("  ✓  No extreme edges found (good)")
    if suspicious:
        print(f"  ⚠  {len(suspicious)} trades with near-certain model probability:")
        for c in suspicious[:10]:
            print(f"     {c['ticker']:35s} model_prob={c.get('model_prob',0):.3f}  confidence={c.get('confidence',0):.2f}")
    else:
        print("  ✓  No near-certain probabilities (good)")
    print()

    # ── Test #2: Counterfactual Ranking ──
    print("── #2 Counterfactual Ranking ──")
    # Rank ALL candidates that got through to edge computation (have fee_adjusted_edge)
    ranked = [c for c in candidates if c.get("fee_adjusted_edge") is not None]
    ranked.sort(key=lambda c: c.get("fee_adjusted_edge", 0), reverse=True)

    if ranked:
        print(f"  Top 10 by fee-adjusted edge (of {len(ranked)} scoreable):")
        for i, c in enumerate(ranked[:10], 1):
            status_mark = "✓" if c["status"] == "selected" else "✗"
            reason = c.get("reject_reason", "")
            print(
                f"  {i:2d}. {status_mark} {c['ticker']:35s} "
                f"edge={c.get('fee_adjusted_edge',0):+.4f}  "
                f"model={c.get('model_prob',0):.3f}  "
                f"mkt={c.get('market_prob',0):.3f}  "
                f"conf={c.get('confidence',0):.2f}  "
                f"{'['+reason+']' if reason else ''}"
            )

        # Check: are selected trades actually the highest-ranked?
        selected_ranks = []
        for i, c in enumerate(ranked):
            if c["status"] == "selected":
                selected_ranks.append(i + 1)
        if selected_ranks:
            avg_rank = sum(selected_ranks) / len(selected_ranks)
            print(f"\n  Selected trades: avg rank = {avg_rank:.0f}/{len(ranked)}  "
                  f"(lower=better, worst={max(selected_ranks)})")
            if avg_rank <= 5:
                print("  ✓  Engine is picking from the top — good signal")
            elif avg_rank <= 20:
                print("  ~  Engine is picking mid-tier — may be leaving edge on table")
            else:
                print("  ✗  Engine is NOT picking the best opportunities")
    print()

    # ── Test #3: Best Opportunity Test ──
    print("── #3 Best-Opportunity Test ──")
    selected_with_edge = [c for c in selected if c.get("fee_adjusted_edge") is not None]
    rejected_with_edge = [c for c in rejected if c.get("fee_adjusted_edge") is not None and c.get("fee_adjusted_edge", 0) > 0]
    if selected_with_edge and rejected_with_edge:
        avg_sel_edge = sum(c["fee_adjusted_edge"] for c in selected_with_edge) / len(selected_with_edge)
        avg_rej_edge = sum(c["fee_adjusted_edge"] for c in rejected_with_edge) / len(rejected_with_edge)
        print(f"  Selected avg edge: {avg_sel_edge:+.4f}  ({len(selected_with_edge)} trades)")
        print(f"  Rejected (positive edge) avg: {avg_rej_edge:+.4f}  ({len(rejected_with_edge)} trades)")
        if avg_sel_edge > avg_rej_edge:
            print("  ✓  Selected trades have higher edge than rejected positive-edge trades")
        else:
            print("  ✗  Rejected trades had HIGHER edge — check filtering logic")

        # How many rejected trades had edge > best selected?
        if selected_with_edge:
            best_selected = max(c["fee_adjusted_edge"] for c in selected_with_edge)
            missed = [c for c in rejected_with_edge if c["fee_adjusted_edge"] > best_selected]
            if missed:
                print(f"  ⚠  {len(missed)} rejected trades had edge ABOVE best selected ({best_selected:.4f}):")
                for c in missed[:5]:
                    print(f"     {c['ticker']:35s} edge={c['fee_adjusted_edge']:.4f} reason={c.get('reject_reason','')}")
    print()

    # ── Test #4: Human Audit — Selected Trades ──
    print("── #4 Human Audit — Selected Trades ──")
    if selected:
        for c in selected:
            side_dir = f"BUY {c.get('side','?').upper()}"
            tp = c.get("target_price", "?")
            verdict = ""
            edge = c.get("fee_adjusted_edge", 0)
            model = c.get("model_prob", 0.5)
            if model > 0.95 and edge > 0.40:
                verdict = " *** EXTREME — probably mispricing"
            elif edge > 0.20:
                verdict = " ** HIGH EDGE — verify"
            elif edge < 0.015:
                verdict = " * MARGINAL"

            print(f"  {c['ticker']:35s} {side_dir:8s} @{tp:>6s}  "
                  f"edge={edge:+.4f}  model={model:.3f}  mkt={c.get('market_prob',0):.3f}  "
                  f"conf={c.get('confidence',0):.2f}  qty={c.get('contracts','?')}  "
                  f"liq={c.get('liquidity','?')}{verdict}")
    else:
        print("  No trades selected this period")
    print()

    # ── Test #5: No-Trade Discipline ──
    # (Computed from cycle summaries separately)

    # ── Test #6: Calibration by Confidence ──
    print("── #6 Calibration by Confidence Bucket ──")
    conf_buckets = defaultdict(lambda: {"count": 0, "edges": [], "probs": []})
    for c in candidates:
        conf = c.get("confidence")
        if conf is None:
            continue
        if conf < 0.4:
            bucket = "low (0.30-0.40)"
        elif conf < 0.55:
            bucket = "med (0.40-0.55)"
        elif conf < 0.70:
            bucket = "high (0.55-0.70)"
        else:
            bucket = "very high (0.70+)"
        conf_buckets[bucket]["count"] += 1
        if c.get("fee_adjusted_edge") is not None:
            conf_buckets[bucket]["edges"].append(c["fee_adjusted_edge"])
        if c.get("model_prob") is not None:
            conf_buckets[bucket]["probs"].append(c["model_prob"])

    for bucket in sorted(conf_buckets.keys()):
        data = conf_buckets[bucket]
        avg_edge = sum(data["edges"]) / len(data["edges"]) if data["edges"] else 0
        print(f"  {bucket:20s}: {data['count']:4d} candidates, avg_edge={avg_edge:+.4f}")
    print()

    # ── Test #7: Edge Distribution ──
    print("── #7 Edge Distribution ──")
    all_edges = [c.get("fee_adjusted_edge", 0) for c in candidates if c.get("fee_adjusted_edge") is not None]
    if all_edges:
        negative = sum(1 for e in all_edges if e < 0)
        zero_to_5 = sum(1 for e in all_edges if 0 <= e < 0.05)
        five_to_10 = sum(1 for e in all_edges if 0.05 <= e < 0.10)
        ten_to_20 = sum(1 for e in all_edges if 0.10 <= e < 0.20)
        twenty_plus = sum(1 for e in all_edges if e >= 0.20)
        print(f"  Negative:  {negative:4d}")
        print(f"  0-5%:      {zero_to_5:4d}")
        print(f"  5-10%:     {five_to_10:4d}")
        print(f"  10-20%:    {ten_to_20:4d}")
        print(f"  20%+:      {twenty_plus:4d}  {'⚠ verify these!' if twenty_plus > 5 else ''}")
    print()

    # ── Model Probability Distribution ──
    print("── Model Probability Distribution ──")
    all_probs = [c.get("model_prob", 0.5) for c in candidates if c.get("model_prob") is not None]
    if all_probs:
        at_extreme = sum(1 for p in all_probs if p >= 0.96 or p <= 0.04)
        in_middle = sum(1 for p in all_probs if 0.30 <= p <= 0.70)
        print(f"  Extreme (<0.04 or >0.96): {at_extreme}")
        print(f"  Middle (0.30-0.70):       {in_middle}")
        print(f"  Total with prob:          {len(all_probs)}")
        if at_extreme > len(all_probs) * 0.5:
            print("  ⚠ Over half at extremes — calibration issue!")
    print()


def analyze_cycles(cycles):
    """Analyze per-cycle summaries for no-trade discipline."""
    if not cycles:
        return

    print("── #5 No-Trade Discipline (Cycle Summary) ──")
    zero_selection_cycles = sum(1 for c in cycles if c.get("selected", 0) == 0)
    total_cycles = len(cycles)
    avg_evaluated = sum(c.get("evaluated", 0) for c in cycles) / total_cycles
    avg_selected = sum(c.get("selected", 0) for c in cycles) / total_cycles
    avg_elapsed = sum(c.get("elapsed_s", 0) for c in cycles) / total_cycles

    print(f"  Total cycles: {total_cycles}")
    print(f"  Zero-selection cycles: {zero_selection_cycles} ({zero_selection_cycles/total_cycles*100:.0f}%)")
    print(f"  Avg evaluated/cycle: {avg_evaluated:.0f}")
    print(f"  Avg selected/cycle: {avg_selected:.1f}")
    print(f"  Avg cycle time: {avg_elapsed:.1f}s")

    if zero_selection_cycles / total_cycles > 0.50:
        print("  ✓  System shows no-trade discipline (>50% empty cycles)")
    elif zero_selection_cycles / total_cycles > 0.20:
        print("  ~  Moderate selectivity — check if too aggressive")
    else:
        print("  ⚠  System trades nearly every cycle — likely over-trading")
    print()


def analyze_dry_runs(dry_runs):
    """Fallback analysis for old-format dry_run.would_trade entries."""
    if not dry_runs:
        return

    print("=" * 70)
    print(f"  DRY RUN TRADE LOG — {len(dry_runs)} would_trade entries")
    print("=" * 70)

    weather_tickers = {'KXLOWT', 'KXHIGHT', 'KXRAIN', 'KXSNOW', 'KXWIND', 'KXPRECIP', 'KXTEMP'}
    weather = [t for t in dry_runs if any(p in t.get('ticker', '') for p in weather_tickers)]
    sports = [t for t in dry_runs if t not in weather]

    print(f"  Weather: {len(weather)}")
    print(f"  Sports:  {len(sports)}")
    print()

    for t in weather:
        side_dir = f"BUY {t.get('side','?').upper()}"
        tp = float(t.get('target_price', 0))
        edge = t.get('edge', 0)
        model = t.get('model_prob', 0)
        implied = t.get('implied_prob', 0)
        qty = t.get('contracts', 0)
        verdict = ""
        if model > 0.95 and edge > 0.40:
            verdict = " *** EXTREME"
        elif edge > 0.20:
            verdict = " ** HIGH EDGE"
        elif edge < 0.015:
            verdict = " * MARGINAL"
        print(f"  {t['ticker']:35s} {side_dir:8s} @${tp:.2f}  "
              f"edge={edge:.4f}  model={model:.3f}  mkt={implied:.3f}  qty={qty}{verdict}")
    print()


def compute_trade_iq(candidates, cycles):
    """Compute the Trade IQ Score from the stress test framework.

    Components:
    - Ranking quality (25%): are selected trades among the top-ranked?
    - Selectivity (20%): appropriate no-trade discipline
    - Edge sanity (20%): no extreme/implausible edges
    - Calibration spread (15%): model probabilities use full range
    - Confidence-edge coherence (10%): higher confidence → better edges
    - Rejection diversity (10%): using multiple rejection reasons
    """
    if not candidates:
        return None

    score = 0.0
    details = {}

    # 1. Ranking quality (25 pts)
    ranked = [c for c in candidates if c.get("fee_adjusted_edge") is not None]
    ranked.sort(key=lambda c: c.get("fee_adjusted_edge", 0), reverse=True)
    selected = [c for c in candidates if c["status"] == "selected"]

    if ranked and selected:
        selected_ranks = []
        for i, c in enumerate(ranked):
            if c["status"] == "selected":
                selected_ranks.append(i + 1)
        if selected_ranks:
            avg_rank_pct = sum(selected_ranks) / len(selected_ranks) / max(len(ranked), 1)
            # Perfectly top-ranked = 25, random = 12.5, bottom = 0
            ranking_score = max(0, 25 * (1 - 2 * avg_rank_pct))
            score += ranking_score
            details["ranking"] = round(ranking_score, 1)

    # 2. Selectivity (20 pts)
    if cycles:
        zero_rate = sum(1 for c in cycles if c.get("selected", 0) == 0) / len(cycles)
        # Ideal: 50-80% empty cycles
        if 0.50 <= zero_rate <= 0.80:
            sel_score = 20
        elif 0.30 <= zero_rate <= 0.90:
            sel_score = 15
        elif zero_rate > 0.90:
            sel_score = 10  # Too conservative
        else:
            sel_score = 5  # Too aggressive
        score += sel_score
        details["selectivity"] = sel_score

    # 3. Edge sanity (20 pts)
    extreme_count = sum(1 for c in selected if c.get("fee_adjusted_edge", 0) > 0.30)
    if selected:
        extreme_pct = extreme_count / len(selected)
        if extreme_pct == 0:
            sanity_score = 20
        elif extreme_pct < 0.10:
            sanity_score = 15
        elif extreme_pct < 0.25:
            sanity_score = 10
        else:
            sanity_score = 0
        score += sanity_score
        details["edge_sanity"] = sanity_score

    # 4. Calibration spread (15 pts)
    probs = [c.get("model_prob") for c in candidates if c.get("model_prob") is not None]
    if probs:
        extreme_prob_pct = sum(1 for p in probs if p > 0.96 or p < 0.04) / len(probs)
        if extreme_prob_pct < 0.20:
            cal_score = 15
        elif extreme_prob_pct < 0.40:
            cal_score = 10
        elif extreme_prob_pct < 0.60:
            cal_score = 5
        else:
            cal_score = 0
        score += cal_score
        details["calibration"] = cal_score

    # 5. Confidence-edge coherence (10 pts)
    high_conf = [c for c in candidates if (c.get("confidence") or 0) >= 0.55 and c.get("fee_adjusted_edge") is not None]
    low_conf = [c for c in candidates if 0.30 <= (c.get("confidence") or 0) < 0.45 and c.get("fee_adjusted_edge") is not None]
    if high_conf and low_conf:
        avg_high = sum(c["fee_adjusted_edge"] for c in high_conf) / len(high_conf)
        avg_low = sum(c["fee_adjusted_edge"] for c in low_conf) / len(low_conf)
        if avg_high > avg_low:
            coh_score = 10
        else:
            coh_score = 3  # Higher confidence ≠ better edges = problem
        score += coh_score
        details["coherence"] = coh_score

    # 6. Rejection diversity (10 pts)
    reject_reasons = set(c.get("reject_reason", "") for c in candidates if c["status"] == "rejected")
    n_reasons = len(reject_reasons)
    if n_reasons >= 5:
        div_score = 10
    elif n_reasons >= 3:
        div_score = 7
    elif n_reasons >= 1:
        div_score = 3
    else:
        div_score = 0
    score += div_score
    details["rejection_diversity"] = div_score

    return round(score, 1), details


def main():
    candidates, cycles, dry_runs = parse_logs(sys.stdin)

    if candidates:
        analyze_candidates(candidates)
        analyze_cycles(cycles)

        iq_result = compute_trade_iq(candidates, cycles)
        if iq_result is not None:
            score, details = iq_result
            print("=" * 70)
            print(f"  TRADE IQ SCORE: {score}/100")
            print("=" * 70)
            for component, pts in sorted(details.items()):
                print(f"    {component:25s}: {pts}")
            print()
            if score >= 70:
                print("  Rating: GOOD — system shows intelligent trade selection")
            elif score >= 50:
                print("  Rating: FAIR — some improvement needed")
            elif score >= 30:
                print("  Rating: POOR — significant issues with trade selection")
            else:
                print("  Rating: FAILING — system is not selecting trades intelligently")
    elif dry_runs:
        analyze_dry_runs(dry_runs)
    else:
        print("No engine.candidate or dry_run.would_trade entries found in input.")
        print("Make sure you're passing JSON log lines via stdin.")


if __name__ == "__main__":
    main()
