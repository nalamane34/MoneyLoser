#!/usr/bin/env python3
"""Full-scale model training on 83K+ historical traded markets.

Loads historical_markets.json (~1GB, 500K markets), filters to the ~83K
that actually traded (volume > 0 AND last_price > 0), groups them into
high-volume categories, and trains per-category LightGBM models with
isotonic calibration.

Models are ONLY saved when they demonstrably beat the market baseline
(Model_Brier < Market_Brier).

Usage::

    source .venv/bin/activate && python3 scripts/train_full_models.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "historical_markets.json"
MODEL_DIR = ROOT / "models" / "trained"
MIN_MARKETS_PER_CATEGORY = 100

# ---------------------------------------------------------------------------
# Category definitions — high-volume groups
# ---------------------------------------------------------------------------

CATEGORY_PREFIXES: dict[str, list[str]] = {
    "crypto_daily": [
        "KXBTCD", "KXETHD", "KXXRPD", "KXSOLD", "KXDOGED",
    ],
    "crypto_range": [
        "KXBTC-", "KXETH-", "KXXRP-", "KXSOLE", "KXDOGE-",
        # Exclude 15M by requiring the prefix NOT to match crypto_15m
    ],
    "crypto_15m": [
        "KXBTC15M", "KXETH15M",
    ],
    "index": [
        "KXINXU", "KXNASDAQ100U",
    ],
    "sports_game": [
        "KXNCAAMBGAME", "KXNBAGAME", "KXNHLGAME", "KXNFLGAME",
        "KXNCAAWBGAME", "KXNCAAFGAME", "KXCBAGAME",
        "KXEFLCHAMPIONSHIPGAME", "KXATPMATCH", "KXWTAMATCH",
    ],
    "sports_spread": [
        "KXNCAAMBSPREAD", "KXNBASPREAD", "KXNHLSPREAD",
        "KXNFLSPREAD", "KXNCAAFSPREAD", "KXEPLSPREAD",
    ],
    "sports_total": [
        "KXNCAAMBTOTAL", "KXNBATOTAL", "KXNHLTOTAL",
        "KXNFLTOTAL", "KXNCAAFTOTAL",
    ],
    "sports_props": [
        "KXNBAPTS", "KXNFLRECYDS", "KXNHLGOAL", "KXNHLFIRSTGOAL",
        "KXNBAREB", "KXNBA3PT", "KXNFLRSHYDS", "KXNBAAST",
        "KXNFLANYTD", "KXNFLFIRSTTD", "KXNFLPASSYDS", "KXNHLPTS",
        "KXNFLREC", "KXNFLPASSTDS", "KXNFL2TD", "KXNBA2D",
        "KXNHLAST", "KXNBA3D", "KXNFLTEAMTOTAL",
    ],
}


def classify_market(ticker: str) -> str | None:
    """Classify a market ticker into a category. Returns None if no match."""
    # Check crypto_15m FIRST since KXBTC15M would also match KXBTC-
    for prefix in CATEGORY_PREFIXES["crypto_15m"]:
        if ticker.startswith(prefix):
            return "crypto_15m"

    # Check crypto_daily before crypto_range (KXBTCD vs KXBTC)
    for prefix in CATEGORY_PREFIXES["crypto_daily"]:
        if ticker.startswith(prefix):
            return "crypto_daily"

    # Now check remaining categories
    for category, prefixes in CATEGORY_PREFIXES.items():
        if category in ("crypto_15m", "crypto_daily"):
            continue
        for prefix in prefixes:
            if prefix.endswith("-"):
                # Exact prefix match (avoid KXBTC matching KXBTCD)
                if ticker.startswith(prefix):
                    return category
            else:
                if ticker.startswith(prefix):
                    return category

    return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def safe_float(val, default: float = 0.0) -> float:
    """Safely parse a value to float."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_features(market: dict) -> dict[str, float] | None:
    """Extract feature vector from a single market.

    Returns None if the market lacks required data.
    """
    result = market.get("result")
    if result not in ("yes", "no"):
        return None

    last_price = safe_float(market.get("last_price_dollars"))
    yes_bid = safe_float(market.get("yes_bid_dollars"))
    yes_ask = safe_float(market.get("yes_ask_dollars"))
    volume = safe_float(market.get("volume_fp"))
    open_interest = safe_float(market.get("open_interest_fp"))

    # Parse times
    close_time_str = market.get("close_time", "")
    open_time_str = market.get("open_time", "")
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        open_dt = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
        hours_open = max((close_dt - open_dt).total_seconds() / 3600.0, 0.0)
        hour_of_close = close_dt.hour
        day_of_week = close_dt.weekday()
        # Use close_time as a sortable timestamp for time-based splitting
        close_ts = close_dt.timestamp()
    except (ValueError, AttributeError):
        hours_open = 0.0
        hour_of_close = 0
        day_of_week = 0
        close_ts = 0.0

    # Price momentum: direction of most recent price move
    prev_price = safe_float(market.get("previous_price_dollars"))
    price_momentum = last_price - prev_price  # positive = price moving up (YES favoured)
    price_accel = abs(price_momentum)         # magnitude of move regardless of direction

    # Settlement timer: markets with longer timers are less locked-in at close
    settlement_timer = safe_float(market.get("settlement_timer_seconds"), default=1800.0)
    settlement_timer_days = settlement_timer / 86400.0

    features: dict[str, float] = {
        "last_price": last_price,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "spread": yes_ask - yes_bid,
        "volume_log": np.log1p(volume),
        "open_interest_log": np.log1p(open_interest),
        "hours_open": hours_open,
        "hour_of_close": float(hour_of_close),
        "day_of_week": float(day_of_week),
        # New features (all available pre-resolution)
        "price_momentum": price_momentum,
        "price_accel": price_accel,
        "prev_price": prev_price,
        "settlement_timer_days": settlement_timer_days,
        "_close_ts": close_ts,  # for time-based split, excluded from training
    }

    # Threshold features (for floor_strike markets)
    floor_strike = market.get("floor_strike")
    if floor_strike is not None and floor_strike != "":
        try:
            features["threshold"] = float(floor_strike)
        except (ValueError, TypeError):
            features["threshold"] = 0.0
            features["is_above"] = 0.0
        else:
            strike_type = market.get("strike_type", "")
            features["is_above"] = 1.0 if "greater" in str(strike_type) else 0.0
    else:
        features["threshold"] = 0.0
        features["is_above"] = 0.0

    # Label
    features["label"] = 1.0 if result == "yes" else 0.0

    return features


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def log_loss_score(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def expected_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10,
) -> float:
    n = len(probs)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = mask.sum()
        if count > 0:
            ece += (count / n) * abs(outcomes[mask].mean() - probs[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "last_price", "yes_bid", "yes_ask", "spread",
    "volume_log", "open_interest_log",
    "hours_open", "hour_of_close", "day_of_week",
    "threshold", "is_above",
    # New features (all available pre-resolution, no leakage)
    "price_momentum", "price_accel", "prev_price",
    "settlement_timer_days",
]


def train_category(
    cat: str,
    df: pd.DataFrame,
) -> dict | None:
    """Train a LightGBM model for one category.

    Uses time-based 80/20 split. Within the 80% train portion, splits
    again 80/20 for fit/calibration.

    Returns results dict or None if category has insufficient data.
    """
    n = len(df)
    if n < MIN_MARKETS_PER_CATEGORY:
        return None

    # Sort by close time for temporal split
    df = df.sort_values("_close_ts").reset_index(drop=True)

    labels = df["label"].values
    pos_rate = float(labels.mean())

    # Need both classes
    if pos_rate == 0.0 or pos_rate == 1.0:
        print(f"  [SKIP] {cat}: only one class (pos_rate={pos_rate:.3f})")
        return None

    # 80/20 time-based split
    split_idx = int(n * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Verify both classes in train and test
    if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        print(f"  [SKIP] {cat}: single class in train or test after temporal split")
        return None

    X_train_full = train_df[FEATURE_COLS].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_train_full = train_df["label"]
    X_test = test_df[FEATURE_COLS].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_test = test_df["label"]

    # Split train into fit (80%) and calibration (20%)
    n_train = len(X_train_full)
    cal_split = int(n_train * 0.80)
    X_fit = X_train_full.iloc[:cal_split]
    y_fit = y_train_full.iloc[:cal_split]
    X_cal = X_train_full.iloc[cal_split:]
    y_cal = y_train_full.iloc[cal_split:]

    # Further split fit into fit_actual (85%) and validation (15%) for early stopping
    n_fit = len(X_fit)
    val_split = int(n_fit * 0.85)
    X_fit_actual = X_fit.iloc[:val_split]
    y_fit_actual = y_fit.iloc[:val_split]
    X_val = X_fit.iloc[val_split:]
    y_val = y_fit.iloc[val_split:]

    # Train LightGBM with conservative hyperparameters
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=25,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )

    if len(X_val) >= 10:
        model.fit(
            X_fit_actual.values, y_fit_actual.values,
            eval_set=[(X_val.values, y_val.values)],
            callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)],
        )
    else:
        model.fit(X_fit.values, y_fit.values)

    # Isotonic calibration
    calibrator = None
    if len(X_cal) >= 20:
        raw_cal = model.predict_proba(X_cal.values)[:, 1]
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal.values)

    # Predict on test
    raw_test = model.predict_proba(X_test.values)[:, 1]
    if calibrator is not None:
        model_preds = np.clip(calibrator.predict(raw_test), 0.0, 1.0)
    else:
        model_preds = raw_test

    # Evaluate
    outcomes = y_test.values
    market_probs = X_test["last_price"].values

    model_brier = brier_score(model_preds, outcomes)
    market_brier = brier_score(market_probs, outcomes)
    edge = market_brier - model_brier

    model_ece = expected_calibration_error(model_preds, outcomes)
    market_ece = expected_calibration_error(market_probs, outcomes)

    model_ll = log_loss_score(model_preds, outcomes)
    market_ll = log_loss_score(market_probs, outcomes)

    # Feature importances
    importances = model.feature_importances_
    total_imp = float(importances.sum()) or 1.0
    feat_imp = {
        name: float(imp)
        for name, imp in sorted(
            zip(FEATURE_COLS, importances), key=lambda x: -x[1]
        )
    }

    result = {
        "category": cat,
        "n_total": n,
        "n_train": len(X_train_full),
        "n_test": len(X_test),
        "pos_rate": pos_rate,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "edge": edge,
        "model_ece": model_ece,
        "market_ece": market_ece,
        "model_logloss": model_ll,
        "market_logloss": market_ll,
        "feature_importances": feat_imp,
        "model": model,
        "calibrator": calibrator,
    }

    return result


def save_model(cat: str, model, calibrator, metrics: dict) -> Path:
    """Save a trained model + calibrator + metadata."""
    out_dir = MODEL_DIR / f"full_{cat}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "feature_names": FEATURE_COLS,
        "metrics": {
            "model_brier": metrics["model_brier"],
            "market_brier": metrics["market_brier"],
            "edge": metrics["edge"],
            "model_ece": metrics["model_ece"],
            "market_ece": metrics["market_ece"],
            "model_logloss": metrics["model_logloss"],
            "market_logloss": metrics["market_logloss"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "pos_rate": metrics["pos_rate"],
        },
        "feature_importances": metrics["feature_importances"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "category": cat,
    }

    model_path = out_dir / "model.pkl"
    joblib.dump(artifact, model_path)

    meta = {k: v for k, v in artifact.items() if k not in ("model", "calibrator")}
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 95
    dash = "-" * 95

    print(sep)
    print("  MONEYGONE -- Full-Scale Model Training (83K+ Historical Markets)")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(sep)

    # ---- Load data ----
    print(f"\n  Loading {DATA_PATH.name} ...")
    with open(DATA_PATH) as f:
        all_markets = json.load(f)
    print(f"  Loaded {len(all_markets):,} total markets")

    # ---- Filter to traded markets ----
    traded = []
    for m in all_markets:
        vol = safe_float(m.get("volume_fp"))
        lp = safe_float(m.get("last_price_dollars"))
        if vol > 0 and lp > 0:
            traded.append(m)
    print(f"  Traded markets (volume > 0 AND last_price > 0): {len(traded):,}")

    # Free memory
    del all_markets

    # ---- Group by category ----
    category_markets: dict[str, list[dict]] = {}
    unclassified = 0
    for m in traded:
        ticker = m.get("ticker", "")
        cat = classify_market(ticker)
        if cat is None:
            unclassified += 1
            continue
        category_markets.setdefault(cat, []).append(m)

    print(f"  Classified into {len(category_markets)} categories ({unclassified:,} unclassified)")

    # ---- Extract features per category ----
    category_dfs: dict[str, pd.DataFrame] = {}
    print(f"\n  {'Category':<20} {'Raw':>8} {'Featured':>10} {'Yes%':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8}")

    for cat in sorted(category_markets.keys()):
        markets = category_markets[cat]
        rows = []
        for m in markets:
            feats = extract_features(m)
            if feats is not None:
                rows.append(feats)
        if rows:
            df = pd.DataFrame(rows)
            category_dfs[cat] = df
            pos_rate = df["label"].mean()
            print(f"  {cat:<20} {len(markets):>8,} {len(df):>10,} {pos_rate:>7.1%}")

    # Free memory
    del traded, category_markets

    # ---- Train per-category models ----
    print(f"\n{dash}")
    print("  PER-CATEGORY MODEL TRAINING")
    print(dash)

    all_results: list[dict] = []
    saved_models: list[str] = []
    skipped: list[str] = []

    for cat in sorted(category_dfs.keys()):
        df = category_dfs[cat]
        n = len(df)

        if n < MIN_MARKETS_PER_CATEGORY:
            skipped.append(f"{cat} ({n} markets)")
            continue

        print(f"\n  Training: {cat} ({n:,} traded markets) ...")
        result = train_category(cat, df)

        if result is None:
            skipped.append(f"{cat} (insufficient data or single class)")
            continue

        all_results.append(result)

        # Print per-category results
        edge_sign = "+" if result["edge"] > 0 else ""
        print(f"    Model Brier:  {result['model_brier']:.6f}")
        print(f"    Market Brier: {result['market_brier']:.6f}")
        print(f"    Edge:         {edge_sign}{result['edge']:.6f}")
        print(f"    Model ECE:    {result['model_ece']:.6f}  |  Market ECE: {result['market_ece']:.6f}")
        print(f"    Log loss:     {result['model_logloss']:.6f}  |  Market:     {result['market_logloss']:.6f}")

        # Feature importances
        print(f"    Feature importances:")
        for fname, fval in list(result["feature_importances"].items())[:6]:
            total = sum(result["feature_importances"].values()) or 1.0
            pct = 100.0 * fval / total
            print(f"      {fname:<22} {fval:>5.0f}  ({pct:>5.1f}%)")

        # Save ONLY if model beats market
        if result["edge"] > 0:
            path = save_model(cat, result["model"], result["calibrator"], result)
            saved_models.append(cat)
            print(f"    --> SAVED to {path}")
        else:
            print(f"    --> NOT SAVED (no edge over market)")

    if skipped:
        print(f"\n  Skipped categories:")
        for s in skipped:
            print(f"    - {s}")

    # ---- Summary table ----
    print(f"\n{sep}")
    print("  COMPREHENSIVE SUMMARY")
    print(sep)

    header = (
        f"  {'Category':<20} {'Markets':>8} {'Traded':>8} "
        f"{'Model_Brier':>12} {'Market_Brier':>13} {'Edge':>9} "
        f"{'ECE_model':>10} {'ECE_market':>11}"
    )
    print(header)
    divider = (
        f"  {'-'*20} {'-'*8} {'-'*8} "
        f"{'-'*12} {'-'*13} {'-'*9} "
        f"{'-'*10} {'-'*11}"
    )
    print(divider)

    for r in sorted(all_results, key=lambda x: -x["edge"]):
        edge_sign = "+" if r["edge"] > 0 else ""
        marker = " *" if r["edge"] > 0 else ""
        print(
            f"  {r['category']:<20} {r['n_total']:>8,} {r['n_test']:>8,} "
            f"{r['model_brier']:>12.6f} {r['market_brier']:>13.6f} "
            f"{edge_sign}{r['edge']:>8.6f} "
            f"{r['model_ece']:>10.6f} {r['market_ece']:>11.6f}{marker}"
        )

    # ---- Key insight ----
    print(f"\n{dash}")
    print("  KEY INSIGHTS")
    print(dash)

    edge_cats = [r for r in all_results if r["edge"] > 0]
    no_edge_cats = [r for r in all_results if r["edge"] <= 0]

    if edge_cats:
        print(f"\n  Categories where MODEL BEATS MARKET (positive edge):")
        for r in sorted(edge_cats, key=lambda x: -x["edge"]):
            print(
                f"    {r['category']:<20} edge=+{r['edge']:.6f}  "
                f"({r['n_test']:,} test markets, "
                f"model={r['model_brier']:.4f} vs market={r['market_brier']:.4f})"
            )
        print(f"\n  --> Focus trading on: {', '.join(r['category'] for r in sorted(edge_cats, key=lambda x: -x['edge']))}")
    else:
        print("\n  NO categories show model edge over market pricing.")
        print("  The market is well-calibrated across all categories.")

    if no_edge_cats:
        print(f"\n  Categories where market price is already optimal (no edge):")
        for r in sorted(no_edge_cats, key=lambda x: x["edge"]):
            edge_sign = "+" if r["edge"] > 0 else ""
            print(
                f"    {r['category']:<20} edge={edge_sign}{r['edge']:.6f}  "
                f"(market wins)"
            )

    # ---- Final stats ----
    print(f"\n{dash}")
    print(f"  Models trained: {len(all_results)}")
    print(f"  Models saved (beat market): {len(saved_models)}")
    if saved_models:
        print(f"  Saved categories: {', '.join(saved_models)}")
    print(f"  Models directory: {MODEL_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
