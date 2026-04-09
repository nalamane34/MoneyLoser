#!/usr/bin/env python3
"""Comprehensive model training pipeline for all Kalshi market categories.

Loads all settled markets from data/all_markets.json, groups them by category,
extracts features from market data alone (no external data), trains GBM and
logistic regression models per category, calibrates them, and evaluates whether
the model can beat the market's own last_price as a probability estimate.

Usage::

    source .venv/bin/activate && python3 scripts/train_all_models.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "all_markets.json"
MODEL_DIR = ROOT / "models" / "trained"
MIN_MARKETS_PER_CATEGORY = 50


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("financial_hourly", ["KXEURUSDH", "KXUSDJPYH"]),
    ("financial_daily", ["KXNATGASD", "KXBRENTD", "KXCOPPERD", "KXXRPD", "KXDOGED",
                         "KXSILVERD", "KXGOLDD", "KXSTEELW"]),
    ("financial_index", ["NASDAQ100I", "INXI"]),
    ("financial_range", ["KXEURUSD", "KXUSDJPY", "KXWTIW",
                         "KXXRP", "KXDOGE"]),
    ("weather", ["KXTEMPNYCH", "KXHIGH", "KXLOW"]),
    ("sports_parlay", ["KXMVESPORTSMULTIGAMEEXTENDED"]),
    ("cross_category_parlay", ["KXMVECROSSCATEGORY"]),
    ("entertainment", ["KXSPOTIFY", "KXTRUMPMENTION"]),
    ("sports_single", ["KXPGA", "KXMLB", "KXATPMATCH", "KXWTAMATCH",
                        "KXDOTA"]),
    ("economic", ["KXUSPSPEND", "KXUSPINCOME", "KXJOBLESSCLAIMS",
                   "KXPCECORE"]),
    ("compute", ["KXA"]),
]


def classify_market(ticker: str) -> str:
    """Determine the category for a market based on its ticker prefix."""
    for category, prefixes in CATEGORY_RULES:
        for prefix in prefixes:
            if ticker.startswith(prefix):
                return category
    return "other"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_threshold(market: dict) -> float | None:
    """Extract the numeric threshold/strike from a market."""
    # First check floor_strike field
    fs = market.get("floor_strike")
    if fs is not None and fs != "":
        try:
            return float(fs)
        except (ValueError, TypeError):
            pass

    # Try extracting from ticker (e.g. -T47.99 or -T3.195)
    match = re.search(r"-T([\d.]+)$", market.get("ticker", ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def extract_features(market: dict, category: str) -> dict[str, float] | None:
    """Extract feature dict from a single market record.

    Returns None if market lacks sufficient data for training.
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
    created_time_str = market.get("created_time", "")
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        created_dt = datetime.fromisoformat(created_time_str.replace("Z", "+00:00"))
        hours_to_close = (close_dt - created_dt).total_seconds() / 3600.0
        hour_of_day = close_dt.hour
        day_of_week = close_dt.weekday()
    except (ValueError, AttributeError):
        hours_to_close = 0.0
        hour_of_day = 0
        day_of_week = 0

    # Universal features
    features: dict[str, float] = {
        "last_price": last_price,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "spread": yes_ask - yes_bid,
        "log_volume": np.log1p(volume),
        "open_interest": open_interest,
        "hours_to_close": max(hours_to_close, 0.0),
        "hour_of_day": float(hour_of_day),
        "day_of_week": float(day_of_week),
        # Mid price can be more informative than last_price for illiquid markets
        "mid_price": (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else last_price,
        # Spread as fraction of mid
        "spread_pct": (yes_ask - yes_bid) / max((yes_bid + yes_ask) / 2.0, 0.01),
    }

    # Threshold features (for markets with floor_strike)
    threshold = extract_threshold(market)
    if threshold is not None:
        features["threshold"] = threshold
        strike_type = market.get("strike_type", "")
        features["is_above"] = 1.0 if strike_type in ("greater", "greater_or_equal") else 0.0
    else:
        features["threshold"] = 0.0
        features["is_above"] = 0.0

    # Parlay features (for MVE markets)
    legs = market.get("mve_selected_legs")
    if isinstance(legs, list) and len(legs) > 0:
        num_legs = float(len(legs))
        features["num_legs"] = num_legs
        # Implied single-leg probability: if parlay prob = p, each leg ~= p^(1/n)
        if last_price > 0 and last_price < 1:
            features["implied_single_prob"] = last_price ** (1.0 / num_legs)
        else:
            features["implied_single_prob"] = last_price
    else:
        features["num_legs"] = 0.0
        features["implied_single_prob"] = 0.0

    # Liquidity indicator
    liquidity = safe_float(market.get("liquidity_dollars"))
    features["log_liquidity"] = np.log1p(liquidity)

    return features


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probability predictions."""
    return float(np.mean((probs - outcomes) ** 2))


def log_loss_score(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss."""
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Weighted average absolute gap between predicted and actual per bin."""
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
# Model training
# ---------------------------------------------------------------------------

def train_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LGBMClassifier, IsotonicRegression | None, np.ndarray]:
    """Train a LightGBM model with early stopping and isotonic calibration.

    Returns (model, calibrator, test_predictions).
    """
    n = len(X_train)
    # Split train into fit + validation + calibration
    n_val = max(int(n * 0.15), 10)
    n_cal = max(int(n * 0.10), 10)
    n_fit = n - n_val - n_cal

    if n_fit < 20:
        # Too small to split -- use all data
        n_fit = n
        n_val = 0
        n_cal = 0

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=max(20, n_fit // 50),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )

    if n_val > 0:
        X_fit = X_train.iloc[:n_fit]
        y_fit = y_train.iloc[:n_fit]
        X_val = X_train.iloc[n_fit:n_fit + n_val]
        y_val = y_train.iloc[n_fit:n_fit + n_val]

        model.fit(
            X_fit.values, y_fit.values,
            eval_set=[(X_val.values, y_val.values)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
    else:
        model.fit(X_train.values, y_train.values)

    # Calibration
    calibrator = None
    if n_cal > 0:
        X_cal = X_train.iloc[n_fit + n_val:]
        y_cal = y_train.iloc[n_fit + n_val:]
        raw_cal = model.predict_proba(X_cal.values)[:, 1]
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal.values)

    # Predict on test set
    raw_test = model.predict_proba(X_test.values)[:, 1]
    if calibrator is not None:
        test_preds = np.clip(calibrator.predict(raw_test), 0.0, 1.0)
    else:
        test_preds = raw_test

    return model, calibrator, test_preds


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LogisticRegression, StandardScaler, IsotonicRegression | None, np.ndarray]:
    """Train a logistic regression with scaling and isotonic calibration.

    Returns (model, scaler, calibrator, test_predictions).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n = len(X_train)
    n_cal = max(int(n * 0.15), 10) if n >= 60 else 0
    n_fit = n - n_cal

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)

    calibrator = None
    if n_cal > 0:
        model.fit(X_train_s[:n_fit], y_train.iloc[:n_fit].values)
        raw_cal = model.predict_proba(X_train_s[n_fit:])[:, 1]
        y_cal = y_train.iloc[n_fit:].values
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal)
    else:
        model.fit(X_train_s, y_train.values)

    raw_test = model.predict_proba(X_test_s)[:, 1]
    if calibrator is not None:
        test_preds = np.clip(calibrator.predict(raw_test), 0.0, 1.0)
    else:
        test_preds = raw_test

    return model, scaler, calibrator, test_preds


# ---------------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------------

def save_model_artifact(
    name: str,
    model,
    calibrator,
    feature_names: list[str],
    metrics: dict,
    scaler=None,
    feature_importances: dict | None = None,
) -> Path:
    """Save model, calibrator, and metadata to disk."""
    import joblib

    out_dir = MODEL_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_importances": feature_importances or {},
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = out_dir / "model.pkl"
    joblib.dump(artifact, model_path)

    meta_path = out_dir / "meta.json"
    meta = {
        "name": name,
        "feature_names": feature_names,
        "metrics": metrics,
        "feature_importances": feature_importances or {},
        "trained_at": artifact["trained_at"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_category_dataset(markets: list[dict]) -> dict[str, pd.DataFrame]:
    """Group markets by category and extract features, returning DataFrames.

    Each DataFrame has feature columns plus 'label' (0/1) and 'last_price'
    (for market baseline comparison).
    """
    category_rows: dict[str, list[dict]] = {}

    for m in markets:
        ticker = m.get("ticker", "")
        cat = classify_market(ticker)
        features = extract_features(m, cat)
        if features is None:
            continue
        result = m.get("result")
        label = 1.0 if result == "yes" else 0.0
        features["label"] = label
        category_rows.setdefault(cat, []).append(features)

    datasets: dict[str, pd.DataFrame] = {}
    for cat, rows in category_rows.items():
        df = pd.DataFrame(rows)
        datasets[cat] = df

    return datasets


def run_category_training(
    cat: str,
    df: pd.DataFrame,
    save: bool = True,
) -> dict | None:
    """Train GBM + logistic for one category and return results dict."""
    n = len(df)
    if n < MIN_MARKETS_PER_CATEGORY:
        return None

    labels = df["label"].values
    pos_rate = labels.mean()

    # Check we have both classes
    if pos_rate == 0.0 or pos_rate == 1.0:
        print(f"  [SKIP] Only one class present (pos_rate={pos_rate:.3f})")
        return None

    # Separate features and labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Fill NaN with 0
    X = X.fillna(0.0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0.0)

    # Train/test split (stratified)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y,
        )
    except ValueError:
        # Stratification can fail with tiny minority class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42,
        )

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Market baseline: use last_price as probability
    market_probs = X_test["last_price"].values
    outcomes = y_test.values

    # --- GBM ---
    gbm_model, gbm_cal, gbm_preds = train_gbm(X_train, y_train, X_test, y_test)
    gbm_brier = brier_score(gbm_preds, outcomes)
    gbm_ll = log_loss_score(gbm_preds, outcomes)
    gbm_ece = expected_calibration_error(gbm_preds, outcomes)

    # Feature importances
    importances = gbm_model.feature_importances_
    feat_imp = {name: float(imp) for name, imp in zip(feature_cols, importances)}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

    # --- Logistic ---
    lr_model, lr_scaler, lr_cal, lr_preds = train_logistic(X_train, y_train, X_test, y_test)
    lr_brier = brier_score(lr_preds, outcomes)
    lr_ll = log_loss_score(lr_preds, outcomes)
    lr_ece = expected_calibration_error(lr_preds, outcomes)

    # --- Market baseline ---
    mkt_brier = brier_score(market_probs, outcomes)
    mkt_ll = log_loss_score(market_probs, outcomes)
    mkt_ece = expected_calibration_error(market_probs, outcomes)

    results = {
        "category": cat,
        "n_total": n,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_rate": float(pos_rate),
        "market": {"brier": mkt_brier, "log_loss": mkt_ll, "ece": mkt_ece},
        "gbm": {"brier": gbm_brier, "log_loss": gbm_ll, "ece": gbm_ece},
        "logistic": {"brier": lr_brier, "log_loss": lr_ll, "ece": lr_ece},
        "gbm_edge": mkt_brier - gbm_brier,
        "lr_edge": mkt_brier - lr_brier,
        "feature_importances": feat_imp_sorted,
        "feature_names": feature_cols,
    }

    # Save models
    if save:
        save_model_artifact(
            name=f"gbm_{cat}",
            model=gbm_model,
            calibrator=gbm_cal,
            feature_names=feature_cols,
            metrics=results["gbm"],
            feature_importances=feat_imp_sorted,
        )
        save_model_artifact(
            name=f"logistic_{cat}",
            model=lr_model,
            calibrator=lr_cal,
            feature_names=feature_cols,
            metrics=results["logistic"],
            scaler=lr_scaler,
        )

    return results


def run_universal_training(
    datasets: dict[str, pd.DataFrame],
    save: bool = True,
) -> dict | None:
    """Train a universal model on ALL categories combined."""
    all_frames = []
    cat_map: dict[str, int] = {}

    for cat, df in datasets.items():
        if len(df) < 10:
            continue
        cat_idx = len(cat_map)
        cat_map[cat] = cat_idx
        frame = df.copy()
        frame["category_id"] = float(cat_idx)
        all_frames.append(frame)

    if not all_frames:
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Universal model: {len(combined)} markets across {len(cat_map)} categories")

    result = run_category_training("universal", combined, save=save)
    if result:
        result["category_map"] = cat_map
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 90) -> None:
    print(char * width)


def print_category_result(r: dict) -> None:
    """Print results for a single category."""
    cat = r["category"]
    print(f"\n  Category: {cat}")
    print(f"  Markets: {r['n_total']} total  |  Train: {r['n_train']}  |  Test: {r['n_test']}  |  Yes rate: {r['pos_rate']:.3f}")
    print()

    # Metrics table
    print(f"  {'Metric':<12} {'Market':>10} {'GBM':>10} {'Logistic':>10} {'GBM Edge':>10} {'LR Edge':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for metric_name in ["brier", "log_loss", "ece"]:
        mkt = r["market"][metric_name]
        gbm = r["gbm"][metric_name]
        lr = r["logistic"][metric_name]
        gbm_edge = mkt - gbm
        lr_edge = mkt - lr

        # Format edge with sign indicator
        gbm_sign = "+" if gbm_edge > 0 else ""
        lr_sign = "+" if lr_edge > 0 else ""

        print(f"  {metric_name:<12} {mkt:>10.6f} {gbm:>10.6f} {lr:>10.6f} {gbm_sign}{gbm_edge:>9.6f} {lr_sign}{lr_edge:>9.6f}")

    # Edge assessment
    best_edge = max(r["gbm_edge"], r["lr_edge"])
    best_model = "GBM" if r["gbm_edge"] >= r["lr_edge"] else "Logistic"
    if best_edge > 0.001:
        print(f"\n  --> {best_model} BEATS market by {best_edge:.6f} Brier points")
    elif best_edge > 0:
        print(f"\n  --> {best_model} marginally better ({best_edge:.6f}), likely noise")
    else:
        print(f"\n  --> NO EDGE: Market price is better than both models")

    # Top features
    fi = r["feature_importances"]
    top_feats = list(fi.items())[:5]
    if top_feats:
        total_imp = sum(fi.values()) or 1.0
        print(f"\n  Top features (GBM importance):")
        for fname, fval in top_feats:
            pct = 100.0 * fval / total_imp
            print(f"    {fname:<25} {fval:>6.0f}  ({pct:>5.1f}%)")


def main() -> None:
    print_separator()
    print("  MONEYGONE - Comprehensive Model Training Pipeline")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print_separator()

    # Load data
    print(f"\n  Loading markets from {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        markets = json.load(f)
    print(f"  Loaded {len(markets)} markets")

    # Build datasets
    print("  Extracting features and grouping by category ...")
    datasets = build_category_dataset(markets)

    # Summary of categories
    print(f"\n  {'Category':<30} {'Total':>8} {'Settled':>8} {'Yes%':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(datasets.keys()):
        df = datasets[cat]
        n = len(df)
        pos_rate = df["label"].mean()
        print(f"  {cat:<30} {n:>8} {n:>8} {pos_rate:>7.1%}")

    # Train per-category models
    print_separator("-")
    print("  PER-CATEGORY MODEL TRAINING")
    print_separator("-")

    all_results: list[dict] = []
    skipped: list[str] = []

    for cat in sorted(datasets.keys()):
        df = datasets[cat]
        n = len(df)

        if n < MIN_MARKETS_PER_CATEGORY:
            skipped.append(f"{cat} ({n} markets)")
            continue

        print(f"\n  Training models for: {cat} ({n} settled markets) ...")
        result = run_category_training(cat, df)
        if result:
            print_category_result(result)
            all_results.append(result)
        else:
            skipped.append(f"{cat} (single class)")

    if skipped:
        print(f"\n  Skipped categories (< {MIN_MARKETS_PER_CATEGORY} markets or single class):")
        for s in skipped:
            print(f"    - {s}")

    # Train universal model
    print_separator("-")
    print("  UNIVERSAL MODEL (ALL CATEGORIES)")
    print_separator("-")

    universal = run_universal_training(datasets)
    if universal:
        print_category_result(universal)
        all_results.append(universal)

    # Final summary
    print_separator("=")
    print("  FINAL SUMMARY")
    print_separator("=")

    print(f"\n  {'Category':<28} {'N':>6} {'Mkt Brier':>10} {'GBM Brier':>10} {'LR Brier':>10} {'Best Edge':>10} {'Winner':>10}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in all_results:
        cat = r["category"]
        n = r["n_test"]
        mkt_b = r["market"]["brier"]
        gbm_b = r["gbm"]["brier"]
        lr_b = r["logistic"]["brier"]
        best_edge = max(r["gbm_edge"], r["lr_edge"])
        winner = "GBM" if r["gbm_edge"] >= r["lr_edge"] else "LR"
        if best_edge <= 0:
            winner = "MARKET"

        sign = "+" if best_edge > 0 else ""
        print(f"  {cat:<28} {n:>6} {mkt_b:>10.6f} {gbm_b:>10.6f} {lr_b:>10.6f} {sign}{best_edge:>9.6f} {winner:>10}")

    # Key insight
    print()
    print_separator("-")
    print("  KEY INSIGHT")
    print_separator("-")
    edge_cats = [r for r in all_results if max(r["gbm_edge"], r["lr_edge"]) > 0.001]
    no_edge = [r for r in all_results if max(r["gbm_edge"], r["lr_edge"]) <= 0.001]

    if edge_cats:
        print(f"\n  Categories where model BEATS market (Brier edge > 0.001):")
        for r in edge_cats:
            best_e = max(r["gbm_edge"], r["lr_edge"])
            print(f"    {r['category']}: edge = {best_e:.6f}")
    else:
        print("\n  NO categories show meaningful edge over market price.")

    if no_edge:
        print(f"\n  Categories where market price is already well-calibrated:")
        for r in no_edge:
            best_e = max(r["gbm_edge"], r["lr_edge"])
            print(f"    {r['category']}: edge = {best_e:.6f}")

    print(f"\n  Models saved to: {MODEL_DIR}")
    print_separator()


if __name__ == "__main__":
    main()
