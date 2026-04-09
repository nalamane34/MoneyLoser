#!/usr/bin/env python3
"""Train a logistic regression model on settled KXTEMPNYCH weather markets.

Parses features from the ticker format and market data, trains a logistic
regression model, evaluates it, and saves the trained model.

Ticker format: KXTEMPNYCH-26APR0909-T47.99
  - 26 = year suffix (2026)
  - APR = month
  - 09 = day
  - 09 = hour (EDT)
  - T47.99 = temperature threshold

Usage::

    python scripts/train_weather_model.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict

import structlog

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.data.store import DataStore

log = structlog.get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "settled_markets.json"
DB_PATH = DATA_DIR / "moneygone.duckdb"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Ticker pattern: KXTEMPNYCH-{YY}{MON}{DD}{HH}-T{threshold}
TICKER_RE = re.compile(
    r"^KXTEMPNYCH-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T([\d.]+)$"
)


def parse_ticker_features(ticker: str) -> dict | None:
    """Parse features from a KXTEMPNYCH ticker string.

    Returns a dict with: threshold, hour, month, day, or None if the
    ticker doesn't match the expected pattern.
    """
    m = TICKER_RE.match(ticker)
    if not m:
        return None

    year_suffix, month_str, day_str, hour_str, threshold_str = m.groups()

    month = MONTH_MAP.get(month_str)
    if month is None:
        return None

    return {
        "threshold": float(threshold_str),
        "hour": int(hour_str),
        "month": month,
        "day": int(day_str),
        "year": 2000 + int(year_suffix),
    }


def build_dataset(markets: list[dict]) -> pd.DataFrame:
    """Build a feature DataFrame from weather market data.

    Each row is one market with features extracted from the ticker and
    market data, plus the binary yes/no result as the label.
    """
    records = []
    skipped = 0

    for m in markets:
        ticker = m["ticker"]
        features = parse_ticker_features(ticker)
        if features is None:
            skipped += 1
            continue

        result = m.get("result")
        if result not in ("yes", "no"):
            skipped += 1
            continue

        # Parse market data
        last_price = _parse_dollar(m.get("last_price_dollars"))
        volume = _parse_fp(m.get("volume_fp"))
        open_interest = _parse_fp(m.get("open_interest_fp"))
        yes_bid = _parse_dollar(m.get("yes_bid_dollars"))
        yes_ask = _parse_dollar(m.get("yes_ask_dollars"))

        record = {
            "ticker": ticker,
            "threshold": features["threshold"],
            "hour": features["hour"],
            "month": features["month"],
            "day": features["day"],
            "last_price": last_price if last_price is not None else 0.0,
            "volume": volume if volume is not None else 0,
            "open_interest": open_interest if open_interest is not None else 0,
            "yes_bid": yes_bid if yes_bid is not None else 0.0,
            "yes_ask": yes_ask if yes_ask is not None else 0.0,
            "spread": (yes_ask or 0.0) - (yes_bid or 0.0),
            "label": 1 if result == "yes" else 0,
        }
        records.append(record)

    if skipped:
        print(f"  Skipped {skipped} markets (unparseable ticker or missing result)")

    return pd.DataFrame(records)


def _parse_dollar(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_fp(val) -> int | None:
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("WEATHER MODEL TRAINING - KXTEMPNYCH Logistic Regression")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print("\n[1/5] Loading weather market data ...")

    # Try DuckDB first, fall back to JSON
    weather_markets = []
    if DB_PATH.exists():
        print(f"  Reading from DuckDB: {DB_PATH}")
        store = DataStore(DB_PATH)
        store.initialize_schema()
        rows = store._conn.execute(
            "SELECT * FROM market_states WHERE ticker LIKE 'KXTEMPNYCH%'"
        ).fetchall()
        if rows:
            cols = [desc[0] for desc in store._conn.description]
            for row in rows:
                d = dict(zip(cols, row))
                # Map back to the API field names for consistency
                weather_markets.append({
                    "ticker": d["ticker"],
                    "event_ticker": d["event_ticker"],
                    "title": d["title"],
                    "status": d["status"],
                    "yes_bid_dollars": str(d["yes_bid"]) if d["yes_bid"] is not None else None,
                    "yes_ask_dollars": str(d["yes_ask"]) if d["yes_ask"] is not None else None,
                    "last_price_dollars": str(d["last_price"]) if d["last_price"] is not None else None,
                    "volume_fp": str(d["volume"]) if d["volume"] is not None else None,
                    "open_interest_fp": str(d["open_interest"]) if d["open_interest"] is not None else None,
                    "result": d["result"],
                })
        store.close()

    if not weather_markets:
        print(f"  Falling back to JSON: {JSON_PATH}")
        with open(JSON_PATH) as f:
            all_markets = json.load(f)
        weather_markets = [m for m in all_markets if m["ticker"].startswith("KXTEMPNYCH")]

    print(f"  Found {len(weather_markets)} KXTEMPNYCH markets")

    # ---------------------------------------------------------------
    # 2. Build feature matrix
    # ---------------------------------------------------------------
    print("\n[2/5] Building feature matrix ...")

    df = build_dataset(weather_markets)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Threshold range: [{df['threshold'].min():.2f}, {df['threshold'].max():.2f}]")
    print(f"  Hours present: {sorted(df['hour'].unique())}")

    feature_cols = ["threshold", "hour", "month", "day", "last_price", "volume",
                    "open_interest", "yes_bid", "yes_ask", "spread"]

    X = df[feature_cols]
    y = df["label"]

    print("\n  Feature summary:")
    print(X.describe().to_string())

    # ---------------------------------------------------------------
    # 3. Train model
    # ---------------------------------------------------------------
    print("\n[3/5] Training logistic regression ...")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    # Use cross-validation predictions for evaluation
    if len(df) >= 10:
        n_splits = min(5, len(df))
        cv_probs = cross_val_predict(model, X, y, cv=n_splits, method="predict_proba")[:, 1]
        has_cv = True
    else:
        cv_probs = None
        has_cv = False
        print("  WARNING: Too few samples for cross-validation")

    # Train the final model on all data
    model.fit(X.values, y.values)
    train_probs = model.predict_proba(X.values)[:, 1]

    print(f"  Model trained on {len(X)} samples with {len(feature_cols)} features")

    # ---------------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------------
    print("\n[4/5] Evaluation ...")

    y_np = y.values.astype(float)

    # Training set metrics
    train_brier = brier_score_loss(y_np, train_probs)
    train_logloss = log_loss(y_np, train_probs)
    train_ece = expected_calibration_error(y_np, train_probs)

    print("\n  --- Training Set Metrics ---")
    print(f"  Brier Score:  {train_brier:.4f}  (lower is better, perfect=0)")
    print(f"  Log Loss:     {train_logloss:.4f}  (lower is better)")
    print(f"  ECE:          {train_ece:.4f}  (lower is better, perfect=0)")

    if has_cv:
        cv_brier = brier_score_loss(y_np, cv_probs)
        cv_logloss = log_loss(y_np, cv_probs)
        cv_ece = expected_calibration_error(y_np, cv_probs)

        print("\n  --- Cross-Validated Metrics (out-of-fold) ---")
        print(f"  Brier Score:  {cv_brier:.4f}")
        print(f"  Log Loss:     {cv_logloss:.4f}")
        print(f"  ECE:          {cv_ece:.4f}")

    # Feature importances (coefficients)
    print("\n  Feature Coefficients:")
    coefs = model.coef_[0]
    for name, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:20s}  {coef:+.4f}")
    print(f"    {'intercept':20s}  {model.intercept_[0]:+.4f}")

    # Calibration table
    print("\n  Calibration Table (predicted vs actual):")
    if has_cv:
        probs_for_table = cv_probs
    else:
        probs_for_table = train_probs

    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    print(f"    {'Bin':>12s}  {'Count':>6s}  {'Predicted':>10s}  {'Actual':>8s}  {'Gap':>8s}")
    for i in range(n_bins):
        mask = (probs_for_table >= bin_edges[i]) & (probs_for_table < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        pred_mean = probs_for_table[mask].mean()
        actual_mean = y_np[mask].mean()
        gap = actual_mean - pred_mean
        bin_label = f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})"
        print(f"    {bin_label:>12s}  {mask.sum():>6d}  {pred_mean:>10.3f}  {actual_mean:>8.3f}  {gap:>+8.3f}")

    # ---------------------------------------------------------------
    # 5. Save model
    # ---------------------------------------------------------------
    print("\n[5/5] Saving model ...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "weather_logistic_v1.pkl"

    # Save using the LogisticModel wrapper for compatibility
    try:
        from moneygone.models.trainers.logistic import LogisticModel

        wrapper = LogisticModel(version="1.0.0", calibration_method=None, C=1.0, max_iter=1000)
        wrapper._model = model
        wrapper._feature_names = feature_cols
        wrapper._fitted = True
        wrapper.trained_at = pd.Timestamp.now(tz="UTC").to_pydatetime()
        wrapper.save(model_path)
        print(f"  Saved (LogisticModel wrapper) to {model_path}")
    except Exception as exc:
        print(f"  Could not save via LogisticModel wrapper ({exc}), using raw joblib ...")
        joblib.dump(
            {
                "model": model,
                "feature_names": feature_cols,
                "version": "1.0.0",
                "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "metrics": {
                    "train_brier": train_brier,
                    "train_logloss": train_logloss,
                    "train_ece": train_ece,
                    "cv_brier": cv_brier if has_cv else None,
                    "cv_logloss": cv_logloss if has_cv else None,
                    "cv_ece": cv_ece if has_cv else None,
                    "n_samples": len(X),
                },
            },
            model_path,
        )
        print(f"  Saved (raw joblib) to {model_path}")

    # Verify saved model
    print("\n  Verifying saved model ...")
    try:
        loaded = LogisticModel.load(model_path)
        test_features = {name: float(X.iloc[0][name]) for name in feature_cols}
        pred = loaded.predict_proba(test_features)
        print(f"  Verification OK: P(yes)={pred.probability:.4f} for first sample")
    except Exception as exc:
        print(f"  Verification via wrapper failed: {exc}")
        loaded_raw = joblib.load(model_path)
        print(f"  Raw model loaded successfully, features: {loaded_raw.get('feature_names', 'N/A')}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
