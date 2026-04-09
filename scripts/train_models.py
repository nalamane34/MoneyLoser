#!/usr/bin/env python3
"""Model training CLI.

Loads features from the DataStore, trains the specified model type,
evaluates calibration on a held-out set, and registers the trained model
in the ModelRegistry.

Usage::

    python scripts/train_models.py --model-type gbm --category weather
    python scripts/train_models.py --model-type logistic --train-start 2024-01-01 --train-end 2024-06-30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneygone.config import load_config
from moneygone.data.store import DataStore
from moneygone.utils.logging import setup_logging

log = structlog.get_logger(__name__)


def _parse_date(s: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format to a UTC datetime."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _load_training_data(
    store: DataStore,
    category: str | None,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and labels from the data store.

    Returns
    -------
    tuple
        ``(X, y)`` where X is a DataFrame of features and y is a Series
        of binary outcomes.
    """
    # Query all predictions with outcomes in the time range
    conn = store._conn  # noqa: SLF001 — internal access for training

    query = """
        SELECT DISTINCT p.ticker, p.prediction_time
        FROM predictions p
        JOIN settlements_log s ON p.ticker = s.ticker
        WHERE p.prediction_time >= ? AND p.prediction_time <= ?
        ORDER BY p.prediction_time
    """
    results = conn.execute(query, [start, end]).fetchall()

    if not results:
        log.warning(
            "train.no_data",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return pd.DataFrame(), pd.Series(dtype=float)

    feature_rows: list[dict[str, float]] = []
    labels: list[float] = []

    for ticker, pred_time in results:
        # Get features at prediction time
        features = store.get_features_at(ticker, pred_time)
        if not features:
            continue

        # Get settlement outcome
        settlements = store.get_settlements(ticker)
        if not settlements:
            continue

        settlement = settlements[-1]  # Most recent
        outcome = 1.0 if settlement["market_result"] in ("yes", "all_yes") else 0.0

        feature_rows.append(features)
        labels.append(outcome)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="outcome")

    # Fill NaN with column medians
    X = X.fillna(X.median())

    log.info(
        "train.data_loaded",
        n_samples=len(X),
        n_features=X.shape[1] if not X.empty else 0,
        positive_rate=round(y.mean(), 3) if len(y) > 0 else 0,
    )

    return X, y


def _train_logistic(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a logistic regression model with isotonic calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(
        base_model, method="isotonic", cv=5
    )
    calibrated.fit(X_scaled, y)

    return {
        "type": "logistic",
        "model": calibrated,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def _train_gbm(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a LightGBM model with isotonic calibration."""
    from sklearn.calibration import CalibratedClassifierCV

    import lightgbm as lgb

    base_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )

    calibrated = CalibratedClassifierCV(
        base_model, method="isotonic", cv=5
    )
    calibrated.fit(X, y)

    return {
        "type": "gbm",
        "model": calibrated,
        "feature_names": list(X.columns),
    }


def _train_bayesian(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a Bayesian logistic regression model."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use BayesianRidge as a regression model, then calibrate
    from sklearn.linear_model import LogisticRegression

    base_model = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Strong regularization for Bayesian-like behavior
        solver="lbfgs",
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(
        base_model, method="isotonic", cv=5
    )
    calibrated.fit(X_scaled, y)

    return {
        "type": "bayesian",
        "model": calibrated,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def _evaluate(
    model_artifact: dict,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained model on a held-out set."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    model = model_artifact["model"]
    scaler = model_artifact.get("scaler")

    X_input = X_eval
    if scaler is not None:
        X_input = scaler.transform(X_eval)

    probas = model.predict_proba(X_input)[:, 1]

    metrics: dict[str, float] = {
        "brier_score": round(float(brier_score_loss(y_eval, probas)), 6),
        "log_loss": round(float(log_loss(y_eval, probas)), 6),
        "n_eval": len(y_eval),
    }

    # AUC requires both classes
    if len(set(y_eval)) > 1:
        metrics["auc"] = round(float(roc_auc_score(y_eval, probas)), 6)
    else:
        metrics["auc"] = 0.0

    # ECE
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probas >= bin_edges[i]) & (probas < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probas >= bin_edges[i]) & (probas <= bin_edges[i + 1])
        count = mask.sum()
        if count > 0:
            avg_pred = probas[mask].mean()
            avg_outcome = y_eval.values[mask].mean()
            ece += (count / len(probas)) * abs(avg_pred - avg_outcome)
    metrics["ece"] = round(float(ece), 6)

    log.info("train.evaluation", **metrics)
    return metrics


def _save_model(
    model_artifact: dict,
    metrics: dict[str, float],
    config,
    model_type: str,
    category: str | None,
) -> Path:
    """Save the trained model to disk."""
    import pickle

    model_dir = Path(config.model.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cat_suffix = f"_{category}" if category else ""
    model_name = f"{model_type}{cat_suffix}_{timestamp}"
    model_path = model_dir / f"{model_name}.pkl"
    meta_path = model_dir / f"{model_name}_meta.json"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "category": category,
        "feature_names": model_artifact.get("feature_names", []),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(
        "train.model_saved",
        model_path=str(model_path),
        meta_path=str(meta_path),
    )
    return model_path


def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    config = load_config(base_path=Path(args.config))
    setup_logging(config.log_level)

    log.info(
        "train_models.starting",
        model_type=args.model_type,
        category=args.category,
    )

    # Parse date ranges
    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    eval_start = _parse_date(args.eval_start)
    eval_end = _parse_date(args.eval_end)

    # Open data store
    db_path = Path(config.data_dir) / "moneygone.duckdb"
    if not db_path.exists():
        log.error("train.db_not_found", path=str(db_path))
        sys.exit(1)

    store = DataStore(db_path)

    try:
        # Load data
        X_train, y_train = _load_training_data(
            store, args.category, train_start, train_end
        )
        X_eval, y_eval = _load_training_data(
            store, args.category, eval_start, eval_end
        )

        if X_train.empty or y_train.empty:
            log.error("train.no_training_data")
            sys.exit(1)

        if X_eval.empty or y_eval.empty:
            log.warning("train.no_eval_data_using_train_split")
            # Fall back to train/test split
            from sklearn.model_selection import train_test_split

            X_train, X_eval, y_train, y_eval = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        # Ensure consistent features
        common_features = list(set(X_train.columns) & set(X_eval.columns))
        if not common_features:
            log.error("train.no_common_features")
            sys.exit(1)

        X_train = X_train[common_features]
        X_eval = X_eval[common_features]

        log.info(
            "train.data_ready",
            train_samples=len(X_train),
            eval_samples=len(X_eval),
            features=len(common_features),
        )

        # Train
        trainers = {
            "logistic": _train_logistic,
            "gbm": _train_gbm,
            "bayesian": _train_bayesian,
        }

        trainer = trainers.get(args.model_type)
        if trainer is None:
            log.error("train.unknown_model_type", model_type=args.model_type)
            sys.exit(1)

        log.info("train.training_started", model_type=args.model_type)
        model_artifact = trainer(X_train, y_train)

        # Evaluate
        metrics = _evaluate(model_artifact, X_eval, y_eval)

        # Save
        model_path = _save_model(
            model_artifact, metrics, config, args.model_type, args.category
        )

        log.info(
            "train_models.complete",
            model_path=str(model_path),
            brier=metrics["brier_score"],
            ece=metrics["ece"],
        )

    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train prediction models from recorded data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "gbm", "bayesian"],
        default="gbm",
        help="Model type to train (default: gbm)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Market category to filter training data (default: all)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2024-01-01",
        help="Training period start (YYYY-MM-DD, default: 2024-01-01)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2024-09-30",
        help="Training period end (YYYY-MM-DD, default: 2024-09-30)",
    )
    parser.add_argument(
        "--eval-start",
        type=str,
        default="2024-10-01",
        help="Evaluation period start (YYYY-MM-DD, default: 2024-10-01)",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default="2024-12-31",
        help="Evaluation period end (YYYY-MM-DD, default: 2024-12-31)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
