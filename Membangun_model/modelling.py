#!/usr/bin/env python3
"""Train the Bank Marketing baseline with MLflow sklearn autolog."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "namadataset_preprocessing")
    parser.add_argument("--tracking-uri", default=str(ROOT / "mlruns"))
    parser.add_argument("--experiment-name", default="bank-marketing-model-development")
    return parser.parse_args(argv)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed artifacts in {data_dir}: {missing}")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train_frame = pd.read_csv(data_dir / "y_train.csv")
    y_test_frame = pd.read_csv(data_dir / "y_test.csv")
    if "y" not in y_train_frame.columns or "y" not in y_test_frame.columns:
        raise ValueError("Target column 'y' missing")
    y_train = y_train_frame["y"]
    y_test = y_test_frame["y"]
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature schemas differ")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts differ")
    if X_train.empty or X_test.empty:
        raise ValueError("Processed feature data is empty")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Processed features contain null values")
    if not set(y_train.unique()).issubset({0, 1}) or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Targets must be binary")
    if "duration" in X_train.columns or "duration" in X_test.columns:
        raise ValueError("Leakage column 'duration' found in features")
    return X_train, X_test, y_train, y_test


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main() -> int:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="baseline_random_forest_autolog") as run:
        model = build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "student": "Muhammad_Yusuf_Tri_Daryanto",
                "dicoding_username": "muhammad_yusuf_tdrv4",
                "logging_mode": "autolog",
            }
        )
        print(f"run_id: {run.info.run_id}")
        print(f"metrics: {metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
