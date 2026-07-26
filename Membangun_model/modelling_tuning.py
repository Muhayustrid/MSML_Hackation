#!/usr/bin/env python3
"""Tune Bank Marketing RandomForest with manual MLflow logging."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modelling import load_data


ROOT = Path(__file__).resolve().parent
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 12],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "namadataset_preprocessing")
    parser.add_argument("--tracking-uri", default=str(ROOT / "mlruns"))
    parser.add_argument("--experiment-name", default="bank-marketing-tuning")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "tuning_artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(disable=True)

    search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        PARAM_GRID,
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name="rf_grid_search_manual") as run:
        started = time.perf_counter()
        search.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started
        model = search.best_estimator_
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues", ax=axis)
        figure.tight_layout()
        figure.savefig(args.artifact_dir / "confusion_matrix.png", dpi=120)
        plt.close(figure)

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc']:.4f}")
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.artifact_dir / "roc_curve.png", dpi=120)
        plt.close(figure)

        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision)
        axis.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        figure.tight_layout()
        figure.savefig(args.artifact_dir / "precision_recall_curve.png", dpi=120)
        plt.close(figure)

        (args.artifact_dir / "classification_report.json").write_text(
            json.dumps(classification_report(y_test, predictions, output_dict=True, zero_division=0), indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).to_csv(
            args.artifact_dir / "feature_importance_or_coefficients.csv", index=False
        )
        pd.DataFrame(search.cv_results_).to_csv(
            args.artifact_dir / "cross_validation_results.csv", index=False
        )
        (args.artifact_dir / "dataset_profile.json").write_text(
            json.dumps(
                {
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "feature_count": X_train.shape[1],
                    "train_positive_rate": y_train.mean(),
                    "test_positive_rate": y_test.mean(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (args.artifact_dir / "model_summary.json").write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "model_type": type(model).__name__,
                    "best_params": search.best_params_,
                    "best_cv_f1": search.best_score_,
                    "metrics": metrics,
                    "training_seconds": training_seconds,
                    "random_state": 42,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_params(search.best_params_)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_metrics(metrics | {"best_cv_f1": float(search.best_score_), "training_seconds": training_seconds})
        mlflow.set_tags(
            {
                "student": "Muhammad_Yusuf_Tri_Daryanto",
                "dicoding_username": "muhammad_yusuf_tdrv4",
                "model_family": "random_forest",
                "logging_mode": "manual",
            }
        )
        mlflow.log_artifacts(str(args.artifact_dir))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
        )
        print(f"run_id: {run.info.run_id}")
        print(f"best_params: {search.best_params_}")
        print(f"metrics: {metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
