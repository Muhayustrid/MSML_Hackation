#!/usr/bin/env python3
"""Inference proxy with Prometheus metrics for Bank Marketing model (Criterion 4).

Exposes:
- POST /predict  — model inference
- GET  /health   — liveness
- GET  /metrics  — Prometheus scrape endpoint
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.responses import Response

RANDOM_STATE = 42
START_TIME = time.time()

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total prediction requests", ["status"]
)
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors", ["type"])
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS_POSITIVE = Counter("predictions_positive_total", "Predicted positive class")
PREDICTIONS_NEGATIVE = Counter("predictions_negative_total", "Predicted negative class")
MODEL_UP = Gauge("model_up", "1 if model is loaded")
PROCESS_CPU = Gauge("process_cpu_percent", "Process CPU percent")
PROCESS_MEMORY = Gauge("process_memory_bytes", "Process RSS memory bytes")
PAYLOAD_VALIDATION_ERRORS = Counter(
    "payload_validation_errors_total", "Invalid payload count"
)
ACTIVE_REQUESTS = Gauge("active_requests", "In-flight prediction requests")
EXPORTER_UPTIME = Gauge("exporter_uptime_seconds", "Exporter uptime seconds")
PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Predicted positive-class probability",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL: Any = None
FEATURE_NAMES: list[str] = []
PROCESS = psutil.Process()


class PredictRequest(BaseModel):
    """Either a single feature map or a list of feature maps / dense vector."""

    dataframe_records: list[dict[str, float]] | None = None
    dataframe_split: dict[str, Any] | None = None
    instances: list[list[float]] | None = None
    features: dict[str, float] | None = None


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    latency_seconds: float
    model_loaded: bool = True


def resolve_model_uri() -> str:
    env_uri = os.environ.get("MODEL_URI")
    if env_uri:
        return env_uri
    # Prefer local tuned model under Membangun_model/mlruns if present
    base = Path(__file__).resolve().parents[1] / "Membangun_model"
    candidates = list((base / "mlruns").glob("*/*/artifacts/model"))
    if candidates:
        # newest by mtime
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0].as_uri() if hasattr(candidates[0], "as_uri") else f"file://{candidates[0]}"
    # sklearn joblib fallback path
    joblib_path = Path(__file__).resolve().parent / "model.joblib"
    if joblib_path.exists():
        return str(joblib_path)
    raise FileNotFoundError(
        "No MODEL_URI set and no local model found. "
        "Train models first or set MODEL_URI=runs:/<id>/model"
    )


def load_model() -> Any:
    uri = resolve_model_uri()
    if uri.endswith(".joblib") or uri.endswith(".pkl"):
        return joblib.load(uri.replace("file://", ""))
    return mlflow.sklearn.load_model(uri)


def load_feature_names() -> list[str]:
    path = (
        Path(__file__).resolve().parents[1]
        / "Membangun_model"
        / "namadataset_preprocessing"
        / "feature_names.json"
    )
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return []


def to_dataframe(payload: PredictRequest) -> pd.DataFrame:
    if payload.dataframe_records is not None:
        return pd.DataFrame(payload.dataframe_records)
    if payload.features is not None:
        return pd.DataFrame([payload.features])
    if payload.dataframe_split is not None:
        cols = payload.dataframe_split.get("columns")
        data = payload.dataframe_split.get("data")
        return pd.DataFrame(data, columns=cols)
    if payload.instances is not None:
        if FEATURE_NAMES and len(payload.instances[0]) == len(FEATURE_NAMES):
            return pd.DataFrame(payload.instances, columns=FEATURE_NAMES)
        return pd.DataFrame(payload.instances)
    raise ValueError("No supported input fields provided")


@asynccontextmanager
async def _model_lifespan(_: FastAPI):
    global MODEL, FEATURE_NAMES
    try:
        FEATURE_NAMES = load_feature_names()
        MODEL = load_model()
        MODEL_UP.set(1)
        print("Model loaded. features=", len(FEATURE_NAMES))
    except Exception as exc:  # noqa: BLE001
        MODEL = None
        MODEL_UP.set(0)
        print(f"WARNING: model not loaded: {exc}")
    yield


# FastAPI requires a lifespan callback signature of (app) -> AsyncContextManager
app = FastAPI(title="Bank Marketing Inference Exporter", lifespan=_model_lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    EXPORTER_UPTIME.set(time.time() - START_TIME)
    PROCESS_CPU.set(PROCESS.cpu_percent(interval=None))
    PROCESS_MEMORY.set(PROCESS.memory_info().rss)
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_up": MODEL is not None,
        "uptime_seconds": time.time() - START_TIME,
        "n_features": len(FEATURE_NAMES),
    }


@app.get("/metrics")
def metrics() -> Response:
    EXPORTER_UPTIME.set(time.time() - START_TIME)
    PROCESS_CPU.set(PROCESS.cpu_percent(interval=None))
    PROCESS_MEMORY.set(PROCESS.memory_info().rss)
    MODEL_UP.set(1 if MODEL is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        if MODEL is None:
            PREDICTION_ERRORS.labels(type="model_unavailable").inc()
            PREDICTION_REQUESTS.labels(status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            df = to_dataframe(payload)
        except Exception as exc:  # noqa: BLE001
            PAYLOAD_VALIDATION_ERRORS.inc()
            PREDICTION_ERRORS.labels(type="validation").inc()
            PREDICTION_REQUESTS.labels(status="error").inc()
            raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc

        if FEATURE_NAMES:
            missing = [c for c in FEATURE_NAMES if c not in df.columns]
            if missing:
                PAYLOAD_VALIDATION_ERRORS.inc()
                PREDICTION_ERRORS.labels(type="schema").inc()
                PREDICTION_REQUESTS.labels(status="error").inc()
                raise HTTPException(status_code=400, detail=f"Missing features: {missing[:5]}...")
            df = df[FEATURE_NAMES]

        with PREDICTION_LATENCY.time():
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(df)[:, 1]
            else:
                proba = MODEL.predict(df).astype(float)
            preds = (np.asarray(proba) >= 0.5).astype(int)

        for p, pr in zip(preds, proba):
            PREDICTION_PROBABILITY.observe(float(pr))
            if int(p) == 1:
                PREDICTIONS_POSITIVE.inc()
            else:
                PREDICTIONS_NEGATIVE.inc()

        latency = time.perf_counter() - start
        PREDICTION_REQUESTS.labels(status="success").inc()
        return PredictResponse(
            predictions=[int(x) for x in preds],
            probabilities=[float(x) for x in proba],
            latency_seconds=latency,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.labels(type="inference").inc()
        PREDICTION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ACTIVE_REQUESTS.dec()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Bank Marketing inference + Prometheus exporter")
    parser.add_argument("--host", default=os.environ.get("EXPORTER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EXPORTER_PORT", "8088")),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", str(Path(__file__).resolve().parent / "served_model")),
    )
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = str(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
