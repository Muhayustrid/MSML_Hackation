#!/usr/bin/env python3
"""Inference client for the Bank Marketing serving endpoint (Criterion 4)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send inference requests to model server")
    p.add_argument(
        "--url",
        default=os.environ.get("INFERENCE_URL", "http://127.0.0.1:8088/predict"),
        help="Prediction endpoint URL",
    )
    p.add_argument(
        "--health-url",
        default=os.environ.get("HEALTH_URL", "http://127.0.0.1:8088/health"),
    )
    p.add_argument(
        "--sample",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Membangun_model"
        / "namadataset_preprocessing"
        / "X_test.csv",
        help="CSV with model-ready features",
    )
    p.add_argument("--n", type=int, default=3, help="Number of rows to send")
    p.add_argument("--timeout", type=float, default=10.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        health = requests.get(args.health_url, timeout=args.timeout)
        print("health_status:", health.status_code, health.text)
    except requests.RequestException as exc:
        print(f"ERROR: cannot reach health endpoint: {exc}", file=sys.stderr)
        return 1

    if not args.sample.exists():
        print(f"ERROR: sample file not found: {args.sample}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.sample).head(args.n)
    # Ensure pure Python floats for strict JSON serialization
    records = json.loads(df.to_json(orient="records"))
    payload = {"dataframe_records": records}

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            args.url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"ERROR: request failed: {exc}", file=sys.stderr)
        return 1
    latency = time.perf_counter() - t0

    print("http_status:", resp.status_code)
    print("client_latency_seconds:", round(latency, 4))
    try:
        body = resp.json()
        print(json.dumps(body, indent=2))
    except Exception:
        print(resp.text)

    if resp.status_code >= 400:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
