# Monitoring dan Logging

Bank Marketing serving and monitoring for `muhammad_yusuf_tdrv4`.

## Deterministic Local Serving

Run from the aggregate repository root after tuning. This selects the newest
manual tuning run at runtime; no experiment ID or run ID is fixed.

PowerShell:

```powershell
$root = (Resolve-Path ".").Path
$python = (Resolve-Path (Join-Path $root ".venv\Scripts\python.exe")).Path
$trackingDir = (Resolve-Path (Join-Path $root "Membangun_model\mlruns")).Path
$env:MLFLOW_TRACKING_URI = ([System.Uri]$trackingDir).AbsoluteUri
$resolver = @'
from mlflow import MlflowClient

client = MlflowClient()
experiment_ids = [item.experiment_id for item in client.search_experiments()]
runs = client.search_runs(
    experiment_ids,
    filter_string="tags.logging_mode = 'manual'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    raise SystemExit("Run tuning manual tidak ditemukan")
print(f"{runs[0].info.artifact_uri.rstrip('/')}/model")
'@
$env:MODEL_URI = ($resolver | & $python -).Trim()
& $python (Join-Path $root "Monitoring dan Logging\3.prometheus_exporter.py") --host 127.0.0.1 --port 8088
```

## Local URLs

- API documentation: `http://127.0.0.1:8088/docs`
- Health: `http://127.0.0.1:8088/health`
- Metrics: `http://127.0.0.1:8088/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Evidence Status

K1 and K3 public-repository publication and GitHub Actions runs are verified.
Three genuine UI screenshots remain pending manual capture:
`Membangun_model/screenshoot_dashboard.png`,
`Membangun_model/screenshoot_artifak.png`, and
`Workflow-CI/Workflow Artifact.png`. They are not fabricated by this package.
