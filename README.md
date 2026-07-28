# Submission Membangun Sistem Machine Learning

Muhammad Yusuf Tri Daryanto (`muhammad_yusuf_tdrv4`)

## Repositori publik

- Eksperimen dan preprocessing: https://github.com/Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto
- Workflow retraining: https://github.com/Muhayustrid/Workflow-CI

## Setup baru

Jalankan di PowerShell. Aggregate tidak menyimpan K1/K3 sebagai gitlink, jadi
keduanya harus di-clone setelah aggregate.

```powershell
git clone --branch repair/submission-reviewer --single-branch https://github.com/Muhayustrid/MSML_Hackation.git SMSML_Muhammad_Yusuf_Tri_Daryanto
Set-Location .\SMSML_Muhammad_Yusuf_Tri_Daryanto
git clone https://github.com/Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto.git
git clone https://github.com/Muhayustrid/Workflow-CI.git

$root = (Resolve-Path ".").Path
py -3.12 -m venv (Join-Path $root ".venv")
$python = (Resolve-Path (Join-Path $root ".venv\Scripts\python.exe")).Path
& $python -m pip install --upgrade pip
& $python -m pip install `
  -r (Join-Path $root "Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\requirements.txt") `
  -r (Join-Path $root "Membangun_model\requirements.txt") `
  -r (Join-Path $root "Workflow-CI\MLProject\requirements.txt") `
  -r (Join-Path $root "Monitoring dan Logging\requirements.txt")
```

## Model monitoring

Setelah menjalankan tuning, pilih run manual terbaru melalui `MlflowClient`;
tidak ada experiment ID atau run ID tetap.

```powershell
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

Langkah lengkap untuk preprocessing, training, MLflow Project, monitoring, dan
screenshot asli ada di [PANDUAN_MENJALANKAN_DAN_SCREENSHOT.md](PANDUAN_MENJALANKAN_DAN_SCREENSHOT.md).
