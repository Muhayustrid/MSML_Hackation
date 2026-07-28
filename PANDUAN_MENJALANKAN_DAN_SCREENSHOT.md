# Panduan Menjalankan Submission dan Mengambil Screenshot

Panduan ini dijalankan di PowerShell pada Windows dengan Python 3.12. Semua
perintah Python memakai satu virtual environment aggregate `.venv` dan semua
path ditentukan saat runtime.

## 1. Setup dari clone baru

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

Jangan lanjut jika salah satu instalasi gagal.

## 2. K1: preprocessing dan notebook

Jalankan test dan preprocessing otomatis:

```powershell
$k1 = Join-Path $root "Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto"
& $python -m pytest -q (Join-Path $k1 "tests")
& $python (Join-Path $k1 "preprocessing\automate_Muhammad_Yusuf_Tri_Daryanto.py") `
  --input (Join-Path $k1 "namadataset_raw\bank-full.csv") `
  --output (Join-Path $k1 "preprocessing\bank_marketing_preprocessing")
```

Eksekusi ulang notebook eksperimen:

```powershell
Push-Location (Join-Path $k1 "preprocessing")
& $python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  "Eksperimen_Muhammad_Yusuf_Tri_Daryanto.ipynb"
Pop-Location
```

## 3. K2: baseline, tuning, dan MLflow UI

```powershell
$k2 = Join-Path $root "Membangun_model"
$trackingDir = Join-Path $k2 "mlruns"
New-Item -ItemType Directory -Force $trackingDir | Out-Null
& $python -m pytest -q (Join-Path $k2 "tests")
Push-Location $k2
& $python modelling.py --tracking-uri $trackingDir
& $python modelling_tuning.py --tracking-uri $trackingDir
Pop-Location
& $python -m mlflow ui --backend-store-uri $trackingDir --host 127.0.0.1 --port 5000
```

Biarkan terminal MLflow UI tetap hidup dan buka http://127.0.0.1:5000. Hentikan
dengan `Ctrl+C` setelah kedua screenshot MLflow selesai diambil.

### Screenshot dashboard MLflow

1. Buka experiment `bank-marketing-model-development`.
2. Pastikan tabel menampilkan run baseline dan run tuning manual terbaru.
3. Tampilkan nama experiment, nama/status run, parameter utama, serta metric
   accuracy, precision, recall, F1, dan ROC AUC.
4. Ambil screenshot asli dan simpan tepat sebagai
   `Membangun_model/screenshoot_dashboard.png`.

### Screenshot artifact MLflow

1. Buka run tuning manual terbaru, lalu pilih tab **Artifacts**.
2. Pastikan pohon artifact memperlihatkan `model/`, `estimator.html`, dan
   artifact evaluasi seperti `confusion_matrix.png`, kurva ROC/precision-recall,
   classification report, serta model summary.
3. Ambil screenshot asli dan simpan tepat sebagai
   `Membangun_model/screenshoot_artifak.png`.

## 4. K3: MLflow Project lokal

```powershell
$k3 = Join-Path $root "Workflow-CI"
$ciTrackingDir = Join-Path $k3 "MLProject\mlruns"
New-Item -ItemType Directory -Force $ciTrackingDir | Out-Null
$env:PATH = "$(Split-Path -Parent $python);$env:PATH"
$env:MLFLOW_TRACKING_URI = ([System.Uri]$ciTrackingDir).AbsoluteUri
Push-Location $k3
& $python -m pytest -q ".\MLProject\test_modelling.py"
& $python -m mlflow run MLProject --env-manager local `
  --experiment-name bank-marketing-ci -P data_dir=namadataset_preprocessing
Pop-Location
```

Run harus selesai dengan status sukses dan menghasilkan `run_summary.json`,
`estimator.html`, `confusion_matrix.png`, serta artifact `model/`.

## 5. K4: monitoring model tuning terbaru

Pilih run terbaru yang mempunyai tag `logging_mode=manual` melalui
`MlflowClient`. Perintah ini tidak memakai experiment ID atau run ID tetap.

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
& $python -m pytest -q (Join-Path $root "Monitoring dan Logging")
& $python (Join-Path $root "Monitoring dan Logging\3.prometheus_exporter.py") `
  --host 127.0.0.1 --port 8088
```

Saat exporter hidup, buka terminal PowerShell kedua dari root aggregate:

```powershell
$root = (Resolve-Path ".").Path
$python = (Resolve-Path (Join-Path $root ".venv\Scripts\python.exe")).Path
& $python (Join-Path $root "Monitoring dan Logging\7.inference.py") --n 3
Start-Process "http://127.0.0.1:8088/health"
Start-Process "http://127.0.0.1:8088/metrics"
```

Pastikan health menunjukkan model aktif, inference sukses, dan metric prediksi
muncul. Hentikan exporter dengan `Ctrl+C`.

## 6. Screenshot artifact GitHub Actions

1. Buka run terverifikasi:
   https://github.com/Muhayustrid/Workflow-CI/actions/runs/30405176099
2. Pastikan halaman menunjukkan workflow **Model Retraining CI** berstatus
   sukses dan bagian **Artifacts** menampilkan `ci-training-artifacts`.
3. Ambil screenshot asli halaman tersebut dan simpan tepat sebagai
   `Workflow-CI/Workflow Artifact.png`.

## 7. Aturan bukti dan commit setelah capture

Screenshot harus berasal dari UI MLflow dan GitHub Actions yang benar-benar
dijalankan. **Jangan memalsukan, membuat secara programatis, atau memakai ulang
screenshot lama/referensi.** Periksa isi dan path sebelum commit.

Commit dua screenshot MLflow ke branch aggregate:

```powershell
git add -- "Membangun_model/screenshoot_dashboard.png" "Membangun_model/screenshoot_artifak.png"
git commit -m "docs: add genuine MLflow evidence"
git push origin repair/submission-reviewer
```

Commit screenshot workflow di repositori standalone Workflow-CI:

```powershell
Push-Location (Join-Path $root "Workflow-CI")
git add -- "Workflow Artifact.png"
git commit -m "docs: add genuine workflow evidence"
git push origin main
Pop-Location
```

Sebelum push, pastikan tidak ada virtual environment, `mlruns`, cache, token,
atau file hasil download Actions yang ikut ter-stage.

## 8. Pemeriksaan statis dokumentasi

Jalankan dari root aggregate untuk memastikan panduan tetap reproducible:

```powershell
$readme = Get-Content -LiteralPath "README.md" -Raw
$guide = Get-Content -LiteralPath "PANDUAN_MENJALANKAN_DAN_SCREENSHOT.md" -Raw

$requiredReadme = @(
  "git clone https://github.com/Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto.git",
  "git clone https://github.com/Muhayustrid/Workflow-CI.git",
  "Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\requirements.txt",
  "Membangun_model\requirements.txt",
  "Workflow-CI\MLProject\requirements.txt",
  "Monitoring dan Logging\requirements.txt"
)
$requiredGuide = @(
  "Membangun_model/screenshoot_dashboard.png",
  "Membangun_model/screenshoot_artifak.png",
  "Workflow-CI/Workflow Artifact.png",
  "http://127.0.0.1:5000",
  "https://github.com/Muhayustrid/Workflow-CI/actions/runs/30405176099",
  "from mlflow import MlflowClient",
  "tags.logging_mode = 'manual'",
  "attributes.start_time DESC",
  "model/",
  "estimator.html"
)

foreach ($value in $requiredReadme) {
  if (-not $readme.Contains($value)) { throw "README tidak memuat: $value" }
}
foreach ($value in $requiredGuide) {
  if (-not $guide.Contains($value)) { throw "Panduan tidak memuat: $value" }
}
if (($readme + $guide) -match "(?i)\b[a-f0-9]{32}\b") {
  throw "Ditemukan run ID MLflow tetap"
}
if (($readme + $guide) -match "(?i)\b[A-Z]:\\") {
  throw "Ditemukan path absolut drive Windows"
}
"STATIC_CHECK_OK"
```
