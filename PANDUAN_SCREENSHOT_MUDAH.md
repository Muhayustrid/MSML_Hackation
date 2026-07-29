# Panduan Screenshot Mudah

Jalankan semua perintah di PowerShell dari root submission saat ini. Panduan ini hanya untuk tiga screenshot yang belum ada.

## 1. Copy dan Jalankan Blok Ini

Salin dan jalankan seluruh blok dari root submission. Instalasi pertama dapat memakan beberapa menit; pengulangan biasanya lebih cepat.

```powershell
$root = (Resolve-Path ".").Path
if (-not (Test-Path (Join-Path $root "Membangun_model\requirements.txt"))) { throw "Jalankan dari root submission" }
if (-not (Test-Path (Join-Path $root "Workflow-CI"))) { throw "Folder Workflow-CI tidak ditemukan" }
if (-not (Test-Path (Join-Path $root ".venv_screenshot\Scripts\python.exe"))) {
  py -3.12 -m venv (Join-Path $root ".venv_screenshot")
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $root ".venv_screenshot\Scripts\python.exe"))) { throw "Gagal membuat .venv_screenshot dengan Python 3.12" }
}
$python = (Resolve-Path (Join-Path $root ".venv_screenshot\Scripts\python.exe")).Path
& $python -m pip install -r (Join-Path $root "Membangun_model\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Gagal memasang Membangun_model/requirements.txt" }
$k2 = Join-Path $root "Membangun_model"
$trackingDir = Join-Path $k2 "mlruns"
New-Item -ItemType Directory -Force $trackingDir | Out-Null
Push-Location $k2
try {
  & $python -m pytest -q tests
  if ($LASTEXITCODE -ne 0) { throw "Test K2 gagal" }
  & $python modelling.py --tracking-uri $trackingDir --experiment-name bank-marketing-model-development
  if ($LASTEXITCODE -ne 0) { throw "Training baseline gagal" }
  & $python modelling_tuning.py --tracking-uri $trackingDir --experiment-name bank-marketing-model-development
  if ($LASTEXITCODE -ne 0) { throw "Training tuning gagal" }
} finally {
  Pop-Location
}
& $python -m mlflow ui --backend-store-uri $trackingDir --host 127.0.0.1 --port 5000
```

Buka http://127.0.0.1:5000 setelah terminal menampilkan bahwa server siap. Tekan `Ctrl+C` hanya setelah dua screenshot MLflow selesai.

## 2. Ambil Tiga Screenshot

1. **Dashboard MLflow:** buka experiment `bank-marketing-model-development`. Pastikan tabel menampilkan run baseline dan tuning terbaru beserta kolom nama/status run, waktu, parameter `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`, dan `random_state`, serta metric `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, dan `test_roc_auc`. Simpan screenshot asli sebagai `Membangun_model/screenshoot_dashboard.png`.
2. **Artifact MLflow:** buka run `rf_grid_search_manual` terbaru yang bertag `logging_mode=manual`, lalu buka tab **Artifacts**. Pastikan daftar artifact memperlihatkan `model/`, `estimator.html`, dan artifact evaluasi. Simpan screenshot asli sebagai `Membangun_model/screenshoot_artifak.png`.
3. **Artifact GitHub Actions:** buka https://github.com/Muhayustrid/Workflow-CI/actions/runs/30405176099. Pastikan workflow **Model Retraining CI** berstatus sukses dan bagian **Artifacts** menampilkan `ci-training-artifacts`. Simpan screenshot asli sebagai `Workflow-CI/Workflow Artifact.png`.

Jangan membuat screenshot secara programatis atau memakai screenshot lama.

## Jika Ada Error

- `Jalankan dari root submission`: buka PowerShell di folder submission yang berisi `Membangun_model/` dan `Workflow-CI/`, lalu ulangi dari langkah 1.
- Port 5000 sudah dipakai: hentikan proses MLflow lama dengan `Ctrl+C`, lalu ulangi blok di atas.
- Untuk diagnosis lebih lengkap, lihat `PANDUAN_MENJALANKAN_DAN_SCREENSHOT.md`.
