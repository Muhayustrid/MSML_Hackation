# Submission Membangun Sistem Machine Learning

Muhammad Yusuf Tri Daryanto (`muhammad_yusuf_tdrv4`)

## Public repositories

- Experiment and preprocessing: https://github.com/Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto
- Retraining workflow: https://github.com/Muhayustrid/Workflow-CI

## Local commands

All Python commands use the experiment virtual environment. Run these commands
from the aggregate repository root in PowerShell.

```powershell
$python = (Resolve-Path ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts\python.exe").Path

# K1 tests and preprocessing
& $python -m pytest -q ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\tests"
& $python ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\preprocessing\automate_Muhammad_Yusuf_Tri_Daryanto.py" --input ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\namadataset_raw\bank-full.csv" --output ".\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\preprocessing\bank_marketing_preprocessing"

# K2 tests, baseline, and tuning
& $python -m pytest -q ".\Membangun_model\tests"
Push-Location ".\Membangun_model"
& $python modelling.py --tracking-uri ./mlruns
& $python modelling_tuning.py --tracking-uri ./mlruns
Pop-Location

# K4 tests and monitoring smoke service
& $python -m pytest -q ".\Monitoring dan Logging"
$env:MODEL_URI = (Resolve-Path ".\Membangun_model\mlruns\922817180905645765\a101eca9146c4bd5b650b36df8667d06\artifacts\model").Path
& $python ".\Monitoring dan Logging\3.prometheus_exporter.py" --host 127.0.0.1 --port 8088

# K3 MLflow Project
$env:PATH = "$(Resolve-Path '.\Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\.venv\Scripts');$env:PATH"
Push-Location ".\Workflow-CI"
New-Item -ItemType Directory -Force ".\MLProject\mlruns" | Out-Null
$env:MLFLOW_TRACKING_URI = ([System.Uri]((Resolve-Path ".\MLProject\mlruns").Path)).AbsoluteUri
& $python -m mlflow run MLProject --env-manager local --experiment-name bank-marketing-ci -P data_dir=namadataset_preprocessing
Pop-Location
```

Stop the monitoring service with `Ctrl+C` after the smoke check.
