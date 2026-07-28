# Membangun Model - Bank Marketing

Student: **Muhammad Yusuf Tri Daryanto**
Dicoding username: `muhammad_yusuf_tdrv4`

This K2 package consumes the 50 train-ready features produced by K1. The
post-call leakage feature `duration` is excluded.

Verified environment: Python `3.12.10`, MLflow `2.19.0`, scikit-learn
`1.5.2`, NumPy `2.5.1`, and pandas `2.3.3`.

## Training contracts

- Baseline: `RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)`.
- Baseline logging: MLflow sklearn autolog with model signature and input example, plus five explicit test metrics. Autolog is the only model logger and creates one artifact at `model`.
- Tuning: `GridSearchCV`, scoring `f1`, and `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`.
- Grid: `n_estimators=[100, 200]`, `max_depth=[None, 12]`, `min_samples_leaf=[1, 2]`, and `class_weight=["balanced"]` (8 candidates, 24 fits).
- Tuning logging: manual only; the best model is logged at `model` with eight additional evaluation artifacts.

## Verified results

Baseline run ID: `962c5cd22de5433d975411c230abe145`

| Baseline metric | Value |
|---|---:|
| `test_accuracy` | 0.8372607589335104 |
| `test_precision` | 0.3700440528634361 |
| `test_recall` | 0.555765595463138 |
| `test_f1` | 0.4442765394786551 |
| `test_roc_auc` | 0.7932855181339235 |

Baseline evidence after regeneration: status `FINISHED`; exactly one `model`
directory; `MLmodel` contained a signature and saved input-example metadata;
`input_example.json` and `serving_input_example.json` were present.

Tuning run ID: `a101eca9146c4bd5b650b36df8667d06`

Best parameters: `class_weight=balanced`, `max_depth=12`,
`min_samples_leaf=2`, `n_estimators=200`. Best cross-validation F1:
`0.443618423990767`.

| Tuning metric | Value |
|---|---:|
| `test_accuracy` | 0.8356012833278017 |
| `test_precision` | 0.3696711327649208 |
| `test_recall` | 0.5737240075614367 |
| `test_f1` | 0.44962962962962966 |
| `test_roc_auc` | 0.7968723094476035 |

Tuning evidence after regeneration: status `FINISHED`; model artifact `model`;
and all nine files in `tuning_artifacts/` were logged. The local
`tuning_artifacts/` directory is retained in this package. Generated `mlruns`
tracking data is intentionally excluded.

## Run locally

Install `requirements.txt` in a clean Python 3.12 environment, then run from
this directory in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_modelling.py
python modelling.py --tracking-uri ./mlruns
python modelling_tuning.py --tracking-uri ./mlruns
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

## Screenshot manual gate

`screenshoot_dashboard.png` and `screenshoot_artifak.png` are intentionally
omitted. The inspected seed images are genuine Bank Marketing MLflow screens,
but they show an older randomized-search run rather than the run IDs above;
the artifact image also shows the Overview tab instead of the artifact tree.
After rerunning the commands, screenshots must show the real experiment table
and the tuning run's expanded artifact tree. Do not substitute old or reference
evidence.
