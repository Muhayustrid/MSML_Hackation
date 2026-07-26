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

Baseline run ID: `3bf76f2042c74814b2eea5b5eff2ccec`

| Baseline metric | Value |
|---|---:|
| `test_accuracy` | 0.8364480813889196 |
| `test_precision` | 0.3696594427244582 |
| `test_recall` | 0.5642722117202268 |
| `test_f1` | 0.44668911335578004 |
| `test_roc_auc` | 0.7928300109018209 |

Baseline evidence before cleanup: status `FINISHED`; exactly one `model`
directory; `MLmodel` contained a signature and saved input-example metadata;
`input_example.json` and `serving_input_example.json` were present.

Tuning run ID: `99ea27f07c574968bf9c2d2396505a44`

Best parameters: `class_weight=balanced`, `max_depth=12`,
`min_samples_leaf=2`, `n_estimators=200`. Best cross-validation F1:
`0.44810330731858267`.

| Tuning metric | Value |
|---|---:|
| `test_accuracy` | 0.8342364259648347 |
| `test_precision` | 0.36804308797127466 |
| `test_recall` | 0.5812854442344045 |
| `test_f1` | 0.4507145474532796 |
| `test_roc_auc` | 0.7966022066421798 |

Tuning evidence before cleanup: status `FINISHED`; model artifact `model`; and
all eight files in `tuning_artifacts/` were logged. The local
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
python -m mlflow ui --backend-store-uri ./mlruns
```

## Screenshot manual gate

`screenshoot_dashboard.png` and `screenshoot_artifak.png` are intentionally
omitted. The inspected seed images are genuine Bank Marketing MLflow screens,
but they show an older randomized-search run rather than the run IDs above;
the artifact image also shows the Overview tab instead of the artifact tree.
After rerunning the commands, manually capture the new experiment dashboard
and the tuning run's expanded Artifacts tab. Do not substitute old or reference
evidence.
