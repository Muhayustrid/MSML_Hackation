# Submission Reviewer Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce runnable, publicly reviewable experiment and retraining repositories, consistent MLflow model evidence, and a final submission package without breaking monitoring.

**Architecture:** One deterministic preprocessing implementation generates the seven artifacts copied to K1, K2, and K3. K2 records comparable baseline and tuning runs in one local MLflow experiment; K3 retrains from the same processed data through an MLflow Project and uploads its run store through GitHub Actions. The aggregate repository retains K2/K4 and links to standalone public K1/K3 repositories.

**Tech Stack:** Python 3.12, pandas, scikit-learn 1.5.2, MLflow 2.19.0, pytest, Jupyter nbconvert, GitHub Actions, FastAPI, Prometheus, Grafana.

## Global Constraints

- Criterion 1 target is Advanced; Criterion 3 target is Skilled.
- Keep UCI Bank Marketing, target `y`, Random Forest, and `random_state=42`.
- Do not add Docker Hub integration.
- Do not rename or remove monitoring endpoints, port 8088, Prometheus metrics, Grafana dashboard, or alert rules.
- Use only relative or runtime-resolved paths; never commit workstation-specific absolute paths.
- Never fabricate screenshots or claim an external workflow passed before inspecting the real UI/run.
- Keep the seven required preprocessing artifacts in every rubric-required location.
- Do not commit credentials, virtual environments, caches, or local MLflow stores unless the rubric explicitly requires the artifact.

## File Map

- `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/preprocess_core.py`: authoritative preprocessing and validation.
- `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/automate_Muhammad_Yusuf_Tri_Daryanto.py`: command-line entry point with repository-relative defaults.
- `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/tests/test_preprocess.py`: preprocessing contracts and regression checks.
- `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/Eksperimen_Muhammad_Yusuf_Tri_Daryanto.ipynb`: executed EDA and preprocessing evidence.
- `Membangun_model/modelling.py`: baseline run and shared processed-data contract.
- `Membangun_model/modelling_tuning.py`: tuning run and explicit evaluation artifacts.
- `Membangun_model/tests/test_modelling.py`: model/data/MLflow artifact contracts.
- `Monitoring dan Logging/requirements.txt`: serving dependency compatibility.
- `Monitoring dan Logging/test_exporter.py`: final-model inference compatibility.
- `Workflow-CI/MLProject/modelling.py`: MLflow Project training entry point.
- `Workflow-CI/MLProject/MLproject`: canonical MLflow Project descriptor.
- `Workflow-CI/.github/workflows/mlflow.yml`: automated retraining and artifact upload.
- `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto.txt` and `Workflow-CI.txt`: actual public repository URLs.

---

### Task 0: Establish Local Standalone Repository Boundaries

**Files:**
- Create: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/.gitignore`
- Create: `Workflow-CI/.gitignore`

**Interfaces:**
- Consumes: the two currently untracked directories in the aggregate repository.
- Produces: two independent local Git repositories on branch `main`, with no public remote yet.

- [ ] **Step 1: Confirm neither directory is already a repository**

Run `git status --short` inside each directory. Expected: Git reports that it is not a repository. If either directory already has `.git`, inspect its status and remotes instead of reinitializing it.

- [ ] **Step 2: Add minimal ignore rules before the first commit**

Use this content in both standalone repositories:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
```

Also add these Workflow-CI-only lines:

```gitignore
MLProject/mlruns/
MLProject/training_outputs/
```

Do not ignore raw data, processed data, notebooks, workflows, model source, or screenshot evidence.

- [ ] **Step 3: Initialize local repositories**

Run `git init -b main` once in each standalone directory. Do not add a remote and do not push in this task.

- [ ] **Step 4: Commit the existing local baselines**

Inspect status, stage only rubric-required source/data/docs, and create these commits:

```bash
git commit -m "chore: import experiment submission"
git commit -m "chore: import retraining submission"
```

Run the first command only in the experiment repository and the second only in Workflow-CI.

### Task 1: Correct And Validate Preprocessing

**Files:**
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/tests/test_preprocess.py`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/preprocess_core.py`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/automate_Muhammad_Yusuf_Tri_Daryanto.py`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/requirements.txt`

**Interfaces:**
- Consumes: semicolon-delimited `bank-full.csv` with `REQUIRED_COLUMNS` and target values `yes`/`no`.
- Produces: `fit_transform_split(df, test_size=0.2, random_state=42) -> dict[str, Any]` and the existing seven saved artifacts.

- [ ] **Step 1: Add failing preprocessing regression tests**

Add tests that require runtime-root CLI defaults, reject invalid split sizes and empty data, and guarantee no feature vector crosses train/test:

```python
def test_invalid_test_size_is_rejected(output_dir: Path) -> None:
    with pytest.raises(PreprocessingError, match="test_size"):
        run_preprocessing(RAW_CSV, output_dir, test_size=1.0)


def test_empty_dataset_is_rejected(tmp_path: Path, output_dir: Path) -> None:
    empty = tmp_path / "empty.csv"
    pd.read_csv(RAW_CSV, sep=";").head(0).to_csv(empty, sep=";", index=False)
    with pytest.raises(PreprocessingError, match="empty"):
        run_preprocessing(empty, output_dir)


def test_no_feature_vector_crosses_split(output_dir: Path) -> None:
    run_preprocessing(RAW_CSV, output_dir)
    data = load_processed(output_dir)
    train_hash = set(pd.util.hash_pandas_object(data["X_train"], index=False))
    test_hash = set(pd.util.hash_pandas_object(data["X_test"], index=False))
    assert train_hash.isdisjoint(test_hash)
```

Import the CLI module with `importlib.util`, call `parse_args([])`, and assert its defaults equal paths under `ROOT`.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run from the experiment repository root:

```powershell
python -m pytest -q tests/test_preprocess.py
```

Expected: the new validation/default-path tests fail against the current implementation.

- [ ] **Step 3: Implement minimal preprocessing corrections**

In `fit_transform_split`:

```python
if not 0 < test_size < 1:
    raise PreprocessingError("test_size must be between 0 and 1")
if df.empty:
    raise PreprocessingError("Input dataset is empty")

validate_schema(df)
validate_target_values(df)
df = map_target(df)
X, y = split_features_target(df)
expected_feats = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

non_numeric = [c for c in NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    raise PreprocessingError(f"Numeric columns have invalid dtypes: {non_numeric}")

work = X[expected_feats].copy()
work[TARGET_COL] = y.to_numpy()
conflicts = work.groupby(expected_feats, dropna=False)[TARGET_COL].nunique()
if (conflicts > 1).any():
    raise PreprocessingError("Identical feature rows have conflicting targets")
work, n_dupes = drop_duplicates(work, subset=expected_feats)
X = work[expected_feats]
y = work[TARGET_COL]
if y.value_counts().size != 2 or y.value_counts().min() < 2:
    raise PreprocessingError("Target requires at least two samples in each class")
```

Change `drop_duplicates` to accept `subset: list[str] | None = None` and pass it to pandas. Use `REPO_ROOT = Path(__file__).resolve().parents[1]` for CLI defaults. Pin `scikit-learn==1.5.2` in the experiment requirements.

- [ ] **Step 4: Run preprocessing tests**

```powershell
python -m pytest -q tests/test_preprocess.py
```

Expected: all preprocessing tests pass.

- [ ] **Step 5: Commit in the standalone experiment repository**

```bash
git add preprocessing/preprocess_core.py preprocessing/automate_Muhammad_Yusuf_Tri_Daryanto.py tests/test_preprocess.py requirements.txt
git commit -m "fix: make preprocessing deterministic and leak-safe"
```

### Task 2: Regenerate And Synchronize Experiment Artifacts

**Files:**
- Regenerate: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/bank_marketing_preprocessing/*`
- Regenerate: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/Eksperimen_Muhammad_Yusuf_Tri_Daryanto.ipynb`
- Replace: `Membangun_model/namadataset_preprocessing/*`
- Replace: `Workflow-CI/MLProject/namadataset_preprocessing/*`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/README.md`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/.github/workflows/preprocessing.yml`

**Interfaces:**
- Consumes: corrected `run_preprocessing` from Task 1.
- Produces: byte-identical copies of `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `preprocessor.joblib`, `feature_names.json`, and `preprocessing_metadata.json`.

- [ ] **Step 1: Run automated preprocessing with default paths**

```powershell
python preprocessing/automate_Muhammad_Yusuf_Tri_Daryanto.py
```

Expected: exit code 0, logged train/test counts, 50 features, and seven output files.

- [ ] **Step 2: Execute the notebook in place**

Before execution, change the two path-display statements in the notebook source to avoid saving workstation paths:

```python
print("REPO_ROOT: .")
print("OUTPUT_DIR:", OUTPUT_DIR.relative_to(REPO_ROOT))
print("Saved artifacts to:", OUTPUT_DIR.relative_to(REPO_ROOT))
```

Run from `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing`:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Eksperimen_Muhammad_Yusuf_Tri_Daryanto.ipynb
```

Expected: exit code 0, all code cells executed, no traceback output, and final dimensions matching generated metadata.

- [ ] **Step 3: Copy generated artifacts to K2 and K3**

Run from the aggregate repository root:

```powershell
$source = "Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto\preprocessing\bank_marketing_preprocessing"
$names = "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "preprocessor.joblib", "feature_names.json", "preprocessing_metadata.json"
foreach ($target in "Membangun_model\namadataset_preprocessing", "Workflow-CI\MLProject\namadataset_preprocessing") {
    foreach ($name in $names) { Copy-Item -Force "$source\$name" "$target\$name" }
}
```

Do not copy caches or temporary notebook files.

- [ ] **Step 4: Verify all copies by SHA-256**

Run a Python check that hashes each required filename in all three directories:

```python
from hashlib import sha256
from pathlib import Path

roots = [
    Path("Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/preprocessing/bank_marketing_preprocessing"),
    Path("Membangun_model/namadataset_preprocessing"),
    Path("Workflow-CI/MLProject/namadataset_preprocessing"),
]
names = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
         "preprocessor.joblib", "feature_names.json", "preprocessing_metadata.json"]
for name in names:
    hashes = {sha256((root / name).read_bytes()).hexdigest() for root in roots}
    assert len(hashes) == 1, name
```

Expected: no assertion failures.

- [ ] **Step 5: Update workflow and README commands**

Keep the workflow command explicit, add `cache: pip` to `actions/setup-python`, and make README commands work on Windows and Unix without absolute paths. Document the output directory and screenshot/workflow evidence gate.

- [ ] **Step 6: Re-run experiment gate**

```powershell
python -m pytest -q tests
python preprocessing/automate_Muhammad_Yusuf_Tri_Daryanto.py
```

Expected: both commands succeed.

### Task 3: Strengthen K2 Data And MLflow Contracts

**Files:**
- Modify: `Membangun_model/tests/test_modelling.py`
- Modify: `Membangun_model/modelling.py`
- Modify: `Membangun_model/modelling_tuning.py`
- Modify: `Membangun_model/requirements.txt`

**Interfaces:**
- Consumes: synchronized 50-column K2 train/test artifacts from Task 2.
- Produces: `load_data(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`, baseline run `baseline_random_forest_autolog`, and tuning run `rf_grid_search_manual` in experiment `bank-marketing-model-development`.

- [ ] **Step 1: Add failing data-contract tests**

Create temporary malformed datasets and assert `load_data` rejects them:

```python
def test_data_contract_rejects_mismatched_rows(tmp_path: Path):
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
        shutil.copy(DATA_DIR / name, tmp_path / name)
    y_train = pandas.read_csv(tmp_path / "y_train.csv").iloc[:-1]
    y_train.to_csv(tmp_path / "y_train.csv", index=False)
    with pytest.raises(ValueError, match="row count"):
        modelling.load_data(tmp_path)


def test_experiment_defaults_match():
    assert modelling.parse_args([]).experiment_name == "bank-marketing-model-development"
    assert modelling_tuning.parse_args([]).experiment_name == "bank-marketing-model-development"
```

Adjust both `parse_args` functions to accept optional argv lists for direct testing.

- [ ] **Step 2: Run focused tests and confirm failure**

```powershell
python -m pytest -q tests/test_modelling.py
```

Expected: new row/schema/default experiment tests fail.

- [ ] **Step 3: Implement the shared data contract**

After reading files in `load_data`, validate:

```python
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train/test feature schemas differ")
if len(X_train) != len(y_train) or len(X_test) != len(y_test):
    raise ValueError("Feature and target row counts differ")
if X_train.empty or X_test.empty:
    raise ValueError("Processed feature data is empty")
if X_train.isna().any().any() or X_test.isna().any().any():
    raise ValueError("Processed features contain null values")
if not set(y_train.unique()).issubset({0, 1}) or not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Targets must be binary")
```

Check target column existence before indexing and use the unified experiment default in both scripts. Keep `scikit-learn==1.5.2`.

- [ ] **Step 4: Add estimator HTML to manual tuning**

Use the installed sklearn representation, without another dependency:

```python
from sklearn.utils import estimator_html_repr

(args.artifact_dir / "estimator.html").write_text(
    estimator_html_repr(model), encoding="utf-8"
)
```

The existing `mlflow.log_artifacts` then records it with the other evaluation artifacts.

- [ ] **Step 5: Run K2 unit tests**

```powershell
python -m pytest -q tests/test_modelling.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit K2 code changes in the aggregate repository**

```bash
git add Membangun_model/modelling.py Membangun_model/modelling_tuning.py Membangun_model/tests/test_modelling.py Membangun_model/requirements.txt
git commit -m "fix: align model training and MLflow contracts"
```

### Task 4: Regenerate And Verify K2 MLflow Evidence

**Files:**
- Regenerate locally: `Membangun_model/mlruns/`
- Regenerate: `Membangun_model/tuning_artifacts/*`
- Modify: `Membangun_model/README.md`

**Interfaces:**
- Consumes: K2 scripts and processed data from Tasks 2-3.
- Produces: at least two FINISHED MLflow runs in one experiment and a tuning artifact directory containing `estimator.html` plus existing reports/plots.

- [ ] **Step 1: Remove only stale generated K2 tracking data**

Delete `Membangun_model/mlruns/` after confirming it is untracked. Do not delete committed `tuning_artifacts/` before the replacement run succeeds.

- [ ] **Step 2: Run baseline and tuning**

From `Membangun_model`:

```powershell
python modelling.py --tracking-uri ./mlruns
python modelling_tuning.py --tracking-uri ./mlruns
```

Expected: both commands exit 0 and print distinct run IDs.

- [ ] **Step 3: Verify runs through MLflow API**

Use `MlflowClient(tracking_uri="./mlruns")` to assert one experiment named `bank-marketing-model-development`, at least two FINISHED runs, each with `test_f1`, and both model artifact directories. Assert the tuning run contains `estimator.html`, `confusion_matrix.png`, `roc_curve.png`, and `classification_report.json`.

- [ ] **Step 4: Update README from generated data**

Read run IDs and metrics from MLflow rather than retaining old values. Document:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

State that screenshots must show the real experiment table and expanded tuning artifact tree.

- [ ] **Step 5: Commit reproducible K2 evidence files**

```bash
git add Membangun_model/README.md Membangun_model/tuning_artifacts
git commit -m "docs: refresh verified MLflow training evidence"
```

Do not commit `Membangun_model/mlruns/`; retain it locally for UI screenshots and monitoring verification.

### Task 5: Preserve Monitoring Compatibility

**Files:**
- Modify: `Monitoring dan Logging/requirements.txt`
- Modify: `Monitoring dan Logging/test_exporter.py`
- Modify: `Monitoring dan Logging/README.md`
- Modify: `Monitoring dan Logging/PANDUAN_CAPTURE_BUKTI.md`

**Interfaces:**
- Consumes: final K2 MLflow model through environment variable `MODEL_URI` and existing feature names.
- Produces: unchanged `/health`, `/predict`, `/metrics`, port 8088, and existing Prometheus metric names.

- [ ] **Step 1: Add a final-model selection regression test**

```python
def test_model_uri_environment_has_priority(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "runs:/verified-run/model")
    assert exporter.resolve_model_uri() == "runs:/verified-run/model"
```

Keep the existing feature-path and lifespan tests.

- [ ] **Step 2: Pin serving compatibility**

Change `scikit-learn>=1.4` to `scikit-learn==1.5.2`. Do not alter FastAPI, Prometheus, Grafana, endpoint, or metric definitions.

- [ ] **Step 3: Run monitoring tests with explicit final model URI**

Set `MODEL_URI` to the final tuning artifact directory URI and run:

```powershell
python -m pytest -q "Monitoring dan Logging"
```

Expected: all monitoring tests pass and model loading does not emit a sklearn version mismatch warning.

- [ ] **Step 4: Smoke-test inference interfaces**

Start the exporter with `MODEL_URI` set, run `7.inference.py --n 3`, request `/health` and `/metrics`, then stop the service. Expected: three successful predictions, `model_up=true`, and Prometheus text containing the existing request and model metrics.

- [ ] **Step 5: Document deterministic serving**

Add relative PowerShell and Unix examples that set `MODEL_URI` to the selected `Membangun_model/mlruns/.../artifacts/model` directory. Replace every `D:\hackation\...` command in `PANDUAN_CAPTURE_BUKTI.md` with commands run from the aggregate repository root, while retaining genuine existing screenshots.

- [ ] **Step 6: Commit monitoring compatibility changes**

```bash
git add "Monitoring dan Logging/requirements.txt" "Monitoring dan Logging/test_exporter.py" "Monitoring dan Logging/README.md" "Monitoring dan Logging/PANDUAN_CAPTURE_BUKTI.md"
git commit -m "fix: pin model serving compatibility"
```

### Task 6: Complete The MLflow Project

**Files:**
- Create: `Workflow-CI/MLProject/test_modelling.py`
- Modify: `Workflow-CI/MLProject/modelling.py`
- Rename: `Workflow-CI/MLProject/MLProject` to `Workflow-CI/MLProject/MLproject`
- Modify: `Workflow-CI/MLProject/conda.yaml`
- Modify: `Workflow-CI/MLProject/requirements.txt`

**Interfaces:**
- Consumes: `--data_dir`, `--n_estimators`, `--max_depth`, and `--experiment_name` from `MLproject`.
- Produces: MLflow model, `estimator.html`, `confusion_matrix.png`, `classification_report.json`, and `training_outputs/run_summary.json`.

- [ ] **Step 1: Add failing MLflow Project tests**

Test that `load_data(data_dir)` rejects missing/misaligned inputs, `evaluate(model, X_test, y_test, output_dir)` writes the three evaluation artifacts, and the canonical descriptor exists:

```python
def test_mlproject_uses_canonical_filename():
    root = Path(__file__).resolve().parent
    assert (root / "MLproject").is_file()
    assert not (root / "MLProject").exists()
```

- [ ] **Step 2: Run tests and confirm failure**

From `Workflow-CI/MLProject`:

```powershell
python -m pytest -q test_modelling.py
```

Expected: tests fail because helpers/artifacts/canonical filename do not yet exist.

- [ ] **Step 3: Extract minimal load and evaluation helpers**

Keep them in `modelling.py` and add the required `json`, matplotlib, metrics, and `estimator_html_repr` imports:

```python
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed files in {data_dir}: {missing}")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train_df = pd.read_csv(data_dir / "y_train.csv")
    y_test_df = pd.read_csv(data_dir / "y_test.csv")
    if "y" not in y_train_df or "y" not in y_test_df:
        raise ValueError("Target column 'y' is missing")
    y_train, y_test = y_train_df["y"], y_test_df["y"]
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature schemas differ")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts differ")
    if X_train.empty or X_test.empty:
        raise ValueError("Processed feature data is empty")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Processed features contain null values")
    if "duration" in X_train.columns or "duration" in X_test.columns:
        raise ValueError("Leakage column 'duration' found in features")
    if not set(y_train.unique()).issubset({0, 1}) or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Targets must be binary")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    (output_dir / "estimator.html").write_text(estimator_html_repr(model), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(classification_report(y_test, predictions, output_dict=True, zero_division=0), indent=2),
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues", ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(figure)
    return metrics
```

Do not catch `mlflow.log_artifact` failures; artifact failure must fail the run. Write generated files under `training_outputs/` and log that directory once.

- [ ] **Step 4: Rename descriptor and pin dependencies**

Rename to `MLproject`, retain the existing entry point parameters, and pin `scikit-learn==1.5.2`. Keep MLflow `2.19.0` and Python `3.12`.

- [ ] **Step 5: Run MLflow Project tests**

```powershell
python -m pytest -q test_modelling.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit in the standalone Workflow-CI repository**

```bash
git add MLProject/modelling.py MLProject/test_modelling.py MLProject/MLproject MLProject/conda.yaml MLProject/requirements.txt
git commit -m "fix: produce complete retraining artifacts"
```

### Task 7: Validate Retraining Workflow Locally

**Files:**
- Modify: `Workflow-CI/.github/workflows/mlflow.yml`
- Modify: `Workflow-CI/README.md`
- Modify: `Workflow-CI/.gitignore`

**Interfaces:**
- Consumes: canonical `MLProject/MLproject` and synchronized processed dataset.
- Produces: GitHub Actions artifact `ci-training-artifacts` containing `MLProject/training_outputs/` and `MLProject/mlruns/`.

- [ ] **Step 1: Update workflow training and verification steps**

Use pip caching and an explicit experiment name on the outer run:

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: pip
    cache-dependency-path: MLProject/requirements.txt

- name: Run MLflow Project training
  env:
    MLFLOW_TRACKING_URI: ${{ github.workspace }}/MLProject/mlruns
  run: mlflow run MLProject --env-manager local --experiment-name bank-marketing-ci -P data_dir=namadataset_preprocessing
```

Verify `run_summary.json`, `estimator.html`, `confusion_matrix.png`, and an MLflow `artifacts/model` directory before upload.

- [ ] **Step 2: Add focused ignore rules**

Ignore only `.venv/`, `__pycache__/`, `.pytest_cache/`, `MLProject/mlruns/`, and generated `MLProject/training_outputs/`. Do not ignore `MLProject/namadataset_preprocessing/`.

- [ ] **Step 3: Run the MLflow Project from the Workflow-CI root**

```powershell
$env:MLFLOW_TRACKING_URI = ([System.Uri]((Resolve-Path "MLProject/mlruns").Path)).AbsoluteUri
mlflow run MLProject --env-manager local --experiment-name bank-marketing-ci -P data_dir=namadataset_preprocessing
```

Expected: FINISHED run, model artifact, all evaluation artifacts, and run summary.

- [ ] **Step 4: Validate workflow YAML and commands**

Parse and inspect `.github/workflows/mlflow.yml` with the PyYAML installed by MLflow:

```powershell
python -c "from pathlib import Path; import yaml; text=Path('.github/workflows/mlflow.yml').read_text(); yaml.safe_load(text); assert 'workflow_dispatch' in text; assert 'mlflow run MLProject' in text; assert 'actions/upload-artifact@v4' in text"
```

Then repeat the artifact verification commands locally.

- [ ] **Step 5: Update Workflow-CI README**

Document setup, local `mlflow run`, output locations, workflow triggers, artifact name, and manual screenshot requirements. Use `MLproject` casing consistently.

- [ ] **Step 6: Commit CI workflow changes in the standalone repository**

```bash
git add .github/workflows/mlflow.yml .gitignore README.md
git commit -m "ci: retrain and upload MLflow artifacts"
```

### Task 8: Publish Standalone Repositories And Finalize Submission

**Files:**
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto/.gitignore`
- Modify: `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto.txt`
- Modify: `Workflow-CI.txt`
- Modify or remove draft status from: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: locally verified K1/K3 directories and authenticated GitHub CLI.
- Produces: two public GitHub repository URLs, successful Actions runs, real artifacts, and final aggregate links.

- [ ] **Step 1: Verify GitHub authentication and repository availability**

```powershell
gh auth status
gh repo view Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto
gh repo view Muhayustrid/Workflow-CI
```

Expected: authentication succeeds; create repositories only if `gh repo view` confirms they do not exist. Never overwrite an existing remote without user approval.

- [ ] **Step 2: Initialize and publish the experiment repository**

From the experiment directory, initialize Git if needed, add only rubric files, commit, then run:

```powershell
gh repo create Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto --public --source . --remote origin --push
```

Verify `.github/workflows/preprocessing.yml` appears at the public repository root.

- [ ] **Step 3: Initialize and publish Workflow-CI**

From `Workflow-CI`, initialize Git if needed, add only rubric files, commit, then run:

```powershell
gh repo create Muhayustrid/Workflow-CI --public --source . --remote origin --push
```

Verify `.github/workflows/mlflow.yml` appears at the public repository root.

- [ ] **Step 4: Trigger and inspect real workflows**

```powershell
gh workflow run preprocessing.yml --repo Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto
gh workflow run mlflow.yml --repo Muhayustrid/Workflow-CI
gh run list --repo Muhayustrid/Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto
gh run list --repo Muhayustrid/Workflow-CI
```

Wait for both runs, inspect logs with `gh run view --log`, and verify uploaded artifact names. Do not proceed if either run fails.

- [ ] **Step 5: Write real public URLs into aggregate link files**

Get URLs with `gh repo view --json url --jq .url`. Write exactly one URL and newline to each corresponding `.txt` file; no explanation or placeholder text.

- [ ] **Step 6: Capture genuine manual evidence**

The user opens the real MLflow UI and GitHub Actions pages and captures:

- `Membangun_model/screenshoot_dashboard.png`: experiment name, baseline/tuning rows, parameters, and metrics.
- `Membangun_model/screenshoot_artifak.png`: selected run, `model/`, `estimator.html`, and evaluation artifacts.
- `Workflow-CI/Workflow Artifact.png`: successful workflow and uploaded `ci-training-artifacts`.

Do not generate these images programmatically.

- [ ] **Step 7: Add aggregate ignore rules and finish documentation**

Ignore `.venv/`, `__pycache__/`, `.pytest_cache/`, and local `mlruns/` directories. Keep datasets, committed tuning artifacts, and genuine screenshots. Remove the root `DRAFT` label and describe only final runnable commands and repository links.

- [ ] **Step 8: Run final verification**

Run all local test suites, preprocessing, baseline/tuning, monitoring smoke test, and MLflow Project. Then run:

```powershell
git status --short
git grep -n -E "PENDING_PUBLIC_URL|D:\\\\|github_pat_|ghp_|DOCKERHUB_TOKEN" -- "*.py" "*.yml" "*.yaml" "*.txt" "*.ipynb" "*.md" ":!docs/superpowers/**"
```

Expected: no placeholder URL, hard-coded workstation path in runnable files, or credential; only intended changes remain.

- [ ] **Step 9: Commit and push the aggregate package**

Inspect `git status`, `git diff`, and recent log; stage only intended K2/K4/link/docs/evidence files. Commit without adding nested standalone repositories as gitlinks, then push the aggregate repository.

- [ ] **Step 10: Produce the final checklist report**

Mark each item `YA` only when directly verified. Report screenshots or external workflow state as `TIDAK` if manual evidence or GitHub execution is still unavailable. Include final metrics and exact public URLs from the verified runs.
