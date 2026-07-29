# Submission Reviewer Repair Design

## Goal

Repair the submission for Muhammad Yusuf Tri Daryanto so that the experiment,
model development, and automated retraining criteria are reviewable and
runnable. Preserve the existing monitoring and logging behavior.

Target levels:

- Criterion 1: Advanced
- Criterion 2: retain the existing baseline and tuning coverage
- Criterion 3: Skilled
- Docker Hub: out of scope

## Root Cause

The experiment and Workflow-CI directories exist locally but are untracked by
the aggregate repository. The committed submission contains only placeholder
link files, while GitHub Actions cannot discover workflows nested inside an
untracked subdirectory. The reviewer therefore could not access Criterion 1 or
Criterion 3 even though most of their implementation exists locally.

## Repository Topology

Use three public-facing repositories:

1. The current aggregate submission repository contains the final submission
   package, modelling evidence, monitoring evidence, and two link files.
2. `Eksperimen_SML_Muhammad_Yusuf_Tri_Daryanto` is published as a standalone
   repository with its workflow at `.github/workflows/preprocessing.yml`.
3. `Workflow-CI` is published as a standalone repository with its workflow at
   `.github/workflows/mlflow.yml`.

The two link files in the aggregate package contain only the corresponding
public repository URLs.

## Data Flow

`bank-full.csv` is processed by the notebook and shared preprocessing module.
The automated CLI produces deterministic train/test CSV files, feature names,
metadata, and a fitted preprocessor. Identical copies of those outputs are used
by model development and the MLflow Project.

The leakage feature `duration` is removed before duplicate detection. Duplicate
deployable feature vectors are removed before the stratified split. The fitted
preprocessor only sees training data, and the test set remains untouched until
evaluation.

## Experiment Repository

Keep the existing raw dataset, executed notebook, preprocessing module, CLI,
tests, and generated artifacts. Make only the following corrections:

- Resolve default CLI paths from the standalone repository root.
- Validate test size, non-empty input, required dtypes, target classes, and
  sufficient class samples.
- Drop `duration` before deduplicating deployable feature rows.
- Assert there are no identical deployable feature vectors across train/test.
- Regenerate the processed data and execute the notebook from start to finish.
- Keep saved notebook outputs and remove local absolute paths from displayed
  output where practical.

The preprocessing workflow runs tests and the CLI, validates all seven output
files, and uploads the processed dataset as an Actions artifact.

## Model Development

Keep Random Forest as the model family. Use one clear MLflow experiment with at
least a baseline run and a tuning run so the UI can compare runs directly.

Each relevant run records:

- deterministic parameters and random state;
- accuracy, precision, recall, F1, and ROC-AUC;
- the MLflow model with signature and input example;
- `estimator.html`;
- confusion matrix and other relevant evaluation artifacts.

Strengthen the data contract to check required files, target columns, row
alignment, matching feature schemas, null values, binary labels, and absence of
`duration`. Regenerate runs in the final location and update documentation so
it does not reference stale run IDs.

## Monitoring Compatibility

Do not rename or remove the existing endpoints, ports, Prometheus metric names,
Grafana dashboard, or alert rules. Match the scikit-learn version used for
training and serving. Select the final model through the existing `MODEL_URI`
interface instead of relying on filesystem modification time. Run the existing
monitoring tests and smoke-test `/health`, `/predict`, and `/metrics` against
the final model.

## MLflow Project And CI

Use the conventional filename `MLproject`. The project entry point validates
the preprocessed dataset, trains the model, logs parameters and metrics, logs a
model, and writes evaluation artifacts and a run summary. Any training or
artifact failure exits nonzero.

The GitHub Actions workflow:

1. checks out the standalone repository;
2. installs pinned compatible dependencies;
3. validates required processed data;
4. invokes training with `mlflow run MLProject`, not direct Python execution;
5. verifies the model and evaluation artifacts;
6. uploads the MLflow store and training outputs with `upload-artifact`.

The workflow supports pushes to the primary branch and `workflow_dispatch`.
The MLflow Project is verified locally before publication.

## Evidence And Documentation

Only genuine UI screenshots are accepted. After local runs, the user captures
the MLflow experiment table and selected run artifact tree. After publication,
the GitHub Actions UI must show a successful retraining run and uploaded
artifact before its screenshot is added.

READMEs contain only actionable setup, execution, UI, CI, artifact, and
screenshot instructions. Placeholder URLs, draft labels, stale run IDs, and
absolute workstation paths are removed. Git ignore rules exclude environments,
caches, and generated local tracking stores without excluding rubric-required
datasets or evidence.

## Verification Gates

Work proceeds sequentially:

1. Experiment tests, CLI preprocessing, artifact validation, and notebook
   execution must pass.
2. Baseline and tuning must complete with visible MLflow parameters, metrics,
   model, estimator HTML, and evaluation artifacts.
3. Monitoring tests and inference smoke tests must pass with the final model.
4. `mlflow run` must complete locally and produce the CI artifact tree.
5. Both standalone repositories must be public, workflows must complete on
   GitHub, and link files must contain real URLs.
6. The final checklist is answered honestly; external screenshots or workflow
   results remain incomplete until directly verified.

## Non-Goals

- No new dataset or model problem type.
- No Docker Hub integration.
- No monitoring redesign.
- No fabricated screenshots or external success claims.
- No speculative framework or abstraction work.
