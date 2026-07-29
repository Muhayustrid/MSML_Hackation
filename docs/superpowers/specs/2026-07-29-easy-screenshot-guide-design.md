# Easy Screenshot Guide Design

## Goal

Add one concise Indonesian guide that can be copied into PowerShell while the
terminal is already at the submission root. The guide covers only the three
remaining genuine screenshots.

## File

Create `PANDUAN_SCREENSHOT_MUDAH.md` at the aggregate repository root. Keep the
existing full guide unchanged and link to it only for troubleshooting.

## Flow

1. Verify the terminal is at the submission root by checking the K2 and K3
   directories.
2. Create an isolated `.venv_screenshot` with Python 3.12 and install only
   `Membangun_model/requirements.txt`.
3. Run the K2 tests, baseline, and tuning from root-resolved paths.
4. Start MLflow UI on `127.0.0.1:5000` and leave that terminal open.
5. Explain the exact MLflow dashboard and artifact UI elements to capture and
   the required output filenames.
6. Link directly to the already successful Workflow-CI Actions run and explain
   the UI elements and output filename for the third screenshot.

## Constraints

- Every command starts from the current submission root; no clone or manual
  directory navigation is required.
- Use runtime-resolved relative paths and no workstation-specific path.
- Use scikit-learn 1.5.2 through the committed K2 requirements.
- Do not hardcode an MLflow experiment ID or run ID.
- Do not generate, synthesize, or reuse screenshots.
- Keep setup and capture instructions separate so users do not copy a blocking
  MLflow UI command together with later commands accidentally.

## Verification

Statically verify that the guide contains the three exact screenshot paths,
the MLflow UI and GitHub Actions URLs, the experiment name, `model/`,
`estimator.html`, and no absolute drive path, placeholder, or fixed MLflow run
ID. Parse every PowerShell fenced block before committing.
