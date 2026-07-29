# Easy Screenshot Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one Indonesian copy-paste guide for capturing the three remaining genuine screenshots from the current submission root.

**Architecture:** A single root Markdown file uses runtime-resolved PowerShell paths and an isolated `.venv_screenshot`. It runs only K2 baseline/tuning and MLflow UI, then links to the already successful GitHub Actions run for the third screenshot.

**Tech Stack:** PowerShell 5.1, Python 3.12, MLflow 2.19.0, scikit-learn 1.5.2.

## Global Constraints

- Run every command from the current submission root.
- Do not clone repositories or require manual directory navigation.
- Install only `Membangun_model/requirements.txt` into `.venv_screenshot`.
- Do not hardcode an MLflow experiment ID or run ID.
- Do not generate, synthesize, or reuse screenshots.
- Required outputs are `Membangun_model/screenshoot_dashboard.png`, `Membangun_model/screenshoot_artifak.png`, and `Workflow-CI/Workflow Artifact.png`.

---

### Task 1: Add Copy-Paste Screenshot Guide

**Files:**
- Create: `PANDUAN_SCREENSHOT_MUDAH.md`

**Interfaces:**
- Consumes: current root folders `Membangun_model/` and `Workflow-CI/`, Python 3.12, and the verified Actions run URL.
- Produces: root-relative commands that start MLflow UI and exact instructions for three genuine screenshots.

- [ ] **Step 1: Run a failing file-presence check**

```powershell
if (-not (Test-Path -LiteralPath "PANDUAN_SCREENSHOT_MUDAH.md")) {
  throw "PANDUAN_SCREENSHOT_MUDAH.md belum tersedia"
}
```

Expected: failure because the file does not exist.

- [ ] **Step 2: Create the minimal guide**

The guide must contain these four sections in this order:

```markdown
# Panduan Screenshot Mudah

## 1. Setup Sekali
## 2. Jalankan Training dan MLflow UI
## 3. Ambil Tiga Screenshot
## Jika Ada Error
```

The setup block must:

```powershell
$root = (Resolve-Path ".").Path
if (-not (Test-Path (Join-Path $root "Membangun_model\requirements.txt"))) { throw "Jalankan dari root submission" }
if (-not (Test-Path (Join-Path $root "Workflow-CI"))) { throw "Folder Workflow-CI tidak ditemukan" }
if (-not (Test-Path (Join-Path $root ".venv_screenshot\Scripts\python.exe"))) {
  py -3.12 -m venv (Join-Path $root ".venv_screenshot")
}
$python = (Resolve-Path (Join-Path $root ".venv_screenshot\Scripts\python.exe")).Path
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $root "Membangun_model\requirements.txt")
```

The training block must use `Push-Location`/`Pop-Location` internally, run K2 tests, baseline, tuning, and finally start:

```powershell
& $python -m mlflow ui --backend-store-uri $trackingDir --host 127.0.0.1 --port 5000
```

The capture section must name the experiment `bank-marketing-model-development`, describe the dashboard columns, identify the newest manual tuning run, require `model/` and `estimator.html`, and link to:

```text
https://github.com/Muhayustrid/Workflow-CI/actions/runs/30405176099
```

- [ ] **Step 3: Run focused static validation**

```powershell
$guide = Get-Content -LiteralPath "PANDUAN_SCREENSHOT_MUDAH.md" -Raw
$required = @(
  ".venv_screenshot",
  "Membangun_model/requirements.txt",
  "bank-marketing-model-development",
  "http://127.0.0.1:5000",
  "Membangun_model/screenshoot_dashboard.png",
  "Membangun_model/screenshoot_artifak.png",
  "Workflow-CI/Workflow Artifact.png",
  "https://github.com/Muhayustrid/Workflow-CI/actions/runs/30405176099",
  "model/",
  "estimator.html"
)
foreach ($value in $required) {
  if (-not $guide.Contains($value)) { throw "Panduan tidak memuat: $value" }
}
if ($guide -match "(?i)\b[A-Z]:\\") { throw "Ditemukan path drive absolut" }
if ($guide -match "(?i)\b[a-f0-9]{32}\b") { throw "Ditemukan run ID MLflow tetap" }
"STATIC_CHECK_OK"
```

Expected: `STATIC_CHECK_OK`.

- [ ] **Step 4: Parse every PowerShell block**

Extract every fenced `powershell` block and pass it to `[System.Management.Automation.Language.Parser]::ParseInput`. Fail if any block returns parser errors.

Expected: all blocks parse without errors.

- [ ] **Step 5: Commit and push the PR branch**

```bash
git add PANDUAN_SCREENSHOT_MUDAH.md docs/superpowers/plans/2026-07-29-easy-screenshot-guide.md
git commit -m "docs: add easy screenshot tutorial"
git push origin repair/submission-reviewer
```
