import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import modelling
import modelling_tuning


DATA_DIR = ROOT / "namadataset_preprocessing"


def copy_data(destination: Path) -> None:
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
        shutil.copy(DATA_DIR / name, destination / name)


def test_data_contract():
    X_train, X_test, y_train, y_test = modelling.load_data(DATA_DIR)
    assert X_train.shape[1] == X_test.shape[1] == 50
    assert "duration" not in X_train.columns
    assert set(y_train.unique()) <= {0, 1}


def test_data_contract_rejects_mismatched_rows(tmp_path: Path):
    copy_data(tmp_path)
    y_train = pd.read_csv(tmp_path / "y_train.csv").iloc[:-1]
    y_train.to_csv(tmp_path / "y_train.csv", index=False)

    with pytest.raises(ValueError, match="row count"):
        modelling.load_data(tmp_path)


def test_data_contract_rejects_mismatched_feature_schemas(tmp_path: Path):
    copy_data(tmp_path)
    X_test = pd.read_csv(tmp_path / "X_test.csv").rename(columns={"age": "wrong_feature"})
    X_test.to_csv(tmp_path / "X_test.csv", index=False)

    with pytest.raises(ValueError, match="schemas differ"):
        modelling.load_data(tmp_path)


def test_data_contract_rejects_empty_features(tmp_path: Path):
    copy_data(tmp_path)
    pd.read_csv(tmp_path / "X_train.csv").iloc[:0].to_csv(tmp_path / "X_train.csv", index=False)
    pd.read_csv(tmp_path / "y_train.csv").iloc[:0].to_csv(tmp_path / "y_train.csv", index=False)

    with pytest.raises(ValueError, match="feature data is empty"):
        modelling.load_data(tmp_path)


def test_data_contract_rejects_null_features(tmp_path: Path):
    copy_data(tmp_path)
    X_train = pd.read_csv(tmp_path / "X_train.csv")
    X_train.iloc[0, 0] = None
    X_train.to_csv(tmp_path / "X_train.csv", index=False)

    with pytest.raises(ValueError, match="null values"):
        modelling.load_data(tmp_path)


def test_data_contract_rejects_nonbinary_targets(tmp_path: Path):
    copy_data(tmp_path)
    y_test = pd.read_csv(tmp_path / "y_test.csv")
    y_test.iloc[0, 0] = 2
    y_test.to_csv(tmp_path / "y_test.csv", index=False)

    with pytest.raises(ValueError, match="binary"):
        modelling.load_data(tmp_path)


def test_data_contract_rejects_missing_target_column(tmp_path: Path):
    copy_data(tmp_path)
    y_train = pd.read_csv(tmp_path / "y_train.csv").rename(columns={"y": "target"})
    y_train.to_csv(tmp_path / "y_train.csv", index=False)

    with pytest.raises(ValueError, match="Target column"):
        modelling.load_data(tmp_path)


def test_experiment_defaults_match():
    assert modelling.parse_args([]).experiment_name == "bank-marketing-model-development"
    assert modelling_tuning.parse_args([]).experiment_name == "bank-marketing-model-development"


def test_baseline_model_contract():
    model = modelling.build_model()
    assert model.__class__.__name__ == "RandomForestClassifier"
    assert model.random_state == 42


def test_tuning_grid_contract():
    assert modelling_tuning.PARAM_GRID == {
        "n_estimators": [100, 200],
        "max_depth": [None, 12],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"],
    }


@pytest.mark.filterwarnings("ignore:datetime.datetime.utcnow.*:DeprecationWarning")
def test_manual_tuning_writes_estimator_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X_train = pd.DataFrame({"age": [float(value) for value in range(12)], "balance": [0.0, 1.0] * 6})
    X_test = pd.DataFrame({"age": [float(value) for value in range(12, 16)], "balance": [0.0, 1.0] * 2})
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    pd.DataFrame({"y": [0, 1] * 6}).to_csv(data_dir / "y_train.csv", index=False)
    pd.DataFrame({"y": [0, 1] * 2}).to_csv(data_dir / "y_test.csv", index=False)
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelling_tuning.py",
            "--data-dir",
            str(data_dir),
            "--tracking-uri",
            (tmp_path / "mlruns").as_uri(),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )
    monkeypatch.setattr(
        modelling_tuning,
        "PARAM_GRID",
        {"n_estimators": [1], "max_depth": [2], "min_samples_leaf": [1], "class_weight": ["balanced"]},
    )

    assert modelling_tuning.main() == 0
    assert "RandomForestClassifier" in (artifact_dir / "estimator.html").read_text(encoding="utf-8")


def test_training_uses_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"
