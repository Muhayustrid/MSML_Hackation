import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import modelling
import modelling_tuning


DATA_DIR = ROOT / "namadataset_preprocessing"


def test_data_contract():
    X_train, X_test, y_train, y_test = modelling.load_data(DATA_DIR)
    assert X_train.shape[1] == X_test.shape[1] == 50
    assert "duration" not in X_train.columns
    assert set(y_train.unique()) <= {0, 1}


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


def test_training_uses_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"
