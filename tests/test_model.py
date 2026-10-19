"""Smoke test: the saved lite model loads and predicts on a correctly shaped vector."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FULL, LITE

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_serving_model_loads_and_predicts():
    serving = json.loads((MODELS_DIR / "serving.json").read_text())
    assert serving["features"] == LITE
    assert set(serving["fill_values"]) == set(LITE)
    assert all(v in (-1, 0, 1) for v in serving["fill_values"].values())

    model = joblib.load(MODELS_DIR / serving["model_file"])
    rows = pd.DataFrame([[1] * len(LITE), [-1] * len(LITE)], columns=LITE)
    proba = model.predict_proba(rows)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    # All-legitimate-like features should look less phishy than all-phishing-like.
    assert proba[0, 1] < proba[1, 1]


def test_all_saved_models_load():
    for fs, features in (("full", FULL), ("lite", LITE)):
        row = pd.DataFrame([[1] * len(features)], columns=features)
        for name in ("decision_tree", "random_forest", "svm", "xgboost"):
            model = joblib.load(MODELS_DIR / f"{fs}_{name}.joblib")
            assert model.predict_proba(row).shape == (1, 2)
