"""Tune, cross-validate and fit all four models on both feature sets.

Only the training split is used here. The test split is used once, by
`src/evaluate.py`.

Run: python -m src.train
"""
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src import data
from src.features import FULL, LITE

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = data.SEED
FEATURE_SETS = {"full": FULL, "lite": LITE}
# The spec asks for SVC(probability=True) for ROC curves; scikit-learn 1.9
# deprecates the flag (removal planned for 1.11) but it works in the pinned version.
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

# Parallel workers for the grid search. Capped because every worker is a
# separate process with its own copy of the libraries; on many-core laptops
# n_jobs=-1 can run out of memory.
N_JOBS = min(8, os.cpu_count() or 1)
# Two lite models count as tied on CV F1 if they differ by less than this.
F1_TIE_TOLERANCE = 0.001


def model_grids():
    """Estimator and (deliberately small) hyper-parameter grid per model."""
    return {
        "decision_tree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "svm": (
            Pipeline([
                ("scale", StandardScaler()),
                ("svc", SVC(kernel="rbf", probability=True, random_state=SEED)),
            ]),
            {
                "svc__C": [1, 10, 100],
                "svc__gamma": ["scale", 0.1, 0.01],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=SEED, n_jobs=N_JOBS, tree_method="hist",
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
            },
        ),
    }


def single_prediction_ms(model, row, repeats=200):
    """Median wall time (ms) of predict_proba on one row (a 1-row DataFrame)."""
    model.predict_proba(row)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def tune_feature_set(fs_name, X_tr, y_tr):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    records = []
    for name, (estimator, grid) in model_grids().items():
        # XGBoost parallelises with its own threads; running it inside joblib
        # worker processes crashed on Windows, so its search runs in-process.
        search = GridSearchCV(
            estimator, grid, scoring="f1", cv=cv,
            n_jobs=1 if name == "xgboost" else N_JOBS, refit=True,
            return_train_score=False,
        )
        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        search_s = time.perf_counter() - t0
        i = search.best_index_
        best = search.best_estimator_
        if name == "xgboost":
            # One thread for serving: a single-row prediction gains nothing
            # from threads, and timings stay comparable with the other models.
            best.set_params(n_jobs=1)
        rec = {
            "feature_set": fs_name,
            "model": name,
            "best_params": search.best_params_,
            "cv_f1_mean": float(search.cv_results_["mean_test_score"][i]),
            "cv_f1_std": float(search.cv_results_["std_test_score"][i]),
            "train_time_s": float(search.refit_time_),
            "search_time_s": search_s,
            "inference_ms": single_prediction_ms(best, X_tr.iloc[[0]]),
            "n_configs": len(search.cv_results_["params"]),
        }
        records.append(rec)
        joblib.dump(best, MODELS_DIR / f"{fs_name}_{name}.joblib")
        print(f"  [{fs_name}] {name:14s} CV F1 {rec['cv_f1_mean']:.4f} +/- {rec['cv_f1_std']:.4f}  "
              f"fit {rec['train_time_s']:.2f}s  search {search_s:.1f}s  "
              f"predict {rec['inference_ms']:.2f}ms  best {search.best_params_}")
    return records


def choose_serving_model(lite_records, X_tr):
    """Best lite model by CV F1; models within F1_TIE_TOLERANCE of the best are
    tied, and the tie is broken by single-prediction inference time."""
    best_f1 = max(r["cv_f1_mean"] for r in lite_records)
    tied = [r for r in lite_records if best_f1 - r["cv_f1_mean"] < F1_TIE_TOLERANCE]
    chosen = min(tied, key=lambda r: r["inference_ms"])
    if len(tied) > 1:
        others = ", ".join(f"{r['model']} ({r['cv_f1_mean']:.4f}, {r['inference_ms']:.2f} ms)"
                           for r in tied if r is not chosen)
        reason = (f"Highest 5-fold CV F1 on the lite feature set, tied (within "
                  f"{F1_TIE_TOLERANCE}) with {others}; chosen for the fastest "
                  f"single prediction ({chosen['inference_ms']:.2f} ms).")
    else:
        runner_up = sorted(lite_records, key=lambda r: -r["cv_f1_mean"])[1]
        reason = (f"Highest 5-fold CV F1 on the lite feature set "
                  f"({chosen['cv_f1_mean']:.4f} vs. runner-up {runner_up['model']} "
                  f"{runner_up['cv_f1_mean']:.4f}); no tie within {F1_TIE_TOLERANCE}.")
    # Fill value for page features the extractor could not compute: the most
    # common value of that feature in the training data (see README / report).
    fill = {f: int(X_tr[f].mode().iloc[0]) for f in LITE}
    serving = {
        "model": chosen["model"],
        "model_file": f"lite_{chosen['model']}.joblib",
        "features": LITE,
        "cv_f1_mean": chosen["cv_f1_mean"],
        "inference_ms": chosen["inference_ms"],
        "reason": reason,
        "fill_values": fill,
        "threshold": 0.5,
    }
    if chosen["model"] == "xgboost":
        # XGBoost's own JSON format: lets the API serve the model with only
        # xgboost + numpy installed (identical predictions and SHAP values).
        model = joblib.load(MODELS_DIR / serving["model_file"])
        model.get_booster().save_model(MODELS_DIR / "lite_xgboost.json")
        serving["native_model_file"] = "lite_xgboost.json"
    (MODELS_DIR / "serving.json").write_text(json.dumps(serving, indent=2))
    print(f"Serving model: {chosen['model']} -- {reason}")
    return serving


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y = data.load()
    X_tr, _, y_tr, _ = data.split(X, y)
    all_records = []
    for fs_name, features in FEATURE_SETS.items():
        print(f"Tuning on the {fs_name} feature set ({len(features)} features)")
        all_records += tune_feature_set(fs_name, X_tr[features], y_tr)
    (RESULTS_DIR / "cv_results.json").write_text(json.dumps(all_records, indent=2))
    pd.DataFrame(all_records).drop(columns="best_params").to_csv(
        RESULTS_DIR / "cv_results.csv", index=False)
    choose_serving_model([r for r in all_records if r["feature_set"] == "lite"], X_tr)


if __name__ == "__main__":
    main()
