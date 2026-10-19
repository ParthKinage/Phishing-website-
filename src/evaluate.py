"""Evaluate the saved models once on the held-out test set.

Also runs the duplicate-sensitivity check: each model's tuned configuration
is refit on a de-duplicated train split and scored on a de-duplicated test
split, so no identical row can sit on both sides.

Run (after src.train): python -m src.evaluate
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src import data
from src.plotstyle import MODEL_COLORS, MODEL_LABELS, SEQUENTIAL, TEXT_2, plt, save
from src.train import FEATURE_SETS, MODELS_DIR, RESULTS_DIR

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

MODEL_NAMES = list(MODEL_COLORS)


def scores(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, proba, pred


def to_markdown(df, floatfmt="{:.4f}"):
    """Minimal DataFrame -> GitHub Markdown table (avoids a tabulate dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def plot_confusion_matrices(cms, fs_name):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.sum() * 0.6)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.sum() * 0.3 else "#0b0b0b")
        ax.set_xticks([0, 1], ["legit", "phishing"])
        ax.set_yticks([0, 1], ["legit", "phishing"] if ax is axes[0] else [])
        ax.set_xlabel("Predicted")
        ax.set_title(MODEL_LABELS[name], fontsize=11)
    axes[0].set_ylabel("Actual")
    fig.suptitle(f"Confusion matrices, test set ({fs_name} features)", fontweight="bold")
    save(fig, RESULTS_DIR / f"confusion_{fs_name}.png")


def plot_roc(curves, fs_name):
    """Full ROC curve plus a zoom on the top-left corner, where the models differ."""
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls="--", label="Chance")
    for name, (fpr, tpr, auc) in curves.items():
        for a in (ax, zoom):
            a.plot(fpr, tpr, color=MODEL_COLORS[name],
                   label=f"{MODEL_LABELS[name]} (AUC {auc:.4f})")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.set_title("Full range")
    ax.legend(loc="lower right")
    zoom.set_xlim(0, 0.2)
    zoom.set_ylim(0.8, 1.0)
    zoom.set_title("Zoom: FPR 0-0.2, TPR 0.8-1")
    for a in (ax, zoom):
        a.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (phishing recall)")
    fig.suptitle(f"ROC curves, test set ({fs_name} features)", fontweight="bold")
    save(fig, RESULTS_DIR / f"roc_{fs_name}.png")


def evaluate_test_set():
    cv = pd.DataFrame(json.loads((RESULTS_DIR / "cv_results.json").read_text()))
    X, y = data.load()
    _, X_te, _, y_te = data.split(X, y)
    tables = []
    for fs_name, features in FEATURE_SETS.items():
        rows, cms, curves = [], {}, {}
        for name in MODEL_NAMES:
            model = joblib.load(MODELS_DIR / f"{fs_name}_{name}.joblib")
            m, proba, pred = scores(model, X_te[features], y_te)
            c = cv[(cv.feature_set == fs_name) & (cv.model == name)].iloc[0]
            rows.append({
                "feature_set": fs_name,
                "model": name,
                **m,
                "cv_f1_mean": c.cv_f1_mean,
                "cv_f1_std": c.cv_f1_std,
                "train_time_s": c.train_time_s,
                "inference_ms": c.inference_ms,
            })
            cms[name] = confusion_matrix(y_te, pred, labels=[0, 1])
            fpr, tpr, _ = roc_curve(y_te, proba)
            curves[name] = (fpr, tpr, m["roc_auc"])
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"metrics_{fs_name}.csv", index=False)
        md = df.drop(columns="feature_set").assign(model=df.model.map(MODEL_LABELS))
        md.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                      "CV F1 mean", "CV F1 std", "Train time (s)", "Inference (ms)"]
        (RESULTS_DIR / f"metrics_{fs_name}.md").write_text(to_markdown(md))
        cm_rows = [{"model": n, "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]), "tp": int(cm[1, 1])} for n, cm in cms.items()]
        pd.DataFrame(cm_rows).to_csv(RESULTS_DIR / f"confusion_{fs_name}.csv", index=False)
        plot_confusion_matrices(cms, fs_name)
        plot_roc(curves, fs_name)
        print(f"Test-set results ({fs_name}):\n{to_markdown(md)}")
        tables.append(df)
    return pd.concat(tables)


def dedup_sensitivity(headline):
    """Refit each tuned configuration on de-duplicated data and compare.

    Duplicates are removed within each feature set's own columns, because
    dropping features creates more identical rows (lite has far more)."""
    X, y = data.load()
    rows = []
    for fs_name, features in FEATURE_SETS.items():
        X_tr, X_te, y_tr, y_te = data.split(X[features], y, dedup=True)
        for name in MODEL_NAMES:
            model = clone(joblib.load(MODELS_DIR / f"{fs_name}_{name}.joblib"))
            model.fit(X_tr, y_tr)
            m, _, _ = scores(model, X_te, y_te)
            h = headline[(headline.feature_set == fs_name) & (headline.model == name)].iloc[0]
            rows.append({
                "feature_set": fs_name,
                "model": name,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "accuracy_dedup": m["accuracy"],
                "f1_dedup": m["f1"],
                "roc_auc_dedup": m["roc_auc"],
                "accuracy_standard": h.accuracy,
                "f1_standard": h.f1,
                "f1_change": m["f1"] - h.f1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dedup_sensitivity.csv", index=False)
    md = df.assign(model=df.model.map(MODEL_LABELS))
    md.columns = ["Features", "Model", "Train rows", "Test rows", "Accuracy (dedup)",
                  "F1 (dedup)", "ROC-AUC (dedup)", "Accuracy (standard)",
                  "F1 (standard)", "F1 change"]
    (RESULTS_DIR / "dedup_sensitivity.md").write_text(to_markdown(md))
    print(f"Duplicate sensitivity:\n{to_markdown(md)}")
    return df


def main():
    headline = evaluate_test_set()
    dedup_sensitivity(headline)


if __name__ == "__main__":
    main()
