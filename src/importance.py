"""Feature-importance analysis (the novelty section of the report).

1. Built-in importances: Decision Tree, Random Forest (impurity decrease),
   XGBoost (gain).
2. Permutation importance on the test set for all four models (F1 drop).
3. SHAP TreeExplainer for the best tree model: summary (beeswarm) and bar plots.
4. Agreement between methods: Spearman rank correlation and top-10 overlap.
5. Ablation: refit the best model on its top-k features, k = 5..30.
6. Full-vs-lite gap: which excluded features recover the lost CV F1.

All on the full (30-feature) set. Run (after src.train): python -m src.importance
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import data
from src.evaluate import MODEL_NAMES, to_markdown
from src.features import EXCLUDED_FROM_LITE, FULL, LITE
from src.plotstyle import (
    ACCENT, MODEL_COLORS, MODEL_LABELS, SECOND, TEXT_2, plt, save,
)
from src.train import MODELS_DIR, N_JOBS, RESULTS_DIR, SEED

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

TREE_MODELS = ["decision_tree", "random_forest", "xgboost"]
KS = [5, 10, 15, 20, 30]
# Exact TreeSHAP on a 200-tree, unlimited-depth forest takes ~75 ms per row,
# so SHAP is computed on random samples: test rows for the plots, train rows
# for the feature ranking used by the ablation.
SHAP_SAMPLE = 500


def load_models():
    return {n: joblib.load(MODELS_DIR / f"full_{n}.joblib") for n in MODEL_NAMES}


def best_tree_model():
    cv = pd.DataFrame(json.loads((RESULTS_DIR / "cv_results.json").read_text()))
    cv = cv[(cv.feature_set == "full") & cv.model.isin(TREE_MODELS)]
    return cv.sort_values("cv_f1_mean", ascending=False).iloc[0].model


def shap_values(model, X):
    """SHAP values for the phishing class, shape (n_rows, n_features)."""
    explanation = shap.TreeExplainer(model)(X)
    values = explanation.values
    if values.ndim == 3:  # scikit-learn forests/trees: one column per class
        values = values[:, :, 1]
    return explanation, values


def builtin_importance(models):
    imp = pd.DataFrame({n: models[n].feature_importances_ for n in TREE_MODELS}, index=FULL)
    imp = imp / imp.sum()  # normalise so the three are comparable
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    order = imp.mean(axis=1).sort_values().index
    for ax, n in zip(axes, TREE_MODELS):
        ax.barh(order, imp.loc[order, n], color=MODEL_COLORS[n], height=0.7)
        ax.set_title(MODEL_LABELS[n])
        ax.set_xlabel("Normalised importance")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Built-in feature importance (full feature set)", fontweight="bold")
    save(fig, RESULTS_DIR / "importance_builtin.png")
    return imp


def perm_importance(models, X_te, y_te):
    rows, stds = {}, {}
    for n, model in models.items():
        # XGBoost inside joblib worker processes crashed on Windows (see train.py)
        r = permutation_importance(model, X_te, y_te, scoring="f1", n_repeats=10,
                                   random_state=SEED, n_jobs=1 if n == "xgboost" else N_JOBS)
        rows[n], stds[n] = r.importances_mean, r.importances_std
        print(f"  permutation importance done: {n}")
    imp = pd.DataFrame(rows, index=FULL)
    std = pd.DataFrame(stds, index=FULL)
    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    order = imp.mean(axis=1).sort_values().index
    for ax, n in zip(axes, MODEL_NAMES):
        ax.barh(order, imp.loc[order, n], xerr=std.loc[order, n], color=MODEL_COLORS[n],
                height=0.7, error_kw={"ecolor": TEXT_2, "lw": 1})
        ax.set_title(MODEL_LABELS[n])
        ax.set_xlabel("Drop in test F1 when shuffled")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Permutation importance on the test set (10 repeats, mean +/- std)",
                 fontweight="bold")
    save(fig, RESULTS_DIR / "importance_permutation.png")
    pd.concat({"mean": imp, "std": std}, axis=1).to_csv(RESULTS_DIR / "importance_permutation.csv")
    return imp


def shap_analysis(model, name, X_te):
    explanation, values = shap_values(model, X_te)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=FULL)
    shap.summary_plot(values, X_te, show=False, max_display=20)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary, {MODEL_LABELS[name]}, {len(X_te)} test rows", fontweight="bold")
    save(fig, RESULTS_DIR / "shap_summary.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    top = mean_abs.sort_values().tail(20)
    ax.barh(top.index, top.values, color=MODEL_COLORS[name], height=0.7)
    ax.set_xlabel("Mean |SHAP value| (impact on phishing score)")
    ax.set_title(f"SHAP feature importance, {MODEL_LABELS[name]}")
    ax.grid(axis="y", visible=False)
    save(fig, RESULTS_DIR / "shap_bar.png")
    return mean_abs


def agreement(methods):
    """methods: DataFrame, one column per method, higher = more important."""
    ranks = methods.rank(ascending=False)
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("mean_rank")
    ranks.to_csv(RESULTS_DIR / "importance_ranks.csv")
    corr = pd.DataFrame(spearmanr(methods).statistic, index=methods.columns, columns=methods.columns)
    corr.to_csv(RESULTS_DIR / "importance_spearman.csv")
    consensus_top10 = set(ranks.index[:10])
    overlap = pd.DataFrame({
        "method": methods.columns,
        "top5": [", ".join(methods[c].sort_values(ascending=False).index[:5]) for c in methods],
        "top10_overlap_with_consensus": [
            len(set(methods[c].sort_values(ascending=False).index[:10]) & consensus_top10)
            for c in methods],
    })
    (RESULTS_DIR / "importance_top5.md").write_text(to_markdown(overlap))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rank correlation")
    ax.set_title("Agreement between importance methods (30 features)")
    save(fig, RESULTS_DIR / "importance_agreement.png")
    return ranks, corr


def cv_scores(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1"],
                       n_jobs=1 if "XGB" in type(model).__name__ else N_JOBS)
    return r["test_accuracy"].mean(), r["test_f1"].mean()


def ablation(model, name, ranking, X_tr, y_tr, X_te, y_te):
    """Refit `model`'s configuration on its top-k features.

    The ranking comes from SHAP on a training sample, so choosing the
    features never looks at the test set."""
    rows = []
    for k in KS:
        top = set(ranking.index[:k])
        feats = [f for f in FULL if f in top]  # dataset order, so k=30 is the main model
        cv_acc, cv_f1 = cv_scores(clone(model), X_tr[feats], y_tr)
        m = clone(model).fit(X_tr[feats], y_tr)
        test_acc = (m.predict(X_te[feats]) == y_te).mean()
        rows.append({"k": k, "cv_accuracy": cv_acc, "cv_f1": cv_f1, "test_accuracy": test_acc,
                     "features": ", ".join(feats)})
        print(f"  ablation k={k:2d}: CV acc {cv_acc:.4f}  test acc {test_acc:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    (RESULTS_DIR / "ablation.md").write_text(to_markdown(df.drop(columns="features")))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df.k, df.cv_accuracy, marker="o", ms=8, color=ACCENT, label="5-fold CV accuracy (train)")
    ax.plot(df.k, df.test_accuracy, marker="s", ms=8, color=SECOND, label="Test accuracy")
    for _, r in df.iterrows():
        ax.annotate(f"{r.test_accuracy:.3f}", (r.k, r.test_accuracy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=TEXT_2)
    ax.set_xticks(KS)
    ax.set_xlabel("Number of top features (k), ranked by mean |SHAP| on training data")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Ablation: {MODEL_LABELS[name]} on top-k features")
    ax.legend(loc="lower right")
    save(fig, RESULTS_DIR / "ablation.png")
    return df


def lite_gap(model, name, consensus, X_tr, y_tr):
    """How much of the full-vs-lite CV gap each excluded feature recovers.

    Uses CV on the training set only (no test-set peeking)."""
    lite_acc, lite_f1 = cv_scores(clone(model), X_tr[LITE], y_tr)
    full_acc, full_f1 = cv_scores(clone(model), X_tr[FULL], y_tr)
    gap = full_f1 - lite_f1
    rows = [{"feature_set": "lite (21)", "added": "", "cv_accuracy": lite_acc, "cv_f1": lite_f1,
             "share_of_gap_recovered": 0.0}]
    for f in EXCLUDED_FROM_LITE:
        acc, f1 = cv_scores(clone(model), X_tr[LITE + [f]], y_tr)
        rows.append({"feature_set": "lite + 1", "added": f, "cv_accuracy": acc, "cv_f1": f1,
                     "share_of_gap_recovered": (f1 - lite_f1) / gap})
    # Cumulative: add excluded features in consensus-importance order.
    order = [f for f in consensus.index if f in EXCLUDED_FROM_LITE]
    for i in range(2, len(order)):
        feats = LITE + order[:i]
        acc, f1 = cv_scores(clone(model), X_tr[feats], y_tr)
        rows.append({"feature_set": f"lite + top {i}", "added": ", ".join(order[:i]),
                     "cv_accuracy": acc, "cv_f1": f1, "share_of_gap_recovered": (f1 - lite_f1) / gap})
    rows.append({"feature_set": "full (30)", "added": "all 9", "cv_accuracy": full_acc,
                 "cv_f1": full_f1, "share_of_gap_recovered": 1.0})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "lite_gap.csv", index=False)
    (RESULTS_DIR / "lite_gap.md").write_text(to_markdown(df))
    print(f"  lite gap ({MODEL_LABELS[name]}): CV F1 lite {lite_f1:.4f} vs full {full_f1:.4f}")

    single = df[df.feature_set == "lite + 1"].sort_values("share_of_gap_recovered")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(single.added, single.share_of_gap_recovered * 100, color=ACCENT, height=0.6)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_xlabel("Share of full-vs-lite CV F1 gap recovered (%)")
    ax.set_title("Adding one excluded feature back to the lite set")
    ax.grid(axis="y", visible=False)
    save(fig, RESULTS_DIR / "lite_gap.png")
    return df


def main():
    X, y = data.load()
    X_tr, X_te, y_tr, y_te = data.split(X, y)
    models = load_models()
    best = best_tree_model()
    print(f"Best tree model (full set, CV F1): {best}")

    builtin = builtin_importance(models)
    perm = perm_importance(models, X_te, y_te)
    shap_test = shap_analysis(models[best], best, X_te.sample(n=SHAP_SAMPLE, random_state=SEED))

    sample = X_tr.sample(n=SHAP_SAMPLE, random_state=SEED)
    _, train_values = shap_values(models[best], sample)
    shap_train = pd.Series(np.abs(train_values).mean(axis=0), index=FULL).sort_values(ascending=False)
    shap_train.to_csv(RESULTS_DIR / "shap_train_ranking.csv", header=["mean_abs_shap"])

    methods = pd.concat({
        **{f"builtin_{n}": builtin[n] for n in TREE_MODELS},
        **{f"perm_{n}": perm[n] for n in MODEL_NAMES},
        f"shap_{best}": shap_test,
    }, axis=1)
    ranks, corr = agreement(methods)
    print("Consensus top 10:", list(ranks.index[:10]))
    print("Mean Spearman between methods:",
          round(corr.values[np.triu_indices(len(corr), 1)].mean(), 3))

    ablation(models[best], best, shap_train, X_tr, y_tr, X_te, y_te)
    lite_gap(models[best], best, ranks, X_tr, y_tr)
    (RESULTS_DIR / "importance_meta.json").write_text(json.dumps({"best_tree_model": best}))


if __name__ == "__main__":
    main()
