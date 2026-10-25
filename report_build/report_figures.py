"""Report-sized variants of result figures (readable at A4 page width).

Run: python -m report_build.report_figures
"""
import pandas as pd

from src.plotstyle import MODEL_COLORS, MODEL_LABELS, TEXT_2, plt, save
from src.train import RESULTS_DIR

TOP = 15


def permutation_2x2():
    df = pd.read_csv(RESULTS_DIR / "importance_permutation.csv", header=[0, 1], index_col=0)
    mean, std = df["mean"], df["std"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, model in zip(axes.ravel(), MODEL_LABELS):
        top = mean[model].sort_values().tail(TOP)
        ax.barh(top.index, top.values, xerr=std.loc[top.index, model], color=MODEL_COLORS[model],
                height=0.7, error_kw={"ecolor": TEXT_2, "lw": 1})
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Drop in test F1 when shuffled")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Permutation importance on the test set: top {TOP} features per model "
                 "(10 repeats, mean ± std)", fontweight="bold")
    fig.tight_layout()
    save(fig, RESULTS_DIR / "report_permutation.png")


if __name__ == "__main__":
    permutation_2x2()
