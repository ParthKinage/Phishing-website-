"""Shared matplotlib style so every figure in results/ looks consistent.

Colors follow a validated colorblind-safe categorical palette. Each model
always gets the same hue (color follows the entity, not its rank).
"""
import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to PNG
import matplotlib.pyplot as plt  # noqa: E402

MODEL_COLORS = {
    "decision_tree": "#2a78d6",  # blue
    "random_forest": "#eb6834",  # orange
    "svm": "#1baf7a",            # aqua
    "xgboost": "#eda100",        # yellow
}
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "xgboost": "XGBoost",
}
ACCENT = "#2a78d6"
SECOND = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
