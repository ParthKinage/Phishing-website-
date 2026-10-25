"""Exploratory data analysis figures for the report.

Run: python -m src.eda
Writes results/eda_*.png and results/eda_label_correlation.csv.
"""
import json

import numpy as np
import pandas as pd

from src import data
from src.features import LITE
from src.plotstyle import ACCENT, GRID, SECOND, TEXT_2, plt, save
from src.train import RESULTS_DIR

KEY_FEATURES = ["sslfinal_state", "url_of_anchor", "web_traffic", "prefix_suffix",
                "having_sub_domain", "links_in_tags", "request_url", "sfh"]


def class_and_duplicates(X, y):
    summary = json.loads((RESULTS_DIR / "data_summary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    counts = [summary["legitimate"], summary["phishing"]]
    bars = a.bar(["Legitimate", "Phishing"], counts, color=[ACCENT, SECOND], width=0.55)
    for bar, n in zip(bars, counts):
        a.annotate(f"{n:,} ({n / summary['rows']:.1%})", (bar.get_x() + bar.get_width() / 2, n),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=TEXT_2)
    a.set_ylabel("Websites")
    a.set_title("Class distribution")
    a.set_ylim(0, max(counts) * 1.15)
    a.grid(axis="x", visible=False)

    vals = [summary["unique_rows"], summary["duplicate_rows"]]
    bars = b.bar(["Unique rows", "Exact duplicates"], vals, color=[ACCENT, GRID], width=0.55,
                 edgecolor=TEXT_2, linewidth=0.5)
    for bar, n in zip(bars, vals):
        b.annotate(f"{n:,} ({n / summary['rows']:.0%})", (bar.get_x() + bar.get_width() / 2, n),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=TEXT_2)
    b.set_title("Duplicate rows (features + label)")
    b.set_ylim(0, max(vals) * 1.15)
    b.grid(axis="x", visible=False)
    save(fig, RESULTS_DIR / "eda_class_duplicates.png")


def phishing_rate_by_value():
    counts = pd.read_csv(RESULTS_DIR / "feature_value_counts.csv")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    for ax, f in zip(axes.ravel(), KEY_FEATURES):
        g = counts[counts.feature == f].sort_values("value")
        colors = {-1: SECOND, 0: "#eda100", 1: ACCENT}
        bars = ax.bar([f"{v:+d}" for v in g.value], g.phishing_rate * 100,
                      color=[colors[v] for v in g.value], width=0.6)
        for bar, n in zip(bars, g["count"]):
            ax.annotate(f"n={n:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=TEXT_2)
        ax.set_title(f, fontsize=10)
        ax.set_ylim(0, 112)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Phishing rate (%)")
    fig.supxlabel("Feature value (-1 phishing-like, 0 suspicious, 1 legitimate-like)", color=TEXT_2, fontsize=10)
    fig.suptitle("Share of phishing websites for each value of eight key features", fontweight="bold")
    fig.subplots_adjust(hspace=0.45)
    save(fig, RESULTS_DIR / "eda_phishing_rate.png")


def label_correlation(X, y):
    corr = X.apply(lambda col: np.corrcoef(col, y)[0, 1]).sort_values()
    corr.rename("pearson_r_with_phishing").to_csv(RESULTS_DIR / "eda_label_correlation.csv")
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = [SECOND if v > 0 else ACCENT for v in corr]
    ax.barh(corr.index, corr.values, color=colors, height=0.7)
    for i, f in enumerate(corr.index):
        if f not in LITE:
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_xlabel("Pearson correlation with the phishing label")
    ax.set_title("Correlation of each feature with phishing\n(bold = not in the lite set)")
    ax.grid(axis="y", visible=False)
    save(fig, RESULTS_DIR / "eda_label_correlation.png")
    return corr


def main():
    X, y = data.load()
    class_and_duplicates(X, y)
    phishing_rate_by_value()
    corr = label_correlation(X, y)
    print("Most negative correlations (value -1 => phishing):")
    print(corr.head(5).round(3).to_string())
    print("Most positive:")
    print(corr.tail(3).round(3).to_string())


if __name__ == "__main__":
    main()
