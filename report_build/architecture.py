"""Draws the system block diagram (report Figure 3.1) to results/architecture.png.

Run: python -m report_build.architecture
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.plotstyle import ACCENT, SECOND, TEXT, TEXT_2, plt, save
from src.train import RESULTS_DIR

OFFLINE = "#e7f0fb"   # light blue lane
ONLINE = "#fdeee6"    # light orange lane


def box(ax, x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=edge, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=TEXT, zorder=3)
    ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=8.6,
            color=TEXT_2, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=TEXT_2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(13, 6.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    # Lanes
    ax.add_patch(FancyBboxPatch((0.15, 3.55), 12.7, 2.6, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=OFFLINE, edgecolor="none", zorder=0))
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 12.7, 3.05, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=ONLINE, edgecolor="none", zorder=0))
    ax.text(0.35, 5.98, "A. Offline training pipeline (python -m src.run_all)", fontsize=11.5,
            fontweight="bold", color=ACCENT, va="top")
    ax.text(0.35, 3.03, "B. Online prediction (FastAPI service, local or Vercel)", fontsize=11.5,
            fontweight="bold", color=SECOND, va="top")

    w, h = 2.25, 1.55
    xs = [0.4, 2.9, 5.4, 7.9, 10.4]
    y = 3.8
    offline = [
        ("UCI dataset", "11,055 websites\n30 features (-1/0/1)\nlabel: phishing = 1"),
        ("Pre-processing", "validate & recode\nduplicate check\nstratified 80/20 split"),
        ("Model training", "DT, RF, SVM, XGBoost\nfull (30) & lite (21)\nGridSearchCV, 5-fold"),
        ("Evaluation", "test set used once\nAcc, P, R, F1, AUC\ndedup sensitivity"),
        ("Interpretation", "built-in, permutation\nSHAP, ablation\nfull-vs-lite gap"),
    ]
    for x, (t, b) in zip(xs, offline):
        box(ax, x, y, w, h, t, b, ACCENT)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.02, y + h / 2, b - 0.02, y + h / 2, color=ACCENT)

    y2 = 0.95
    online = [
        ("Client", "web page or\nChrome extension\nsends a URL"),
        ("FastAPI /predict", "validates http(s) URL\n(public mode: blocks\nprivate addresses)"),
        ("Feature extractor", "9 URL features\nDNS + TLS check\nstatic HTML: 12 features"),
        ("XGBoost (lite)", "21 features ->\nphishing probability\n>= 0.5 => phishing"),
        ("Explanation", "per-prediction\nTreeSHAP values\n=> top 5 reasons"),
    ]
    for x, (t, b) in zip(xs, online):
        box(ax, x, y2, w, h, t, b, SECOND)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.02, y2 + h / 2, b - 0.02, y2 + h / 2, color=SECOND)

    # Result back to the client: down, left along the lane floor, up.
    xr, xc, yb = xs[-1] + w / 2, xs[0] + w / 2, 0.45
    ax.plot([xr, xr, xc], [y2 - 0.02, yb, yb], color=SECOND, lw=1.3, ls="--", zorder=1)
    arrow(ax, xc, yb, xc, y2 - 0.03, color=SECOND)
    ax.text((xr + xc) / 2, yb + 0.07, "verdict + probability + top reasons (JSON)",
            ha="center", va="bottom", fontsize=8.5, color=SECOND, style="italic")

    # The trained model links the two lanes.
    ax.add_patch(FancyArrowPatch((xs[2] + w / 2, y - 0.02), (xs[3] + w / 2, y2 + h + 0.03),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                 color=TEXT_2, zorder=1))
    ax.text(xs[3] + 0.05, 3.18, "saved model\n(XGBoost JSON)", ha="left", va="center",
            fontsize=8.5, color=TEXT_2, style="italic")

    save(fig, RESULTS_DIR / "architecture.png")


if __name__ == "__main__":
    main()
