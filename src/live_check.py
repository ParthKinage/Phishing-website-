"""Live sanity check of the deployed pipeline on well-known legitimate sites.

Runs the real extractor (network required) and the served lite model on a
fixed list of popular, legitimate home/login pages, then compares the
share of phishing-like (-1) values per feature with the share among the
*legitimate* rows of the dataset. Large differences show where the live
extractor cannot reproduce the dataset's encoding.

No phishing pages are fetched. Results depend on the sites' current HTML,
so they change over time; the run date is saved with them.

Run (after src.train): python -m src.live_check
"""
import datetime as dt
import json

import joblib
import pandas as pd

from src import data
from src.evaluate import to_markdown
from src.extract import extract, to_vector
from src.features import LITE
from src.train import MODELS_DIR, RESULTS_DIR

LEGIT_URLS = [
    "https://www.google.com/",
    "https://www.wikipedia.org/",
    "https://github.com/login",
    "https://www.python.org/",
    "https://www.bbc.co.uk/",
    "https://www.amazon.com/",
    "https://www.microsoft.com/",
    "https://www.apple.com/",
    "https://stackoverflow.com/",
    "https://www.reddit.com/",
    "https://www.nytimes.com/",
    "https://www.paypal.com/signin",
    "https://www.ox.ac.uk/",
    "https://www.mozilla.org/",
    "https://www.cloudflare.com/",
    "https://www.ebay.com/",
    "https://www.linkedin.com/login",
    "https://www.netflix.com/login",
    "https://www.dropbox.com/login",
    "https://www.gov.uk/",
]


def main():
    serving = json.loads((MODELS_DIR / "serving.json").read_text())
    model = joblib.load(MODELS_DIR / serving["model_file"])
    rows = []
    for url in LEGIT_URLS:
        r = extract(url)
        vec = to_vector(r, serving["fill_values"])
        proba = float(model.predict_proba(pd.DataFrame([vec], columns=LITE))[0, 1])
        rows.append({"url": url, "phishing_probability": proba,
                     "label": "phishing" if proba >= serving["threshold"] else "legitimate",
                     "partial": r["partial"], **dict(zip(LITE, vec))})
        print(f"  {proba:6.3f}  {'PARTIAL ' if r['partial'] else ''}{url}")
    live = pd.DataFrame(rows)
    live.to_csv(RESULTS_DIR / "live_check.csv", index=False)

    X, y = data.load()
    legit = X.loc[y == 0, LITE]
    full_rows = live[~live.partial]
    compare = pd.DataFrame({
        "feature": LITE,
        "dataset_legit_share_-1": [(legit[f] == -1).mean() for f in LITE],
        "live_legit_share_-1": [(full_rows[f] == -1).mean() for f in LITE],
    })
    if "redirect" in LITE:  # redirect has no -1; compare the share of value 1 instead
        i = LITE.index("redirect")
        compare.loc[i, "dataset_legit_share_-1"] = (legit["redirect"] == 1).mean()
        compare.loc[i, "live_legit_share_-1"] = (full_rows["redirect"] == 1).mean()
        compare.loc[i, "feature"] = "redirect (share of 1)"
    compare["difference"] = compare["live_legit_share_-1"] - compare["dataset_legit_share_-1"]
    compare.to_csv(RESULTS_DIR / "live_vs_dataset.csv", index=False)
    (RESULTS_DIR / "live_vs_dataset.md").write_text(to_markdown(compare, "{:.2f}"))

    summary = {
        "run_at": dt.datetime.now().isoformat(timespec="minutes"),
        "n_sites": len(live),
        "n_partial": int(live.partial.sum()),
        "n_flagged_phishing": int((live.label == "phishing").sum()),
        "flagged": live.loc[live.label == "phishing", "url"].tolist(),
        "mean_phishing_probability": float(live.phishing_probability.mean()),
    }
    (RESULTS_DIR / "live_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(to_markdown(compare, "{:.2f}"))


if __name__ == "__main__":
    main()
