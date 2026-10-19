"""Download, cache, validate, summarise and split the UCI Phishing Websites data.

Run `python -m src.data` to download the data and print/save a summary.
"""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.features import FULL, TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE = DATA_DIR / "phishing.csv"

UCI_ID = 327
ARFF_ZIP_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
SEED = 42
TEST_SIZE = 0.2


def _from_ucimlrepo():
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=UCI_ID)
    return pd.concat([d.data.features, d.data.targets], axis=1)


def _from_arff():
    """Fallback: download the original ARFF from the UCI archive and parse it."""
    resp = requests.get(ARFF_ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        text = z.read("Training Dataset.arff").decode("utf-8")
    names, rows, in_data = [], [], False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if in_data:
            rows.append([int(v) for v in line.split(",")])
        elif line.lower().startswith("@attribute"):
            names.append(line.split()[1])
        elif line.lower() == "@data":
            in_data = True
    return pd.DataFrame(rows, columns=names)


def _normalise_columns(df):
    """Lower-case the names; the ARFF uses e.g. 'having_IP_Address', 'Result'.

    Two ARFF names differ from the ucimlrepo ones beyond case, so map them.
    """
    df = df.rename(columns=lambda c: c.strip().lower())
    return df.rename(columns={
        "domain_registeration_length": "domain_registration_length",
        "popupwidnow": "popupwindow",
    })


def load_raw(refresh=False):
    """Return the raw dataframe (30 features + 'result', original -1/1 label)."""
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    DATA_DIR.mkdir(exist_ok=True)
    try:
        df = _from_ucimlrepo()
        source = "ucimlrepo"
    except Exception as e:  # network problems, API changes, ...
        print(f"ucimlrepo failed ({e!r}); falling back to ARFF download")
        df = _from_arff()
        source = "arff"
    df = _normalise_columns(df)
    validate(df)
    df = df[FULL + [TARGET]]
    df.to_csv(CACHE, index=False)
    print(f"Downloaded {len(df)} rows via {source}; cached at {CACHE}")
    return df


def validate(df):
    missing = set(FULL + [TARGET]) - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert not df[FULL + [TARGET]].isna().any().any(), "dataset has missing values"
    bad = {c for c in FULL if not set(df[c].unique()) <= {-1, 0, 1}}
    assert not bad, f"unexpected values in {bad}"
    # The dataset documentation: Result -1 = phishing, 1 = legitimate.
    assert set(df[TARGET].unique()) == {-1, 1}, "unexpected label coding"


def load():
    """Return X (DataFrame, FULL columns) and y (Series, phishing = 1)."""
    df = load_raw()
    X = df[FULL].astype(int)
    y = (df[TARGET] == -1).astype(int).rename("phishing")
    return X, y


def split(X, y, dedup=False):
    """Stratified 80/20 split. With dedup=True, exact duplicate rows
    (same feature values *and* label) are dropped first, so no row can
    appear on both sides of the split."""
    if dedup:
        keep = ~pd.concat([X, y], axis=1).duplicated()
        X, y = X[keep], y[keep]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def summarise(X, y):
    """Print and save shape, class balance, per-feature value counts and duplicates."""
    RESULTS_DIR.mkdir(exist_ok=True)
    both = pd.concat([X, y], axis=1)
    conflicting = both[X.duplicated(keep=False)].groupby(list(X.columns))["phishing"].nunique()
    summary = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "phishing": int(y.sum()),
        "legitimate": int((1 - y).sum()),
        "phishing_share": round(float(y.mean()), 4),
        "duplicate_rows": int(both.duplicated().sum()),
        "unique_rows": int((~both.duplicated()).sum()),
        "duplicate_feature_vectors": int(X.duplicated().sum()),
        "conflicting_feature_vectors": int((conflicting > 1).sum()),
    }
    rows = []
    for c in X.columns:
        for v, n in X[c].value_counts().sort_index().items():
            rows.append({
                "feature": c,
                "value": int(v),
                "count": int(n),
                "share": round(n / len(X), 4),
                "phishing_rate": round(float(y[X[c] == v].mean()), 4),
            })
    counts = pd.DataFrame(rows)
    (RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    counts.to_csv(RESULTS_DIR / "feature_value_counts.csv", index=False)

    print(f"Shape: {X.shape}")
    print(f"Class balance: phishing={summary['phishing']} legitimate={summary['legitimate']} "
          f"({summary['phishing_share']:.1%} phishing)")
    print(f"Exact duplicate rows: {summary['duplicate_rows']} "
          f"(unique rows: {summary['unique_rows']}); feature vectors with conflicting "
          f"labels: {summary['conflicting_feature_vectors']}")
    print("Value counts (count, phishing rate) per feature:")
    for c, g in counts.groupby("feature", sort=False):
        cells = "  ".join(f"{r.value:+d}: {r['count']:5d} ({r.phishing_rate:.2f})" for _, r in g.iterrows())
        print(f"  {c:28s} {cells}")
    return summary


if __name__ == "__main__":
    X, y = load()
    summarise(X, y)
    X_tr, X_te, y_tr, y_te = split(X, y)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
    X_tr, X_te, y_tr, y_te = split(X, y, dedup=True)
    print(f"De-duplicated train: {len(X_tr)}  test: {len(X_te)}")
