# Phishing Website Detection

Machine Learning course project. It classifies websites as **phishing** or
**legitimate** using the UCI Phishing Websites dataset (11,055 sites, 30
features coded -1/0/1). It compares four models (Decision Tree, Random
Forest, SVM, XGBoost) on two feature sets, and includes a working demo:

- **Full (30 features):** the academic comparison.
- **Lite (21 features):** only the features a Python function can compute
  from a live URL and its static HTML in a few seconds. The best lite model
  is served by a local API and used by a Chrome extension.

The results, feature-importance analysis and limitations are in
[report.md](report.md).

## Layout

```
data/                 raw dataset cache (created on first run, gitignored)
src/
  data.py             download (ucimlrepo, ARFF fallback), validate, summarise, split
  features.py         FULL / LITE feature lists and encoding notes
  train.py            grid search with 5-fold CV, final fit, model selection for serving
  evaluate.py         test-set metrics, confusion matrices, ROC curves, duplicate check
  importance.py       built-in / permutation / SHAP importance, ablation, full-vs-lite gap
  extract.py          live URL -> lite feature vector
  live_check.py       optional: runs the live pipeline on 20 well-known legitimate sites
  run_all.py          runs data -> train -> evaluate -> importance
  plotstyle.py        shared figure style
api/
  main.py             FastAPI app (/predict, /health, /)
  static/index.html   try-it page
extension/            Chrome extension (Manifest V3)
models/               saved models (joblib) and serving.json
results/              metrics (CSV/Markdown/JSON) and figures (PNG)
tests/                pytest suite
```

## Setup

Python 3.10+ (tested with 3.13). On Windows:

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\activate
```

On macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

## Train and evaluate everything

One command downloads the data, trains and tunes all eight models, evaluates
them on the test set, and runs the importance analysis. It takes about 7
minutes on a laptop CPU.

```bash
python -m src.run_all
```

Or run the stages separately, in this order:

```bash
python -m src.data
```

```bash
python -m src.train
```

```bash
python -m src.evaluate
```

```bash
python -m src.importance
```

Outputs:

- `results/data_summary.json`, `results/feature_value_counts.csv`: dataset inspection
- `results/cv_results.csv|json`: CV scores, best hyper-parameters, timings
- `results/metrics_full.md|csv`, `results/metrics_lite.md|csv`: test-set comparison tables
- `results/confusion_*.png`, `results/roc_*.png`
- `results/dedup_sensitivity.md|csv`: results on de-duplicated data
- `results/importance_*.png|csv|md`, `results/shap_*.png`, `results/ablation.*`, `results/lite_gap.*`
- `models/{full,lite}_{decision_tree,random_forest,svm,xgboost}.joblib`
- `models/serving.json`: which lite model the API serves and why, plus fill values

Optional network check on 20 legitimate sites (writes `results/live_check.*`
and `results/live_vs_dataset.*`):

```bash
python -m src.live_check
```

## Run the API

Train first (the API loads `models/serving.json`). On Windows, the easiest
way is to double-click **`start_app.bat`** in the project folder; it always
uses the project's `.venv`. Or, from the project root:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- Web page: <http://127.0.0.1:8000/>
- Health: `GET http://127.0.0.1:8000/health`
- Predict: `POST http://127.0.0.1:8000/predict` with body `{"url": "https://example.com/"}`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"url\": \"https://example.com/\"}"
```

The response contains `label`, `phishing_probability`, every extracted
feature, `top_reasons` (per-prediction SHAP values for the served tree
model), and `partial_features`. That flag is true when the page could not
be fetched. In that case the page features are filled with their most common
value in the training data, and the result rests on the URL features.

The server fetches the page you submit (static HTML only, 5 s timeouts,
2 MB cap, no JavaScript). Run it only on your own machine; it binds to
127.0.0.1.

### Test pages (fake phishing)

The API also serves three harmless fake phishing pages. They use no real
brand names, and their forms send nothing. Paste their address into the
try-it page to see the detector at work:

| Address | Pattern | Result |
|---|---|---|
| `http://127.0.0.1:8000/demo/account-verify` | old-style kit: form to `about:blank`, dead links, hidden iframe, right-click disabled | phishing, 99.45% |
| `http://127.0.0.1:8000/demo/invoice` | fake invoice: card form that e-mails the data, status-bar spoofing | phishing, 99.62% |
| `http://127.0.0.1:8000/demo/modern-login` | modern kit: clean HTML, everything on one domain | **legitimate, 0.07% (missed)**: shows the model's main limitation |

## Online deployment (Vercel)

`deploy/vercel/` is a slim copy of the API for hosting, generated from the
project's own code:

```bash
.venv\Scripts\python -m src.build_deploy
```

It serves the XGBoost model through XGBoost's own engine
(`models/lite_xgboost.json`, identical predictions and SHAP values), so it
needs no pandas, scikit-learn or shap and fits Vercel's function size limit.
The online version runs in **public mode** (`PHISHING_PUBLIC=1`):

- URLs pointing at private or internal addresses (including through
  redirects) are refused, so the server cannot be used to probe its own network;
- the fake phishing test pages are not deployed;
- each client is limited to 20 checks per minute.

The Vercel project uses `deploy/vercel` as its root directory, and every push
to `main` redeploys it. After retraining or changing `api/` or
`src/extract.py`, re-run the build command above and commit the result.
The Chrome extension still talks to the local API at `127.0.0.1:8000`.

## Demo notebook (Google Colab)

`demo_colab.ipynb` walks through the dataset, results, feature importance,
how XGBoost decides, and live checks with explanations, including a test
suite of fake phishing pages and real legitimate sites. It uses the project's
own code and saved models.

1. Create the upload bundle (code, models, results; about 12 MB). From the project root:
   ```bash
   .venv\Scripts\python -c "import zipfile; from pathlib import Path; skip={'.venv','data','__pycache__','.pytest_cache','.claude'}; z=zipfile.ZipFile('phishing_project.zip','w',zipfile.ZIP_DEFLATED); [z.write(p, Path('phishing_project')/p) for p in sorted(Path('.').rglob('*')) if p.is_file() and not skip & set(p.parts) and p.name!='phishing_project.zip']; z.close()"
   ```
2. Open <https://colab.research.google.com>, choose **File → Upload notebook**, and pick `demo_colab.ipynb`.
3. Run the cells in order. Cell 1a asks you to upload `phishing_project.zip`. Cell 1b installs the pinned libraries; if Colab then asks you to restart the session, do it and continue from cell 1c.

The notebook also runs locally in Jupyter from the project folder (it skips the upload and install steps).

## Load the Chrome extension

1. Start the API (above).
2. Open `chrome://extensions` in Chrome.
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and choose the `extension/` folder of this project.
5. Pin the extension, open any website, and click the extension icon.

The popup sends the current tab's URL to `http://127.0.0.1:8000/predict` and
shows the verdict, probability, and top reasons. If the API is not running,
it says so. The extension asks only for `activeTab` and access to
`http://127.0.0.1:8000/*`. It sends nothing anywhere else and collects
nothing in the background.

## Run the tests

```bash
python -m pytest
```

The tests use no network: the extractor tests use hand-written URLs and HTML
fixtures, and the API tests replace the page fetch with a fake. The API test
and the model smoke test need the trained models, so run training first.

## Troubleshooting

**`ImportError: DLL load failed ... An Application Control policy has blocked this file`**
(or any import error that mentions `...\Python314\site-packages\...`): the
command ran your system Python instead of the project's virtual environment.
Activate the environment first (`.venv\Scripts\activate`; the prompt then
starts with `(.venv)`), or call its Python directly:

```bash
.venv\Scripts\python -m src.run_all
```

## Notes

- Seeds are fixed (`random_state=42`) everywhere, so re-running gives the
  same numbers on the same library versions.
- scikit-learn 1.9 warns that `SVC(probability=True)` is deprecated (to be
  removed in 1.11). It works in the pinned version; the warning is silenced.
- Grid search uses at most 8 worker processes, because `n_jobs=-1` on a
  32-thread machine ran out of memory. XGBoost tunes in-process with its own
  threads.
