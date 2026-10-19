"""Prediction API for the lite phishing model.

Run from the project root:
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Public mode (environment variable PHISHING_PUBLIC=1, used only for the
online deployment) blocks private/internal addresses, hides the fake
phishing demo pages and rate-limits each client.
"""
import json
import os
import threading
import time
import warnings
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src import extract as extractor
from src.features import DESCRIPTIONS

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TREE_MODELS = {"decision_tree", "random_forest", "xgboost"}
TOP_REASONS = 5
PUBLIC = os.environ.get("PHISHING_PUBLIC") == "1"
RATE_LIMIT = (20, 60)  # public mode: at most 20 predictions per client per 60 s

state = {}


class NativeXGBoost:
    """The served XGBoost model run by XGBoost itself, from its JSON export.

    Needs only xgboost and numpy (no pandas/scikit-learn/shap), which keeps
    the deployment small. `pred_contribs` is XGBoost's built-in TreeSHAP,
    so the explanations equal those of the shap package."""

    def __init__(self, path, features):
        import xgboost as xgb
        self.xgb, self.features = xgb, features
        self.booster = xgb.Booster()
        self.booster.load_model(path)

    def score(self, vector):
        dm = self.xgb.DMatrix(np.array([vector], dtype=float), feature_names=self.features)
        proba = float(self.booster.predict(dm)[0])
        contrib = self.booster.predict(dm, pred_contribs=True)[0, :-1]  # last column = bias
        return proba, contrib, "shap"


class SklearnModel:
    """Fallback for any saved joblib model (needs pandas, scikit-learn and,
    for tree models, shap). Non-tree models are explained by occlusion: the
    change in phishing probability when a feature is replaced by its
    training-set mode."""

    def __init__(self, path, serving):
        import joblib
        import pandas as pd
        self.pd, self.serving = pd, serving
        self.model = joblib.load(path)
        self.explainer = None
        if serving["model"] in TREE_MODELS:
            import shap
            self.explainer = shap.TreeExplainer(self.model)

    def score(self, vector):
        features = self.serving["features"]
        row = self.pd.DataFrame([vector], columns=features)
        proba = float(self.model.predict_proba(row)[0, 1])
        if self.explainer is not None:
            values = self.explainer(row).values[0]
            if values.ndim == 2:  # scikit-learn: one column per class
                values = values[:, 1]
            return proba, values, "shap"
        out = []
        for f in features:
            alt = row.copy()
            alt[f] = self.serving["fill_values"][f]
            out.append(proba - self.model.predict_proba(alt)[0, 1])
        return proba, np.array(out), "occlusion"


def load_model():
    serving = json.loads((MODELS_DIR / "serving.json").read_text())
    native = serving.get("native_model_file")
    if native and (MODELS_DIR / native).exists():
        model = NativeXGBoost(MODELS_DIR / native, serving["features"])
    else:
        model = SklearnModel(MODELS_DIR / serving["model_file"], serving)
    state.update(serving=serving, model=model)


_recent = defaultdict(deque)
_recent_lock = threading.Lock()


def check_rate_limit(request: Request):
    """Public mode only: simple in-memory sliding-window limit per client IP."""
    if not PUBLIC:
        return
    forwarded = request.headers.get("x-forwarded-for", "")
    client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
    limit, window = RATE_LIMIT
    now = time.monotonic()
    with _recent_lock:
        q = _recent[client]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests; try again in a minute.")
        q.append(now)


@asynccontextmanager
async def lifespan(app):
    load_model()  # once, at startup
    yield


app = FastAPI(title="Phishing Website Detector", lifespan=lifespan)
# The Chrome extension's popup calls the API from a chrome-extension:// origin.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-z]+|http://(127\.0\.0\.1|localhost)(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    url: str = Field(..., max_length=2048)

    @field_validator("url")
    @classmethod
    def http_url(cls, v):
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            raise ValueError("url must be an http:// or https:// URL with a host name")
        return v


def value_label(feature, value):
    if feature == "redirect":
        return "redirected 2+ times" if value == 1 else "at most 1 redirect"
    return {-1: "phishing-like", 0: "suspicious", 1: "legitimate-like"}[value]


@app.get("/health")
def health():
    serving = state.get("serving", {})
    return {"status": "ok" if "model" in state else "loading",
            "model": serving.get("model"), "n_features": len(serving.get("features", []))}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    check_rate_limit(request)
    serving = state["serving"]
    try:
        result = extractor.extract(req.url, block_private=PUBLIC)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    vector = extractor.to_vector(result, serving["fill_values"])
    # Contributions: per-prediction SHAP values (log-odds for XGBoost;
    # positive = towards phishing), or occlusion for non-tree models.
    proba, contrib, method = state["model"].score(vector)

    features = []
    for f, v, c in zip(serving["features"], vector, contrib):
        features.append({
            "name": f,
            "value": int(v),
            "meaning": value_label(f, v),
            "checks": DESCRIPTIONS[f],
            "available": result["available"][f],
            "contribution": round(float(c), 4),
        })
    ranked = sorted(features, key=lambda x: -abs(x["contribution"]))[:TOP_REASONS]
    reasons = [{**f, "direction": "towards phishing" if f["contribution"] > 0 else "towards legitimate"}
               for f in ranked if f["contribution"] != 0]

    return {
        "url": req.url,
        "final_url": result["final_url"],
        "label": "phishing" if proba >= serving["threshold"] else "legitimate",
        "phishing_probability": round(proba, 4),
        "partial_features": result["partial"],
        "unavailable_features": [f for f, ok in result["available"].items() if not ok],
        "note": ("Some page features could not be computed and were filled with their most "
                 "common training value; the result is based on partial features.")
                if result["partial"] else None,
        "errors": result["errors"],
        "model": serving["model"],
        "explanation_method": method,
        "top_reasons": reasons,
        "features": features,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Harmless fake phishing pages for demos (no real brands; forms send nothing).
# Check e.g. http://127.0.0.1:8000/demo/account-verify on the try-it page.
DEMO_PAGES = {
    "account-verify": "demo-phishing.html",   # old-style kit: many classic signs
    "invoice": "demo-invoice.html",           # fake invoice, form submits to e-mail
    "modern-login": "demo-modern.html",       # clean modern kit (shows the limitation)
}


@app.get("/demo/{name}")
def demo_page(name: str):
    # Never served by the public deployment: hosted online, fake login forms
    # could be mistaken for real phishing.
    if PUBLIC or name not in DEMO_PAGES:
        raise HTTPException(status_code=404, detail="unknown demo page")
    return FileResponse(STATIC_DIR / DEMO_PAGES[name])
