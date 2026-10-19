"""API tests. The page fetch and DNS lookup are replaced with fakes."""
import pytest
from fastapi.testclient import TestClient

from api import main
from src import extract as extractor
from src.features import LITE

REAL_EXTRACT = extractor.extract  # captured before any test patches it

PHISHY_HTML = """<html><body>
<form action="about:blank"><input name="password"></form>
<a href="#">a</a><a href="#">b</a><a href="javascript:void(0)">c</a>
<script src="https://evil-cdn.com/x.js"></script>
</body></html>"""


@pytest.fixture
def client(monkeypatch):
    real_extract = extractor.extract

    def fake_fetch(url):
        return {"html": PHISHY_HTML, "final_url": url, "n_redirects": 0,
                "cert_valid": True, "is_html": True}

    monkeypatch.setattr(extractor, "extract",
                        lambda url, **kw: real_extract(url, fetcher=fake_fetch, resolver=lambda h: True))
    with TestClient(main.app) as c:   # runs the startup hook that loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == len(LITE)


def test_predict_valid_url(client):
    r = client.post("/predict", json={"url": "http://secure-login.paypal.example.com/verify"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("phishing", "legitimate")
    assert 0.0 <= body["phishing_probability"] <= 1.0
    assert body["partial_features"] is False
    assert [f["name"] for f in body["features"]] == LITE
    assert 1 <= len(body["top_reasons"]) <= 5
    for reason in body["top_reasons"]:
        assert reason["direction"] in ("towards phishing", "towards legitimate")
    # This page has an empty form handler, unsafe anchors and a dash in the host:
    assert body["label"] == "phishing"


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<form" in r.text


def test_demo_phishing_page_is_flagged(client, monkeypatch):
    page = client.get("/demo/account-verify")
    assert page.status_code == 200
    # Feed the demo page's own HTML through the extractor (no network).
    monkeypatch.setattr(extractor, "extract", lambda url, **kw: REAL_EXTRACT(
        url, resolver=lambda h: True,
        fetcher=lambda u: {"html": page.text, "final_url": u, "n_redirects": 0,
                           "cert_valid": True, "is_html": True}))
    r = client.post("/predict", json={"url": "http://127.0.0.1:8000/demo/account-verify"})
    assert r.json()["label"] == "phishing"


@pytest.mark.parametrize("payload", [
    {"url": "ftp://example.com/file"},
    {"url": "javascript:alert(1)"},
    {"url": "not a url"},
    {"url": "http://"},
    {"url": "https://example.com/" + "a" * 3000},
    {"link": "https://example.com/"},
    {},
])
def test_predict_rejects_malformed_input(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_non_json(client):
    r = client.post("/predict", content="https://example.com", headers={"Content-Type": "text/plain"})
    assert r.status_code == 422


def test_cors_allows_extension_origin(client):
    r = client.options("/predict", headers={
        "Origin": "chrome-extension://abcdefghijklmnop",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.headers.get("access-control-allow-origin") == "chrome-extension://abcdefghijklmnop"


def test_cors_rejects_other_origins(client):
    r = client.options("/predict", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in r.headers


# --------------------------------------------------------------- public mode

def test_public_mode_hides_demo_pages(client, monkeypatch):
    monkeypatch.setattr(main, "PUBLIC", True)
    assert client.get("/demo/account-verify").status_code == 404


def test_public_mode_refuses_private_addresses(client, monkeypatch):
    monkeypatch.setattr(main, "PUBLIC", True)
    monkeypatch.setattr(extractor, "extract", REAL_EXTRACT)  # the real check, no fake
    r = client.post("/predict", json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert r.status_code == 422
    assert "private or internal" in r.json()["detail"]


def test_public_mode_rate_limit(client, monkeypatch):
    monkeypatch.setattr(main, "PUBLIC", True)
    monkeypatch.setattr(main, "RATE_LIMIT", (3, 60))
    main._recent.clear()
    codes = [client.post("/predict", json={"url": "https://example.com/"}).status_code for _ in range(4)]
    main._recent.clear()
    assert codes == [200, 200, 200, 429]


def test_native_xgboost_matches_joblib_model():
    import json
    serving = json.loads((main.MODELS_DIR / "serving.json").read_text())
    native = main.NativeXGBoost(main.MODELS_DIR / serving["native_model_file"], serving["features"])
    fallback = main.SklearnModel(main.MODELS_DIR / serving["model_file"], serving)
    for vector in ([1] * len(LITE), [-1] * len(LITE), list(serving["fill_values"].values())):
        p1, c1, _ = native.score(vector)
        p2, c2, _ = fallback.score(vector)
        assert abs(p1 - p2) < 1e-6
        assert max(abs(a - b) for a, b in zip(c1, c2)) < 1e-4
