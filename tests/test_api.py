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
                        lambda url: real_extract(url, fetcher=fake_fetch, resolver=lambda h: True))
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
    monkeypatch.setattr(extractor, "extract", lambda url: REAL_EXTRACT(
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
