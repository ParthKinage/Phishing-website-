"""Extractor tests. No network: page features use HTML fixtures, and
extract() gets fake fetch/DNS functions."""
import pytest
import requests

from src.extract import extract, page_features, to_vector, url_features
from src.features import LITE, PAGE, URL_ONLY


def uf(url, name):
    return url_features(url)[name]


# ------------------------------------------------------------ URL-only rules

@pytest.mark.parametrize("url, expected", [
    ("http://125.98.3.123/fake.html", -1),
    ("http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html", -1),   # hex form
    ("http://[2001:db8::1]/login", -1),
    ("http://www.example.com/", 1),
    ("http://123.example.com/", 1),
])
def test_having_ip_address(url, expected):
    assert uf(url, "having_ip_address") == expected


@pytest.mark.parametrize("length, expected", [(53, 1), (54, 0), (75, 0), (76, -1)])
def test_url_length(length, expected):
    base = "http://example.com/"
    url = base + "a" * (length - len(base))
    assert len(url) == length
    assert uf(url, "url_length") == expected


@pytest.mark.parametrize("url, expected", [
    ("https://bit.ly/19DXSk4", -1),
    ("https://tinyurl.com/abc", -1),
    ("https://t.co/xyz", -1),
    ("https://goo.gl/xyz", -1),
    ("https://www.example.com/", 1),
])
def test_shortining_service(url, expected):
    assert uf(url, "shortining_service") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://example.com@evil.com/", -1),
    ("http://example.com/login", 1),
])
def test_having_at_symbol(url, expected):
    assert uf(url, "having_at_symbol") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://www.legitimate.com//http://www.phishing.com", -1),
    ("https://example.com/a//b", -1),
    ("http://example.com/path", 1),    # '//' at position 6
    ("https://example.com/path", 1),   # '//' at position 7
])
def test_double_slash_redirecting(url, expected):
    assert uf(url, "double_slash_redirecting") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://www.confirme-paypal.com/", -1),
    ("http://www.paypal.com/some-path", 1),   # dash in the path only
])
def test_prefix_suffix(url, expected):
    assert uf(url, "prefix_suffix") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://www.hud.ac.uk/students/", 1),     # authors' own example
    ("http://example.com/", 1),
    ("http://www.example.com/", 1),
    ("http://mail.example.com/", 0),
    ("http://www.mail.example.co.uk/", 0),
    ("http://a.b.example.com/", -1),
    ("http://secure.login.paypal.example.com/", -1),
])
def test_having_sub_domain(url, expected):
    assert uf(url, "having_sub_domain") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://example.com:8080/", -1),
    ("http://example.com:21/", -1),
    ("http://example.com:80/", 1),
    ("https://example.com:443/", 1),
    ("https://example.com/", 1),
])
def test_port(url, expected):
    assert uf(url, "port") == expected


@pytest.mark.parametrize("url, expected", [
    ("http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/", -1),
    ("https://www.paypal.com/", 1),        # scheme is not part of the host
])
def test_https_token(url, expected):
    assert uf(url, "https_token") == expected


def test_url_features_cover_url_only_list():
    assert set(url_features("https://example.com/")) == set(URL_ONLY)


# --------------------------------------------------------------- page rules

PAGE_URL = "https://www.example.com/login"


def pf(body, name, **kw):
    return page_features(f"<html><head></head><body>{body}</body></html>", PAGE_URL, **kw)[name]


def test_page_features_cover_page_list():
    # dnsrecord and sslfinal_state come from the network, not the HTML
    assert set(page_features("<html></html>", PAGE_URL)) == set(PAGE) - {"dnsrecord", "sslfinal_state"}


@pytest.mark.parametrize("html, expected", [
    ('<link rel="icon" href="https://evil.com/favicon.ico">', -1),
    ('<link rel="shortcut icon" href="/favicon.ico">', 1),
    ('<link rel="icon" href="https://cdn.example.com/f.ico">', 1),   # same registered domain
    ("", 1),
])
def test_favicon(html, expected):
    assert pf(html, "favicon") == expected


def imgs(n_ext, n_int):
    return ('<img src="https://other.com/a.png">' * n_ext) + ('<img src="/b.png">' * n_int)


@pytest.mark.parametrize("n_ext, n_int, expected", [
    (0, 10, 1), (2, 8, 1),     # 20% < 22%
    (3, 7, 0), (6, 4, 0),      # 30%, 60%
    (7, 3, -1),                # 70% > 61%
])
def test_request_url(n_ext, n_int, expected):
    assert pf(imgs(n_ext, n_int), "request_url") == expected


def test_request_url_counts_media_and_embeds():
    html = '<video src="https://x.com/v.mp4"></video><embed src="https://y.com/e.swf"><audio src="/a.mp3"></audio>'
    assert pf(html, "request_url") == -1   # 2/3 external


@pytest.mark.parametrize("anchors, expected", [
    (['/a', '/b', '/c', '/d'], 1),                                # 0%
    (['#', '/b', '/c', '/d'], 1),                                 # 25% < 31%
    (['#', 'javascript:void(0)', '/c', '/d'], 0),                 # 50%
    (['https://other.com/', '#skip', 'javascript:void(0)', '/d'], -1),  # 75%
])
def test_url_of_anchor(anchors, expected):
    html = "".join(f'<a href="{h}">x</a>' for h in anchors)
    assert pf(html, "url_of_anchor") == expected


@pytest.mark.parametrize("tags, expected", [
    ('<script src="/a.js"></script><link rel="stylesheet" href="/s.css">', 1),
    ('<script src="https://other.com/a.js"></script><link rel="stylesheet" href="/s.css">', 0),
    ('<script src="https://other.com/a.js"></script><meta property="og:image" content="https://x.com/i.png">', -1),
])
def test_links_in_tags(tags, expected):
    assert pf(tags, "links_in_tags") == expected


@pytest.mark.parametrize("form, expected", [
    ('<form action="about:blank"></form>', -1),
    ('<form action=""></form>', -1),
    ('<form></form>', -1),
    ('<form action="https://collector.evil.com/post"></form>', 0),
    ('<form action="/login"></form>', 1),
    ("", 1),
])
def test_sfh(form, expected):
    assert pf(form, "sfh") == expected


@pytest.mark.parametrize("form, expected", [
    ('<form action="mailto:thief@evil.com"></form>', -1),
    ('<form action="/send.php"><input name="x"></form>', 1),
])
def test_submitting_to_email(form, expected):
    assert pf(form, "submitting_to_email") == expected


@pytest.mark.parametrize("n, expected", [(0, 0), (1, 0), (2, 1), (5, 1)])
def test_redirect(n, expected):
    assert pf("", "redirect", n_redirects=n) == expected


@pytest.mark.parametrize("html, expected", [
    ('<a href="/x" onmouseover="window.status=\'http://paypal.com\'">x</a>', -1),
    ('<a href="/x" onmouseover="highlight(this)">x</a>', 1),
])
def test_on_mouseover(html, expected):
    assert pf(html, "on_mouseover") == expected


@pytest.mark.parametrize("html, expected", [
    ("<script>function c(event){ if (event.button==2) { return false; } }</script>", -1),
    ('<body oncontextmenu="return false">', -1),
    ("<script>console.log(1)</script>", 1),
])
def test_rightclick(html, expected):
    assert pf(html, "rightclick") == expected


@pytest.mark.parametrize("html, expected", [
    ('<iframe src="https://x.com" frameborder="0"></iframe>', -1),
    ('<iframe src="https://x.com" width="0" height="0"></iframe>', -1),
    ('<iframe src="https://x.com" style="display: none"></iframe>', -1),
    ('<iframe src="https://www.youtube.com/embed/x" width="560" height="315"></iframe>', 1),
    ("", 1),
])
def test_iframe(html, expected):
    assert pf(html, "iframe") == expected


# ------------------------------------------------------- extract() end to end

def fake_fetch(html="<html><body><a href='/a'>a</a></body></html>", final_url=None,
               cert_valid=True, n_redirects=0, is_html=True):
    def fetch(url):
        return {"html": html, "final_url": final_url or url, "n_redirects": n_redirects,
                "cert_valid": cert_valid, "is_html": is_html}
    return fetch


def test_extract_full_result():
    r = extract("https://www.example.com/", fetcher=fake_fetch(), resolver=lambda h: True)
    assert list(r["features"]) == LITE
    assert not r["partial"] and r["errors"] == []
    assert r["features"]["sslfinal_state"] == 1
    assert r["features"]["dnsrecord"] == 1


@pytest.mark.parametrize("url, cert_valid, final_url, expected", [
    ("https://example.com/", True, None, 1),
    ("https://example.com/", False, None, 0),
    ("http://example.com/", True, None, -1),
    ("http://example.com/", True, "https://example.com/", 1),   # upgraded by redirect
])
def test_extract_ssl_state(url, cert_valid, final_url, expected):
    r = extract(url, fetcher=fake_fetch(cert_valid=cert_valid, final_url=final_url),
                resolver=lambda h: True)
    assert r["features"]["sslfinal_state"] == expected


def test_extract_unresolvable_host_is_partial():
    def must_not_fetch(url):
        raise AssertionError("should not fetch a host that does not resolve")
    r = extract("https://no-such-host.example/", fetcher=must_not_fetch, resolver=lambda h: False)
    assert r["partial"]
    assert r["features"]["dnsrecord"] == -1
    assert all(r["available"][f] for f in URL_ONLY)
    assert not r["available"]["url_of_anchor"]


def test_extract_fetch_failure_keeps_url_features():
    def failing(url):
        raise requests.ConnectTimeout("timeout")
    r = extract("http://slow.example.com/", fetcher=failing, resolver=lambda h: True)
    assert r["partial"]
    assert r["features"]["sslfinal_state"] == -1    # plain http URL: known without fetching
    assert r["features"]["having_sub_domain"] == 0
    assert "page fetch failed" in r["errors"][0]


def test_to_vector_fills_unavailable_features():
    r = extract("https://no-such-host.example/", fetcher=fake_fetch(), resolver=lambda h: False)
    fill = {f: 7 for f in LITE}  # sentinel so we can see which were filled
    vec = to_vector(r, fill)
    assert len(vec) == len(LITE)
    for f, v in zip(LITE, vec):
        assert (v == 7) == (not r["available"][f])


@pytest.mark.parametrize("url", ["ftp://example.com/", "javascript:alert(1)", "http://"])
def test_extract_rejects_non_http(url):
    with pytest.raises(ValueError):
        extract(url, fetcher=fake_fetch(), resolver=lambda h: True)
