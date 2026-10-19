"""Live feature extractor: URL -> lite feature vector.

Uses the dataset authors' encoding (-1 phishing-like, 0 suspicious,
1 legitimate-like). Rules follow "Phishing Websites Features" by Mohammad,
Thabtah & McCluskey (shipped with the UCI dataset); deviations and
approximations are marked APPROX in the comments and discussed in the report.

The work is split so it can be tested without a network:
  url_features(url)              pure, URL string only
  page_features(html, page_url)  pure, static HTML only (no JavaScript)
  fetch(url), dns_resolves(host) network, with short timeouts
  extract(url)                   puts it together

Features that could not be computed come back as None with
available[name] = False; `to_vector` fills them for the model.
"""
import ipaddress
import re
import socket
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import requests
import tldextract
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from src.features import LITE, PAGE, URL_ONLY

TIMEOUT = 5              # seconds, per network operation
MAX_BYTES = 2_000_000    # stop reading the page after 2 MB
MAX_REDIRECTS = 10
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Use the public-suffix snapshot bundled with tldextract: no network access,
# and the same answer every run.
_tld = tldextract.TLDExtract(suffix_list_urls=())
_dns_pool = ThreadPoolExecutor(max_workers=4)

SHORTENERS = {
    "bit.ly", "bitly.com", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "shorturl.at", "rebrand.ly",
    "tiny.cc", "lnkd.in", "db.tt", "qr.ae", "rb.gy", "t.ly", "v.gd", "x.co",
    "s.id", "short.io", "bl.ink", "soo.gd", "clck.ru", "u.to", "tr.im",
    "po.st", "mcaf.ee", "su.pr", "twurl.nl", "snipurl.com", "short.to",
    "budurl.com", "ping.fm", "post.ly", "just.as", "bkite.com", "snipr.com",
    "fic.kr", "loopt.us", "doiop.com", "kl.am", "wp.me", "rubyurl.com",
    "om.ly", "to.ly", "lnkd.in", "youtu.be", "tinyurl.at", "1url.com",
}


# ---------------------------------------------------------------- helpers

def hostname(url):
    return (urlparse(url).hostname or "").lower()


def registered_domain(host):
    """'login.mail.example.co.uk' -> 'example.co.uk'; IPs are returned as is."""
    e = _tld(host)
    return f"{e.domain}.{e.suffix}" if e.suffix else e.domain


def is_ip(host):
    """Dotted/IPv6 addresses, plus hex/decimal forms such as 0x58.0xCC.0xCA.0x62."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    parts = host.split(".")
    return 1 <= len(parts) <= 4 and all(re.fullmatch(r"0x[0-9a-f]+|\d+", p) for p in parts)


def _bucket(share, low, high):
    """share < low -> 1 (legitimate); low <= share <= high -> 0; else -1."""
    if share < low:
        return 1
    return 0 if share <= high else -1


# ------------------------------------------------------- URL-only features

def url_features(url):
    host = hostname(url)
    parsed = urlparse(url)
    f = {}
    f["having_ip_address"] = -1 if is_ip(host) else 1
    f["url_length"] = 1 if len(url) < 54 else (0 if len(url) <= 75 else -1)
    f["shortining_service"] = -1 if (host in SHORTENERS or registered_domain(host) in SHORTENERS) else 1
    f["having_at_symbol"] = -1 if "@" in url else 1
    # "//" belongs at 1-based position 6 (http) or 7 (https); later means a redirect path.
    f["double_slash_redirecting"] = -1 if url.rfind("//") + 1 > 7 else 1
    # APPROX: the authors say "domain name part"; we check the whole host name.
    f["prefix_suffix"] = -1 if "-" in host else 1
    f["having_sub_domain"] = _sub_domain(host)
    try:
        port = parsed.port
        f["port"] = -1 if port is not None and port not in (80, 443) else 1
    except ValueError:  # malformed port
        f["port"] = -1
    f["https_token"] = -1 if "https" in host else 1
    return f


def _sub_domain(host):
    """The authors drop 'www.' and the country-code TLD, then count dots:
    1 dot -> 1, 2 dots -> 0, more -> -1. 'hud.ac.uk' -> 'hud.ac' has 1 dot,
    so this is equivalent to counting subdomain labels below the registered
    domain: none -> 1, one -> 0, more -> -1."""
    if is_ip(host):
        return -1  # an IP host has 3 dots and no suffix to drop
    sub = _tld(host).subdomain
    if sub == "www":
        sub = ""
    elif sub.startswith("www."):
        sub = sub[4:]
    labels = [label for label in sub.split(".") if label]
    return 1 if not labels else (0 if len(labels) == 1 else -1)


# ------------------------------------------------------ page (HTML) features

def page_features(html, page_url, n_redirects=0):
    """Features from static HTML. `page_url` is the final URL after redirects."""
    soup = BeautifulSoup(html, "html.parser")
    own = registered_domain(hostname(page_url))

    def external(link):
        link = (link or "").strip()
        if not link or link.lower().startswith(("data:", "javascript:", "mailto:")):
            return False
        host = hostname(urljoin(page_url, link))
        return bool(host) and registered_domain(host) != own

    f = {}

    icons = [t.get("href") for t in soup.find_all("link", href=True)
             if any("icon" in r.lower() for r in (t.get("rel") or []))]
    f["favicon"] = -1 if any(external(h) for h in icons) else 1

    media = [t.get("src") for t in soup.find_all(["img", "audio", "video", "source", "embed"], src=True)]
    media += [t.get("data") for t in soup.find_all("object", data=True)]
    f["request_url"] = _bucket(_share(media, external), 0.22, 0.61)

    def unsafe_anchor(a):
        href = (a.get("href") or "").strip()
        return (not href or href.startswith("#") or href.lower().startswith("javascript:")
                or external(href))
    anchors = soup.find_all("a")
    share = sum(map(unsafe_anchor, anchors)) / len(anchors) if anchors else 0.0
    f["url_of_anchor"] = _bucket(share, 0.31, 0.67)

    tag_links = [t.get("src") for t in soup.find_all("script", src=True)]
    tag_links += [t.get("href") for t in soup.find_all("link", href=True)]
    tag_links += [t.get("content") for t in soup.find_all("meta", content=True)
                  if re.match(r"(https?:)?//", t.get("content", "").strip(), re.I)]
    f["links_in_tags"] = _bucket(_share(tag_links, external), 0.17, 0.81)

    f["sfh"] = 1
    f["submitting_to_email"] = 1
    for form in soup.find_all("form"):
        action = (form.get("action") or "").strip()
        # A form without an action attribute submits nowhere useful in static
        # HTML, so it counts as empty, like action="".
        if action.lower() in ("", "about:blank"):
            f["sfh"] = -1
        elif external(action) and f["sfh"] == 1:
            f["sfh"] = 0
        if action.lower().startswith("mailto:") or "mail(" in str(form).lower():
            f["submitting_to_email"] = -1

    # See features.REDIRECT_NOTE: the data only has 0 (<=1 redirect) and 1 (more).
    f["redirect"] = 0 if n_redirects <= 1 else 1

    mouseover = any("status" in t.get("onmouseover", "").lower()
                    for t in soup.find_all(attrs={"onmouseover": True}))
    scripts = " ".join(s.get_text() for s in soup.find_all("script")).lower()
    mouseover = mouseover or ("onmouseover" in scripts and "window.status" in scripts)
    f["on_mouseover"] = -1 if mouseover else 1

    # The authors search for event.button==2. APPROX: we also count the
    # modern equivalent, oncontextmenu="return false".
    rightclick = (re.search(r"event\.button\s*===?\s*2", html)
                  or re.search(r"oncontextmenu\s*=\s*[\"']?\s*return\s+false", html, re.I))
    f["rightclick"] = -1 if rightclick else 1

    f["iframe"] = -1 if any(_hidden_iframe(t) for t in soup.find_all("iframe")) else 1
    return f


def _share(links, external):
    links = [link for link in links if link and link.strip()]
    return sum(map(external, links)) / len(links) if links else 0.0


def _hidden_iframe(tag):
    style = tag.get("style", "").replace(" ", "").lower()
    return (tag.get("frameborder", "").strip() == "0"
            or tag.get("width", "").strip() in ("0", "0px")
            or tag.get("height", "").strip() in ("0", "0px")
            or any(s in style for s in ("display:none", "visibility:hidden",
                                        "border:0", "border:none")))


# --------------------------------------------------------------- network

def dns_resolves(host):
    """True / False, or None if the lookup timed out or failed oddly."""
    if is_ip(host):
        return True
    future = _dns_pool.submit(socket.getaddrinfo, host, None)
    try:
        future.result(timeout=TIMEOUT)
        return True
    except socket.gaierror:
        return False
    except (TimeoutError, OSError, UnicodeError):
        return None


def fetch(url):
    """Download the page (static HTML only, never executes JavaScript).

    Returns {"html", "final_url", "n_redirects", "cert_valid", "is_html"}.
    If the certificate does not validate, the page is fetched again without
    verification so the HTML features can still be computed.
    """
    session = requests.Session()
    session.max_redirects = MAX_REDIRECTS
    kwargs = {"timeout": TIMEOUT, "stream": True, "headers": {"User-Agent": USER_AGENT}}
    cert_valid = True
    try:
        resp = session.get(url, **kwargs)
    except requests.exceptions.SSLError:
        cert_valid = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            resp = session.get(url, verify=False, **kwargs)
    chunks, size, deadline = [], 0, time.monotonic() + 2 * TIMEOUT
    for chunk in resp.iter_content(64 * 1024):
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_BYTES or time.monotonic() > deadline:
            break
    resp.close()
    content_type = resp.headers.get("Content-Type", "").lower()
    return {
        "html": b"".join(chunks)[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace"),
        "final_url": resp.url,
        "n_redirects": len(resp.history),
        "cert_valid": cert_valid,
        "is_html": "html" in content_type or not content_type,
    }


# ------------------------------------------------------------- put together

def extract(url, fetcher=fetch, resolver=dns_resolves):
    """URL -> {"features": {name: value or None}, "available": {...},
    "partial": bool, "final_url", "errors": [...]}, in LITE order."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https") or not hostname(url):
        raise ValueError("only http(s) URLs with a host name are supported")
    values = dict.fromkeys(LITE)
    values.update(url_features(url))
    errors = []
    final_url = url

    resolves = resolver(hostname(url))
    if resolves is None:
        errors.append("DNS lookup timed out")
    else:
        values["dnsrecord"] = 1 if resolves else -1

    page = None
    if resolves is False:
        errors.append("host name does not resolve; page not fetched")
    else:
        try:
            page = fetcher(url)
        except requests.RequestException as e:
            errors.append(f"page fetch failed: {type(e).__name__}")

    if page is not None:
        final_url = page["final_url"]
        # APPROX of the authors' rule (trusted issuer and certificate age >= 1 year):
        # HTTPS with a validating certificate -> 1, certificate error -> 0, no HTTPS -> -1.
        if urlparse(final_url).scheme == "https":
            values["sslfinal_state"] = 1 if page["cert_valid"] else 0
        else:
            values["sslfinal_state"] = -1
        if page["is_html"]:
            values.update(page_features(page["html"], final_url, page["n_redirects"]))
        else:
            values["redirect"] = 0 if page["n_redirects"] <= 1 else 1
            errors.append("response is not HTML; HTML features unavailable")
    elif scheme == "http":
        values["sslfinal_state"] = -1  # the URL itself does not use HTTPS

    available = {f: values[f] is not None for f in LITE}
    return {
        "url": url,
        "final_url": final_url,
        "features": values,
        "available": available,
        "partial": not all(available.values()),
        "errors": errors,
    }


def to_vector(result, fill_values):
    """Feature values in LITE order, with unavailable ones replaced by
    `fill_values` (the training-set mode of each feature)."""
    return [result["features"][f] if result["available"][f] else fill_values[f] for f in LITE]


assert set(URL_ONLY) | set(PAGE) == set(LITE)

if __name__ == "__main__":
    import json
    import sys

    for u in sys.argv[1:]:
        print(json.dumps(extract(u), indent=2))
