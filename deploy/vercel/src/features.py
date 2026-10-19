"""Feature lists and encoding notes.

Every feature in the UCI Phishing Websites dataset is coded with the
convention from Mohammad, Thabtah & McCluskey:

    -1 = phishing-like,  0 = suspicious,  1 = legitimate-like

with one exception: `redirect` only takes the values 0 and 1 in the data
(see REDIRECT_NOTE below).

FULL is all 30 features in dataset order. LITE is the subset that
`src/extract.py` can compute from a URL and its static HTML in a few seconds.
"""

FULL = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "sslfinal_state",
    "domain_registration_length",
    "favicon",
    "port",
    "https_token",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "abnormal_url",
    "redirect",
    "on_mouseover",
    "rightclick",
    "popupwindow",
    "iframe",
    "age_of_domain",
    "dnsrecord",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
]

# Computable from the URL string alone.
URL_ONLY = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "port",
    "https_token",
]

# Need the fetched page (or a DNS lookup / TLS handshake).
PAGE = [
    "sslfinal_state",
    "favicon",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "redirect",
    "on_mouseover",
    "rightclick",
    "iframe",
    "dnsrecord",
]

# Kept in dataset order so the lite vector is a column subset of the full one.
LITE = [f for f in FULL if f in set(URL_ONLY) | set(PAGE)]

# Left out of LITE: need WHOIS, discontinued services (Alexa, PageRank),
# search-engine queries, backlink counts, or blacklists, or (popupwindow)
# JavaScript execution.
EXCLUDED_FROM_LITE = [f for f in FULL if f not in LITE]

TARGET = "result"

REDIRECT_NOTE = (
    "The authors document `redirect` as a three-level rule (<=1 redirect "
    "legitimate, 2-3 suspicious, >=4 phishing), but the data only contains "
    "0 (88% of rows) and 1 (12%). Because 0 is the overwhelming majority, it "
    "must mean the common case (<=1 redirect), so the extractor maps "
    "<=1 redirect -> 0 and >=2 redirects -> 1."
)

# Short neutral label for what each feature checks, used by the API to
# explain predictions alongside the value's meaning.
DESCRIPTIONS = {
    "having_ip_address": "IP address as host",
    "url_length": "URL length",
    "shortining_service": "URL-shortening service",
    "having_at_symbol": "'@' in URL",
    "double_slash_redirecting": "'//' after the scheme",
    "prefix_suffix": "'-' in host name",
    "having_sub_domain": "Number of subdomains",
    "sslfinal_state": "HTTPS and certificate",
    "domain_registration_length": "Domain registration length",
    "favicon": "Favicon domain",
    "port": "Non-standard port",
    "https_token": "'https' inside host name",
    "request_url": "Images/media from other domains",
    "url_of_anchor": "Links pointing elsewhere or nowhere",
    "links_in_tags": "<meta>/<script>/<link> URLs on other domains",
    "sfh": "Form handler (where forms submit)",
    "submitting_to_email": "Form submits to e-mail",
    "abnormal_url": "Host name in WHOIS identity",
    "redirect": "Number of redirects",
    "on_mouseover": "Status-bar rewriting on mouseover",
    "rightclick": "Right-click disabled",
    "popupwindow": "Pop-up window with text fields",
    "iframe": "Invisible or borderless iframe",
    "age_of_domain": "Domain age",
    "dnsrecord": "Host name resolves in DNS",
    "web_traffic": "Web traffic rank",
    "page_rank": "PageRank",
    "google_index": "Indexed by Google",
    "links_pointing_to_page": "Links pointing to the page",
    "statistical_report": "On PhishTank/StopBadware top lists",
}
