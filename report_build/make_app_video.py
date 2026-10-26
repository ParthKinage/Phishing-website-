"""App-only walkthrough video: no slides, just the app being used.

Reuses the recording engine of make_video.py (offline narration, captions,
Edge + Playwright recording, ffmpeg assembly) with browsing-only scenes.

Needs: the local API running (start_app.bat) and internet access.
Run:   python -m report_build.make_app_video <work_dir> <output.mp4>
"""
import sys

from report_build import make_video as mv

LIVE, LOCAL = mv.LIVE, mv.LOCAL
PHISH_URL = "http://127.0.0.1:8000/demo/account-verify"

SCENES = [
    ("intro", "demo", [
        "This video shows how our Phishing Website Detector works, using the app itself.",
        "This is the public version of the app, hosted on Vercel. You paste any website address into this box,",
        "and click Check. Let's try it.",
    ]),
    ("legit", "demo", [
        "We enter the address of a real website, Wikipedia, and click Check.",
        "Behind the scenes, our server downloads the page's HTML, without running any of its scripts, and checks "
        "the web address, its DNS record and its HTTPS certificate.",
        "From these it computes twenty one warning-sign features, and our trained XGBoost model turns them into a "
        "phishing probability. Here it is almost zero: likely legitimate.",
        "The top reasons explain the decision. They are SHAP values: green arrows push towards legitimate, and red "
        "arrows push towards phishing.",
        "Opening all twenty one features shows each value. Minus one means phishing-like, zero means suspicious, and "
        "one means legitimate-like, next to how much each one contributed.",
    ]),
    ("validation", "demo", [
        "The app only accepts normal web addresses. If we enter an F T P address, it is rejected with a clear message.",
    ]),
    ("api", "demo", [
        "The web page is only a front end. All the work happens in our FastAPI service.",
        "Its health endpoint shows which model is loaded: XGBoost, with twenty one features.",
        "And this is the API documentation. The predict endpoint takes a web address and returns the verdict, the "
        "probability, every feature and the reasons, as JSON. Our Chrome extension uses the same endpoint.",
    ]),
    ("fake", "demo", [
        "To show a phishing example safely, we built a harmless fake phishing page. It runs only on our own "
        "computer, and its form sends nothing.",
        "It copies classic phishing tricks: an urgent request to verify your account, a password form that submits "
        "nowhere, links that lead nowhere, and right click disabled.",
        "It is also served from a plain IP address without HTTPS, like many phishing kits.",
    ]),
    ("fake_check", "demo", [
        "Now we check that page in the app running on our computer.",
        "The verdict is likely phishing, with ninety nine point five percent probability.",
        "The red reasons show why: most links lead nowhere, there is no HTTPS, images come from other sites, and "
        "right click is disabled.",
    ]),
    ("invoice", "demo", [
        "Here is a second fake page: an invoice that asks for card details, with a form that would email them to "
        "the attacker.",
        "The app flags it as phishing too, with ninety nine point six percent probability.",
    ]),
    ("modern", "demo", [
        "Finally, an honest limitation. This fake login page is built like a modern phishing kit: clean HTML, with "
        "everything on one domain.",
        "The app rates it as legitimate, so it is missed. Our model learned from data collected around twenty "
        "fifteen, and modern phishing pages look cleaner. That is our main area for future work.",
    ]),
    ("popup", "demo", [
        "The same check is built into our Chrome extension. When you click its icon, it sends the current tab's "
        "address to our local service.",
        "Here the current tab is the fake account verification page, and the popup warns straight away: likely "
        "phishing, with the top reasons.",
    ]),
    ("outro", "demo", [
        "So in short: the app takes a web address, computes twenty one warning signs from the address and the page, "
        "scores them with XGBoost, and explains the result.",
        "Thank you for watching.",
    ]),
]


def check(page, app, url):
    page.goto(app, wait_until="networkidle")
    mv.type_slowly(page, "#url", url)
    page.wait_for_timeout(400)
    page.click("#go")


def d_intro(page, at):
    page.goto(LIVE, wait_until="networkidle")
    at(1)
    page.hover("#url")
    at(2)
    page.hover("#go")


def d_legit(page, at):
    check(page, LIVE, "https://www.wikipedia.org/")
    page.wait_for_selector(".badge", timeout=60_000)
    at(3)
    page.hover("ul.reasons")
    at(4)
    page.click("details summary")
    page.wait_for_timeout(800)
    page.mouse.wheel(0, 260)
    page.wait_for_timeout(3500)
    page.mouse.wheel(0, 260)


def d_validation(page, at):
    check(page, LIVE, "ftp://example.com/file")
    page.wait_for_selector(".error", timeout=30_000)


def d_api(page, at):
    page.goto(LIVE + "health", wait_until="load")
    at(2)
    page.goto(LIVE + "docs", wait_until="networkidle")
    page.wait_for_selector(".opblock", timeout=30_000)
    page.wait_for_timeout(1200)
    page.locator(".opblock-summary", has_text="/predict").first.click()
    page.wait_for_timeout(1500)
    page.mouse.wheel(0, 300)


def show_page(page, path, scroll=True):
    page.goto(LOCAL + path, wait_until="networkidle")
    if scroll:
        page.wait_for_timeout(2500)
        page.mouse.wheel(0, 200)


def d_fake(page, at):
    show_page(page, "demo/account-verify")
    at(1)
    page.hover("button:has-text('Verify')")
    page.wait_for_timeout(1500)
    page.hover("a:has-text('Help')")
    at(2)
    page.mouse.wheel(0, -200)


def d_fake_check(page, at):
    check(page, LOCAL, PHISH_URL)
    page.wait_for_selector(".badge", timeout=60_000)
    at(2)
    page.hover("ul.reasons")


def d_invoice(page, at):
    show_page(page, "demo/invoice")
    at(1)
    check(page, LOCAL, "http://127.0.0.1:8000/demo/invoice")
    page.wait_for_selector(".badge", timeout=60_000)


def d_modern(page, at):
    show_page(page, "demo/modern-login", scroll=False)
    at(1)
    check(page, LOCAL, "http://127.0.0.1:8000/demo/modern-login")
    page.wait_for_selector(".badge", timeout=60_000)


def d_popup(page, at):
    """The real extension popup (popup.html + popup.js), with only chrome.tabs
    stubbed to report the fake phishing page as the current tab."""
    ext = mv.ROOT / "extension"
    stub = ("<script>window.chrome={tabs:{query:async()=>[{url:'%s'}]}};</script>" % PHISH_URL)
    popup = (ext / "popup.html").read_text(encoding="utf-8").replace(
        '<script src="popup.js"></script>', stub + '<script src="popup.js"></script>')
    frame = """<!doctype html><html><body style="margin:0;height:100vh;background:#e9e8e4;
      font-family:'Segoe UI',sans-serif;display:flex;align-items:center;justify-content:center;gap:60px">
      <div style="color:#52514e;font-size:22px;max-width:380px;line-height:1.5">
        <div style="font-size:15px;letter-spacing:.05em;text-transform:uppercase;color:#2a78d6">Chrome extension</div>
        <div style="font-size:30px;font-weight:700;color:#0b0b0b;margin:6px 0 12px">Phishing check popup</div>
        Current tab:<br><code style="font-size:16px">%s</code></div>
      <iframe id="popup" style="width:360px;height:470px;border:1px solid #d6d5d0;border-radius:12px;
        background:#fcfcfb;box-shadow:0 10px 36px rgba(0,0,0,.18)"></iframe></body></html>""" % PHISH_URL
    files = {"/frame": ("text/html", frame), "/popup.html": ("text/html", popup),
             "/popup.js": ("application/javascript", (ext / "popup.js").read_text(encoding="utf-8"))}

    def handler(route):
        path = "/" + route.request.url.split("localhost:8001/", 1)[1].split("?")[0]
        ctype, body = files[path]
        route.fulfill(status=200, content_type=ctype, body=body)

    page.route("http://localhost:8001/**", handler)
    page.goto("http://localhost:8001/frame", wait_until="networkidle")
    at(1)
    page.evaluate("document.getElementById('popup').src = '/popup.html'")


def d_outro(page, at):
    page.goto(LIVE, wait_until="networkidle")
    mv.type_slowly(page, "#url", "https://www.python.org/")
    page.click("#go")
    page.wait_for_selector(".badge", timeout=60_000)


DEMOS = {"intro": d_intro, "legit": d_legit, "validation": d_validation, "api": d_api, "fake": d_fake,
         "fake_check": d_fake_check, "invoice": d_invoice, "modern": d_modern, "popup": d_popup,
         "outro": d_outro}


def main(work, output):
    mv.SCENES = SCENES
    mv.DEMOS = DEMOS
    # No slides in this video; the engine's warm-up page just needs some HTML.
    mv.slides = lambda: {"title": "<!doctype html><html><body></body></html>"}
    mv.main(work, output)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
