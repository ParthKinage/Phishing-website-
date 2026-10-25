"""Screenshots of the running web app for the report (results/screenshot_*.png).

Needs the local API running (start_app.bat) and internet access for the
live site. Uses the installed Microsoft Edge through Playwright.

Run: python -m report_build.screenshots
"""
from playwright.sync_api import sync_playwright

from src.train import RESULTS_DIR

LIVE = "https://phishing-website-detector-wheat.vercel.app/"
LOCAL = "http://127.0.0.1:8000/"

SHOTS = [
    # (file, app page, URL to check)
    ("screenshot_live_legit.png", LIVE, "https://www.wikipedia.org/"),
    ("screenshot_local_phishing.png", LOCAL, "http://127.0.0.1:8000/demo/account-verify"),
    ("screenshot_local_modern.png", LOCAL, "http://127.0.0.1:8000/demo/modern-login"),
]


def check(page, app, url):
    page.goto(app, wait_until="networkidle")
    page.fill("#url", url)
    page.click("#go")
    page.wait_for_selector(".badge", timeout=90_000)
    page.wait_for_timeout(400)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=1.5,
                                color_scheme="light")
        for name, app, url in SHOTS:
            check(page, app, url)
            page.locator("main").screenshot(path=str(RESULTS_DIR / name))
            print("saved", name, "-", page.locator(".badge").inner_text(), page.locator(".prob").inner_text())
        # The fake phishing page itself, as a visitor would see it.
        page.goto(LOCAL + "demo/account-verify", wait_until="networkidle")
        page.screenshot(path=str(RESULTS_DIR / "screenshot_demo_page.png"))
        print("saved screenshot_demo_page.png")
        browser.close()


if __name__ == "__main__":
    main()
