"""Builds the narrated project walkthrough video (MP4).

Each scene is either a slide (HTML) or a live demo of the web application,
recorded in Microsoft Edge with Playwright while captions show the current
sentence. Narration uses the offline Windows text-to-speech voice. ffmpeg
(from imageio-ffmpeg) trims each recording to its narration and joins them.

Needs: the local API running (start_app.bat) and internet access.
Run:   python -m report_build.make_video <work_dir> <output.mp4>
The narration text is also written to <work_dir>/narration.txt.
"""
import base64
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LIVE = "https://phishing-website-detector-wheat.vercel.app/"
LOCAL = "http://127.0.0.1:8000/"
W, H = 1280, 720
VOICE = "Microsoft Zira Desktop"
GAP = 0.35          # silence between sentences (s)
TAIL = 0.8          # extra time at the end of each scene (s)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# Optional: re-record only some scenes, e.g. VIDEO_ONLY=title,live (others re-use earlier clips)
ONLY = {x for x in __import__("os").environ.get("VIDEO_ONLY", "").split(",") if x}

MEMBERS = [("Parth Kinage", "24101B0036"), ("Ronak Boddu", "24101B0044"), ("Vedangi Pokharkar", "24101B0051")]
GUIDE = "Dr. Kanchan Dhuri"


# --------------------------------------------------------------- slides

CSS = """
* { box-sizing: border-box; margin: 0; }
body { width: 1280px; height: 720px; overflow: hidden; background: #fcfcfb; color: #0b0b0b;
       font-family: 'Segoe UI', system-ui, sans-serif; }
.slide { position: absolute; inset: 0; padding: 44px 64px 120px; }
h1 { font-size: 38px; margin-bottom: 6px; }
.kicker { color: #2a78d6; font-weight: 600; font-size: 17px; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 4px; }
.sub { color: #52514e; font-size: 20px; margin-bottom: 22px; }
ul { font-size: 22px; line-height: 1.5; padding-left: 26px; }
li { margin-bottom: 8px; }
b { color: #0b0b0b; }
.cols { display: flex; gap: 34px; align-items: flex-start; }
.cols > div { flex: 1; }
img.fig { max-width: 100%; max-height: 470px; display: block; margin: 0 auto; border-radius: 6px; }
.card { background: #fff; border: 1px solid #e4e3df; border-radius: 12px; padding: 16px 20px; }
.card h3 { font-size: 21px; margin-bottom: 6px; }
.card p { font-size: 17px; color: #52514e; line-height: 1.4; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.stat { font-size: 44px; font-weight: 700; color: #2a78d6; }
.stat.o { color: #eb6834; }
.statlabel { font-size: 17px; color: #52514e; }
table { border-collapse: collapse; font-size: 19px; width: 100%; }
th, td { padding: 7px 10px; border-bottom: 1px solid #e4e3df; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: #52514e; font-weight: 600; font-size: 16px; }
tr.best td { font-weight: 700; }
.foot { position: absolute; left: 64px; right: 64px; bottom: 96px; font-size: 13px; color: #8a897f;
        display: flex; justify-content: space-between; }
.title { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; height: 100%; }
.title h1 { font-size: 46px; max-width: 1000px; }
"""


def page_html(body, n=None):
    foot = (f'<div class="foot"><span>Phishing Website Detection using Machine Learning</span>'
            f'<span>{n}</span></div>') if n else ""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
           f"<body><div class='slide'>{body}</div>{foot}</body></html>"


def img(name, max_h=470):
    data = base64.b64encode((RES / name).read_bytes()).decode()
    return f"<img class='fig' style='max-height:{max_h}px' src='data:image/png;base64,{data}'>"


def results_table(fs):
    df = pd.read_csv(RES / f"metrics_{fs}.csv")
    labels = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "svm": "SVM (RBF)", "xgboost": "XGBoost"}
    best = df.accuracy.idxmax()
    rows = "".join(
        f"<tr class='{'best' if i == best else ''}'><td>{labels[r.model]}</td><td>{100*r.accuracy:.2f}</td>"
        f"<td>{100*r.precision:.2f}</td><td>{100*r.recall:.2f}</td><td>{100*r.f1:.2f}</td><td>{r.roc_auc:.4f}</td></tr>"
        for i, r in df.iterrows())
    return ("<table><tr><th>Model</th><th>Accuracy %</th><th>Precision %</th><th>Recall %</th><th>F1 %</th>"
            f"<th>ROC-AUC</th></tr>{rows}</table>")


def slides():
    logo = base64.b64encode((Path(sys.argv[1]) / "image1.png").read_bytes()).decode()
    members = "<br>".join(f"{n} &nbsp;({r})" for n, r in MEMBERS)
    S = {}
    S["title"] = page_html(f"""
      <div class="title">
        <img src="data:image/png;base64,{logo}" style="height:70px;margin-bottom:26px">
        <div class="kicker">TY B.Tech INFT Sem V &middot; Machine Learning Project</div>
        <h1 style="margin:10px 0 22px">Phishing Website Detection using Machine Learning</h1>
        <div class="sub" style="line-height:1.6">{members}</div>
        <div class="sub">Project guide: <b>{GUIDE}</b></div>
      </div>""")
    S["problem"] = page_html("""
      <div class="kicker">The problem</div><h1>What is phishing?</h1>
      <div class="cols" style="margin-top:20px"><div><ul>
        <li>A <b>fake website</b> that imitates a bank, e-mail or shopping site</li>
        <li>Goal: steal <b>passwords and card details</b></li>
        <li>New sites appear every day and live only a short time, so <b>blocklists lag behind</b></li>
        <li>Idea: learn the <b>warning signs</b> of phishing pages with machine learning</li>
      </ul></div>
      <div style="flex:.8"><div class="card"><h3>Typical warning signs</h3><p>
        No valid HTTPS certificate<br>An IP address instead of a domain name<br>Most links lead nowhere or to other sites<br>
        A login form that submits to a strange place<br>A '-' or many subdomains in the host name</p></div></div></div>""", 2)
    S["dataset"] = page_html(f"""
      <div class="kicker">Data &middot; exploratory data analysis</div><h1>UCI Phishing Websites dataset</h1>
      <div class="cols" style="margin-top:14px">
        <div style="flex:.55"><div class="grid2">
          <div class="card"><div class="stat">11,055</div><div class="statlabel">websites</div></div>
          <div class="card"><div class="stat">30</div><div class="statlabel">features, coded -1 / 0 / 1</div></div>
          <div class="card"><div class="stat o">44.3%</div><div class="statlabel">are phishing</div></div>
          <div class="card"><div class="stat o">47%</div><div class="statlabel">are exact duplicate rows</div></div>
        </div></div>
        <div>{img('eda_class_duplicates.png', 300)}</div></div>""", 3)
    S["eda"] = page_html(f"""
      <div class="kicker">Exploratory data analysis</div><h1>Which signals separate the classes?</h1>
      {img('eda_phishing_rate.png', 440)}""", 4)
    S["concepts"] = page_html("""
      <div class="kicker">Machine learning concepts</div><h1>How the models are trained and tested</h1>
      <div class="grid2" style="margin-top:18px">
        <div class="card"><h3>Supervised binary classification</h3><p>Learn from labelled examples:
          30 features &rarr; phishing (1) or legitimate (0).</p></div>
        <div class="card"><h3>Stratified train / test split</h3><p>80% (8,844) to train, 20% (2,211) held back
          and used only once at the end.</p></div>
        <div class="card"><h3>5-fold cross-validation + grid search</h3><p>Every hyper-parameter combination is scored
          on 5 rotating validation folds; the best is refit on all training data.</p></div>
        <div class="card"><h3>Evaluation metrics</h3><p>Accuracy, precision, recall, F1-score for the phishing class,
          and ROC-AUC. Recall matters most: a missed phishing site is costly.</p></div>
      </div>""", 5)
    S["models"] = page_html("""
      <div class="kicker">Four algorithms</div><h1>Models compared</h1>
      <div class="grid2" style="margin-top:18px">
        <div class="card"><h3>Decision Tree</h3><p>A single tree of if-then questions. Easy to read, but it
          overfits.</p></div>
        <div class="card"><h3>Random Forest &middot; bagging</h3><p>Hundreds of trees trained on random samples and
          random feature subsets; their votes are averaged.</p></div>
        <div class="card"><h3>SVM with RBF kernel</h3><p>Finds the widest margin between the classes after a non-linear
          kernel transformation.</p></div>
        <div class="card"><h3>XGBoost &middot; boosting</h3><p>Adds small trees one at a time, each one correcting the
          errors of the previous trees, with regularisation.</p></div>
      </div>""", 6)
    S["results"] = page_html(f"""
      <div class="kicker">Results on the held-out test set</div><h1>Full model (all 30 features)</h1>
      <div style="margin-top:14px">{results_table('full')}</div>
      <div class="sub" style="margin-top:18px">Lite model (21 live-computable features):</div>
      {results_table('lite')}""", 7)
    S["honest"] = page_html(f"""
      <div class="kicker">Honest evaluation</div><h1>What happens without duplicate rows?</h1>
      <div class="cols" style="margin-top:20px"><div><ul>
        <li>Almost half of the rows are <b>exact copies</b>; a copy can land in both train and test sets</li>
        <li>We removed duplicates, re-split and re-tested every model</li>
        <li>F1-score drops by <b>2.3 to 4.0 points</b>: a more realistic estimate</li>
        <li><b>XGBoost</b> is the most robust (94.9% accuracy on unique data)</li>
      </ul></div><div style="flex:.9">{img('confusion_full.png', 230)}</div></div>""", 8)
    S["explain"] = page_html(f"""
      <div class="kicker">Explainability</div><h1>Which features matter? (SHAP)</h1>
      <div class="cols"><div style="flex:.9">{img('shap_summary.png', 470)}</div><div><ul style="font-size:20px">
        <li><b>Valid HTTPS</b> and <b>where links point</b> dominate every model</li>
        <li>Permutation importance and SHAP agree (mean rank correlation 0.82)</li>
        <li>About ten features barely matter</li>
        <li>15 features reach 97.0%, only 0.7 points below all 30</li></ul></div></div>""", 9)
    S["gap"] = page_html(f"""
      <div class="kicker">From research to a real tool</div><h1>Full vs deployable features</h1>
      <div class="cols"><div><ul style="font-size:20px">
        <li>9 features cannot be computed today (Alexa rank, PageRank, Google index, WHOIS...)</li>
        <li>The <b>lite</b> model uses the other 21: about <b>3 points</b> lower</li>
        <li>No single missing feature explains the gap; together the <b>reputation</b> features do</li>
      </ul></div><div>{img('lite_gap.png', 380)}</div></div>""", 10)
    S["arch"] = page_html(f"""
      <div class="kicker">System architecture</div><h1>How the application works</h1>
      {img('architecture.png', 470)}""", 11)
    S["extension"] = page_html("""
      <div class="kicker">Live demo</div><h1>Chrome extension</h1>
      <div class="cols" style="margin-top:16px"><div><ul style="font-size:21px">
        <li>One click checks the <b>current tab</b></li>
        <li>Calls our local API at 127.0.0.1</li>
        <li>Permissions: only <b>activeTab</b> and the local API</li>
        <li>No background data collection</li></ul></div>
      <div style="flex:.75"><iframe id="popup" style="width:360px;height:420px;border:1px solid #e4e3df;
        border-radius:12px;background:#fcfcfb;box-shadow:0 8px 30px rgba(0,0,0,.12)"></iframe></div></div>""", 15)
    S["deploy"] = page_html("""
      <div class="kicker">Deployment &amp; engineering</div><h1>Deployment and safety</h1>
      <div class="grid2" style="margin-top:18px">
        <div class="card"><h3>FastAPI service</h3><p>/predict returns verdict, probability, all 21 features
          and the top 5 SHAP reasons.</p></div>
        <div class="card"><h3>Public website (Vercel)</h3><p>phishing-website-detector-wheat.vercel.app,
          serving the model with XGBoost's own engine.</p></div>
        <div class="card"><h3>Safety in public mode</h3><p>Blocks private and internal addresses (SSRF), hides the
          fake test pages, limits each client to 20 checks per minute.</p></div>
        <div class="card"><h3>Quality</h3><p>118 automated tests; fully reproducible pipeline;
          code on GitHub: ParthKinage/Phishing-website-</p></div>
      </div>""", 16)
    S["end"] = page_html("""
      <div class="kicker">Conclusion</div><h1>Summary and future scope</h1>
      <div class="cols" style="margin-top:18px"><div><ul style="font-size:20px">
        <li>Ensembles reach <b>97.6%</b> accuracy; XGBoost has the best recall and ROC-AUC</li>
        <li>Deployable lite model: <b>94.6%</b>, explained with SHAP</li>
        <li>Limitation: 2015 data; modern phishing sites with valid HTTPS can be missed</li>
      </ul></div><div><ul style="font-size:20px">
        <li>Future: fresh data from the same extractor</li>
        <li>Modern reputation signals (domain age, Tranco rank)</li>
        <li>Render JavaScript and compare page visuals</li></ul></div></div>
      <div class="sub" style="text-align:center;margin-top:40px;font-size:30px;color:#0b0b0b"><b>Thank you!</b></div>""", 17)
    return S


# ------------------------------------------------------------- narration

SCENES = [
    ("title", "slide", [
        "Hello! We are Parth Kinage, Ronak Boddu and Vedangi Pokharkar, from T Y B Tech, Information Technology.",
        "Our machine learning project, guided by Doctor Kanchan Dhuri, is Phishing Website Detection using Machine Learning.",
    ]),
    ("problem", "slide", [
        "A phishing website is a fake site that imitates a bank, an email provider or a shop, to steal passwords and card details.",
        "New phishing sites appear every day and disappear quickly, so blocklists of known bad sites are always behind.",
        "Our idea is to let a machine learning model learn the warning signs of phishing pages, so it can judge sites it has never seen.",
    ]),
    ("dataset", "slide", [
        "We used the U C I Phishing Websites dataset: eleven thousand and fifty five websites, each described by thirty features.",
        "Every feature is coded as minus one for phishing-like, zero for suspicious, or one for legitimate-like. About forty four percent of the sites are phishing.",
        "During exploratory data analysis we found that forty seven percent of the rows are exact duplicates. We come back to this later, because it affects how honest the results are.",
    ]),
    ("eda", "slide", [
        "This chart shows the share of phishing sites for each value of eight key features.",
        "For example, when most of a page's links point nowhere or to other websites, ninety nine percent of the sites are phishing. A valid HTTPS certificate is a strong sign of a legitimate site.",
    ]),
    ("concepts", "slide", [
        "This is a supervised binary classification problem: the model learns from labelled examples to predict phishing or legitimate.",
        "We split the data into eighty percent for training and twenty percent for testing, keeping the class balance. The test set is used only once, at the very end.",
        "To choose each model's settings, we used grid search with five-fold cross-validation on the training data.",
        "We measure accuracy, precision, recall, F1-score and ROC A U C. Recall on phishing matters most, because a missed phishing site is the costly mistake.",
    ]),
    ("models", "slide", [
        "We compared four algorithms. A decision tree asks a series of if-then questions.",
        "A random forest averages hundreds of trees trained on random samples, which is called bagging.",
        "A support vector machine with an R B F kernel finds the widest margin between the two classes.",
        "And XGBoost adds small trees one after another, each correcting the mistakes of the previous ones. This is called gradient boosting.",
    ]),
    ("results", "slide", [
        "Here are the test results. With all thirty features, random forest, S V M and XGBoost all reach about ninety seven point six percent accuracy.",
        "XGBoost has the best phishing recall and the best ROC A U C. The single decision tree is about one point lower.",
        "The lite model uses only the twenty one features we can compute for a live website, and reaches about ninety four point six percent.",
    ]),
    ("honest", "slide", [
        "Because almost half of the rows are duplicates, a copy of a test row can also appear in the training data. So we removed all duplicates and tested again.",
        "Every model lost between two and four F1 points. That is a more realistic estimate for new websites, and XGBoost held up best.",
    ]),
    ("explain", "slide", [
        "To understand the models, we used built-in importance, permutation importance and SHAP values.",
        "They agree: a valid HTTPS certificate, and where the page's links point, are by far the most important signals.",
        "An ablation study showed that fifteen well-chosen features are almost as good as all thirty.",
    ]),
    ("gap", "slide", [
        "Nine of the original features can no longer be computed, for example Alexa traffic rank and Google PageRank.",
        "Our analysis shows that no single missing feature explains the three point gap, but together the site reputation features explain all of it.",
    ]),
    ("arch", "slide", [
        "This is how the application works. Offline, we train and evaluate the models and save the best lite model, XGBoost.",
        "Online, a user sends a web address. Our FastAPI service downloads the page, computes the twenty one features, gets a phishing probability from XGBoost, and explains it with SHAP.",
    ]),
    ("live", "demo", [
        "Now let's see it working. This is our public website, hosted on Vercel.",
        "We enter the address of a real website, Wikipedia, and click check.",
        "The model says likely legitimate, and lists the top reasons, such as the valid HTTPS certificate and normal links.",
        "We can also open the full list of all twenty one features that were computed for this page.",
    ]),
    ("phish", "demo", [
        "Next, a phishing example. For safety, we built a harmless fake phishing page that runs only on our own computer. Its form sends nothing.",
        "It asks you to verify your account, its links lead nowhere, and right click is disabled.",
        "We check this page in our local app.",
        "The verdict is likely phishing, with ninety nine point five percent probability. The reasons are the links that go nowhere, the missing HTTPS, and images loaded from other sites.",
    ]),
    ("modern", "demo", [
        "To be honest about limitations, here is a modern, cleanly built fake login page.",
        "The model calls it legitimate, so it is missed. Its HTML looks clean, and the training data is from twenty fifteen.",
        "Modern phishing kits look exactly like this, which is why fresher data is our main future work.",
    ]),
    ("extension", "demo", [
        "We also built a Chrome extension. With one click, it sends the current tab's address to our local service and shows the verdict and the reasons.",
        "It only asks for the active tab permission and access to our local service, and it collects no data in the background.",
    ]),
    ("deploy", "slide", [
        "The public version has safety features: it refuses internal network addresses, it does not host the fake test pages, and it limits how often each visitor can check.",
        "The project has one hundred and eighteen automated tests, and all the code is on GitHub.",
    ]),
    ("end", "slide", [
        "To conclude: our ensemble models reach about ninety seven point six percent accuracy, and our deployable model gives explained verdicts for real websites.",
        "In future, we want to train on fresh data and add modern reputation signals. Thank you for watching!",
    ]),
]


# --------------------------------------------------------------- audio

def synthesize(work):
    """One WAV per sentence with the offline Windows voice."""
    jobs = []
    for sid, _, sentences in SCENES:
        for i, text in enumerate(sentences):
            jobs.append({"file": str(work / f"{sid}_{i}.wav"), "text": text})
    (work / "tts.json").write_text(json.dumps(jobs), encoding="utf-8")
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{VOICE}')
$s.Rate = 0
foreach ($j in (Get-Content -Raw -Encoding UTF8 '{work / 'tts.json'}' | ConvertFrom-Json)) {{
  $s.SetOutputToWaveFile($j.file); $s.Speak($j.text)
}}
$s.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def scene_audio(work, sid, n):
    """Joins the sentence WAVs with short gaps; returns (wav, start times, total)."""
    parts, starts, t = [], [], 0.0
    for i in range(n):
        p = work / f"{sid}_{i}.wav"
        starts.append(t)
        d = wav_seconds(p)
        parts.append((p, d))
        t += d + GAP
    total = t - GAP + TAIL
    out = work / f"{sid}.wav"
    inputs, filters = [], []
    for k, (p, d) in enumerate(parts):
        inputs += ["-i", str(p)]
        filters.append(f"[{k}]apad=pad_dur={GAP}[a{k}]")
    chain = "".join(f"[a{k}]" for k in range(len(parts)))
    fc = ";".join(filters) + f";{chain}concat=n={len(parts)}:v=0:a=1,apad=pad_dur={TAIL}[out]"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", fc,
                    "-map", "[out]", "-ar", "48000", "-ac", "2", str(out)], check=True)
    return out, starts, total


# ------------------------------------------------------------ captions

CAPTION_JS = """
(() => {
  if (window.top !== window) return;  // captions only in the top-level page, not in iframes
  const T0 = %(t0)d, CAPS = %(caps)s;
  function tick() {
    if (!document.body) return;
    let el = document.getElementById('__cap');
    if (!el) {
      el = document.createElement('div'); el.id = '__cap';
      el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;min-height:82px;padding:14px 60px;' +
        'background:rgba(11,11,11,.82);color:#fff;font:500 22px/1.35 "Segoe UI",sans-serif;' +
        'text-align:center;z-index:2147483647;display:flex;align-items:center;justify-content:center';
      document.body.appendChild(el);
    }
    const t = (Date.now() - T0) / 1000;
    let cur = '';
    for (const [s, text] of CAPS) if (t >= s) cur = text;
    if (el.textContent !== cur) el.textContent = cur;
  }
  setInterval(tick, 100);
  document.addEventListener('DOMContentLoaded', tick);
})();
"""


# --------------------------------------------------------------- demos

def type_slowly(page, selector, text):
    page.click(selector)
    page.fill(selector, "")
    page.type(selector, text, delay=45)


def demo_live(page, at):
    page.goto(LIVE, wait_until="networkidle")
    at(1)
    type_slowly(page, "#url", "https://www.wikipedia.org/")
    page.wait_for_timeout(500)
    page.click("#go")
    page.wait_for_selector(".badge", timeout=60_000)
    at(3)
    page.click("details summary")
    page.wait_for_timeout(600)
    page.mouse.wheel(0, 380)


def demo_phish(page, at):
    page.goto(LOCAL + "demo/account-verify", wait_until="networkidle")
    at(2)
    page.goto(LOCAL, wait_until="networkidle")
    type_slowly(page, "#url", "http://127.0.0.1:8000/demo/account-verify")
    page.wait_for_timeout(400)
    page.click("#go")
    page.wait_for_selector(".badge", timeout=60_000)


def demo_modern(page, at):
    page.goto(LOCAL + "demo/modern-login", wait_until="networkidle")
    at(1)
    page.goto(LOCAL, wait_until="networkidle")
    type_slowly(page, "#url", "http://127.0.0.1:8000/demo/modern-login")
    page.click("#go")
    page.wait_for_selector(".badge", timeout=60_000)


def demo_extension(page, at, html):
    # Serve the real extension popup (popup.html + popup.js) from a localhost origin
    # that the API's CORS rules accept; only chrome.tabs is stubbed with a fixed URL.
    ext = ROOT / "extension"
    stub = ("<script>window.chrome={tabs:{query:async()=>[{url:'https://www.wikipedia.org/'}]}};</script>")
    popup = (ext / "popup.html").read_text(encoding="utf-8").replace(
        '<script src="popup.js"></script>', stub + '<script src="popup.js"></script>')
    files = {"/slide": ("text/html", html), "/popup.html": ("text/html", popup),
             "/popup.js": ("application/javascript", (ext / "popup.js").read_text(encoding="utf-8"))}

    def handler(route):
        path = "/" + route.request.url.split("localhost:8001/", 1)[1].split("?")[0]
        ctype, body = files[path]
        route.fulfill(status=200, content_type=ctype, body=body)

    page.route("http://localhost:8001/**", handler)
    page.goto("http://localhost:8001/slide", wait_until="networkidle")
    page.wait_for_timeout(1500)
    page.evaluate("document.getElementById('popup').src = '/popup.html'")


DEMOS = {"live": demo_live, "phish": demo_phish, "modern": demo_modern}


# ---------------------------------------------------------------- main

def record_scene(browser, work, sid, kind, starts, sentences, total, html):
    vdir = work / f"rec_{sid}"
    ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(vdir),
                              record_video_size={"width": W, "height": H}, color_scheme="light")
    page = ctx.new_page()
    t_page = time.monotonic()
    t0 = int(time.time() * 1000) + 300  # scene clock starts shortly after the page exists
    caps = json.dumps([[s, text] for s, text in zip(starts, sentences)])
    page.add_init_script(CAPTION_JS % {"t0": t0, "caps": caps})
    scene_start = t_page + 0.3

    def at(i):
        """Wait until sentence i starts."""
        delay = scene_start + starts[i] - time.monotonic()
        if delay > 0:
            page.wait_for_timeout(delay * 1000)

    if kind == "slide":
        # Served as a real page (not set_content) so the caption init script runs.
        page.route("http://localhost:8001/**",
                   lambda route: route.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://localhost:8001/slide", wait_until="load")
    elif sid == "extension":
        demo_extension(page, at, html)
    else:
        DEMOS[sid](page, at)
    remaining = scene_start + total - time.monotonic()
    if remaining > 0:
        page.wait_for_timeout(remaining * 1000)
    ctx.close()
    video = next(vdir.glob("*.webm"))
    overran = -remaining if remaining < 0 else 0.0
    return video, 0.3, overran


def main(work, output):
    work = Path(work)
    sys.stdout.reconfigure(encoding="utf-8")
    (work / "narration.txt").write_text(
        "\n\n".join(f"[{sid}]\n" + "\n".join(s) for sid, _, s in SCENES), encoding="utf-8")
    if not ONLY or not all((work / f"{sid}_0.wav").exists() for sid, _, _ in SCENES):
        print("Synthesizing narration...")
        synthesize(work)
    S = slides()
    with sync_playwright() as p:
        # The extension scene shows the real popup from a stand-in page on localhost.
        # A real extension may call the local API (host_permissions); a plain page
        # is blocked by Edge's local-network rules, so they are relaxed for this
        # recording browser only.
        browser = p.chromium.launch(channel="msedge", headless=True, args=[
            "--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessRespectPreflightResults,"
            "BlockInsecurePrivateNetworkRequests"])
        # Warm-up: the first page in a fresh browser loads slowly, which left the
        # first scene blank and without captions.
        warm = browser.new_page()
        warm.route("http://localhost:8001/**",
                   lambda route: route.fulfill(status=200, content_type="text/html", body=S["title"]))
        warm.goto("http://localhost:8001/slide", wait_until="load")
        warm.wait_for_timeout(1500)
        warm.close()
        clips = []
        for sid, kind, sentences in SCENES:
            clip = work / f"clip_{sid}.mp4"
            if ONLY and sid not in ONLY and clip.exists():
                clips.append(clip)  # re-use the earlier recording of this scene
                continue
            wav, starts, total = scene_audio(work, sid, len(sentences))
            video, offset, overran = record_scene(browser, work, sid, kind, starts, sentences, total, S.get(sid))
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{offset:.2f}", "-i", str(video),
                            "-i", str(wav), "-t", f"{total:.2f}", "-map", "0:v", "-map", "1:a",
                            "-vf", f"fps=30,scale={W}:{H},format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "21", "-c:a", "aac", "-b:a", "160k", "-shortest", str(clip)], check=True)
            clips.append(clip)
            print(f"  {sid:10s} {total:5.1f}s" + (f"  (actions overran by {overran:.1f}s)" if overran else ""))
        browser.close()
    lst = work / "clips.txt"
    lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(output)], check=True)
    print("Saved", output)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
