"""Course video that follows the guide's checklist, point by point.

Order: title slide -> overview -> problem statement -> objectives -> dataset
(source, size, inputs/outputs, sample rows, visualisation) -> preprocessing
(+ real code) -> workflow flowchart -> model choice and alternatives ->
training -> testing and metrics -> confusion matrix -> live demonstration ->
conclusion. Reuses the recording engine of make_video.py.

The guide's own title slide is used if <work_dir>/title_slide.png (or .jpg)
exists; otherwise a stand-in title slide is shown.

Needs: the local API running (start_app.bat) and internet access.
Run:   python -m report_build.make_course_video <work_dir> <output.mp4>
Re-record one scene only: set VIDEO_ONLY=title (other scenes are re-used).
"""
import base64
import html
import inspect
import json
import sys
from pathlib import Path

import pandas as pd

from report_build import make_app_video as app
from report_build import make_video as mv
from src import data, train

RES, ROOT = mv.RES, mv.ROOT
LABELS = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "svm": "SVM (RBF)", "xgboost": "XGBoost"}

EXTRA_CSS = """
pre { background:#16161a; color:#e8e6df; font:15px/1.45 Consolas,'Courier New',monospace; padding:16px 20px;
      border-radius:10px; overflow:hidden; white-space:pre; }
.flow { display:flex; align-items:stretch; gap:0; margin-top:34px; }
.step { flex:1; background:#fff; border:2px solid #2a78d6; border-radius:14px; padding:16px 14px; text-align:center; }
.step h3 { font-size:22px; margin-bottom:8px; color:#0b0b0b; }
.step p { font-size:15.5px; color:#52514e; line-height:1.4; }
.arrow { width:44px; display:flex; align-items:center; justify-content:center; font-size:34px; color:#2a78d6; }
table.data { font-size:15px; }
table.data td, table.data th { text-align:center; padding:6px 6px; }
"""


def page(body, n=None):
    return mv.page_html(body, n).replace("</style>", EXTRA_CSS + "</style>", 1)


def code_of(*objs):
    return html.escape("\n".join(inspect.getsource(o) for o in objs).rstrip())


def sample_table():
    df = data.load_raw()
    cols = ["having_ip_address", "url_length", "prefix_suffix", "having_sub_domain", "sslfinal_state",
            "request_url", "url_of_anchor", "links_in_tags", "sfh", "web_traffic", "result"]
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df[cols].head(8).values)
    return f"<table class='data'><tr>{head}</tr>{rows}</table>"


def training_log():
    """The training summary saved by src/train.py, in the format it prints."""
    cv = json.loads((RES / "cv_results.json").read_text())
    lines = []
    for fs in ("full", "lite"):
        lines.append(f"Tuning on the {fs} feature set ({30 if fs == 'full' else 21} features)")
        for r in (r for r in cv if r["feature_set"] == fs):
            lines.append(f"  [{fs}] {r['model']:14s} configs {r['n_configs']:2d}  CV F1 {r['cv_f1_mean']:.4f} "
                         f"+/- {r['cv_f1_std']:.4f}  fit {r['train_time_s']:.2f}s")
    serving = json.loads((ROOT / "models" / "serving.json").read_text())
    lines.append(f"Serving model: {serving['model']}")
    return html.escape("\n".join(lines))


def slides(work):
    S = {}
    title_img = next((work / n for n in ("title_slide.png", "title_slide.jpg") if (work / n).exists()), None)
    if title_img:
        b64 = base64.b64encode(title_img.read_bytes()).decode()
        mime = "png" if title_img.suffix == ".png" else "jpeg"
        S["title"] = ("<!doctype html><html><body style='margin:0;background:#fff;width:1280px;height:720px;"
                      "display:flex;align-items:center;justify-content:center'>"
                      f"<img src='data:image/{mime};base64,{b64}' style='max-width:1280px;max-height:720px'>"
                      "</body></html>")
    else:
        members = "<br>".join(f"{n} &nbsp;({r})" for n, r in mv.MEMBERS)
        S["title"] = page(f"""<div class="title"><div class="kicker">TY B.Tech INFT Sem V &middot; Machine Learning Project</div>
          <h1 style="margin:10px 0 22px">Phishing Website Detection using Machine Learning</h1>
          <div class="sub" style="line-height:1.6">{members}</div><div class="sub">Project guide: <b>{mv.GUIDE}</b></div></div>""")
    S["overview"] = page("""<div class="kicker">1 &middot; Brief overview</div><h1>Project overview</h1><ul style="margin-top:20px">
        <li>Detects whether a website is <b>phishing</b> or <b>legitimate</b> using machine learning</li>
        <li>Trained and compared <b>4 models</b> on <b>11,055</b> labelled websites</li>
        <li>Best model deployed as a <b>web app</b>, a <b>public website</b> and a <b>Chrome extension</b></li>
        <li>Every verdict is <b>explained</b> with the features that caused it</li></ul>""", 1)
    S["problem"] = page("""<div class="kicker">2 &middot; Problem statement</div><h1>The problem</h1>
      <div class="grid2" style="margin-top:18px">
        <div class="card"><h3>What problem?</h3><p>Phishing websites imitate banks, e-mail and shopping sites to steal
          passwords and card details.</p></div>
        <div class="card"><h3>Why is it important?</h3><p>Phishing is one of the most common ways accounts are hacked, and new
          sites appear faster than blocklists can track them.</p></div>
        <div class="card"><h3>Target users</h3><p>Everyday internet users, students and employees who receive suspicious
          links; IT and security teams who need a quick, explained check.</p></div>
        <div class="card"><h3>Our approach</h3><p>Learn the warning signs of phishing pages from labelled data, so that
          sites never seen before can be judged.</p></div></div>""", 2)
    S["objectives"] = page("""<div class="kicker">3 &middot; Objectives</div><h1>Objectives and expected outcomes</h1>
      <div class="cols" style="margin-top:16px"><div><h3 style="font-size:22px;margin-bottom:8px">Objectives</h3><ul style="font-size:20px">
        <li>Study the dataset and its quality</li><li>Build and compare Decision Tree, Random Forest, SVM, XGBoost</li>
        <li>Find which warning signs matter most</li><li>Deploy the best model in an app that explains its answer</li></ul></div>
      <div><h3 style="font-size:22px;margin-bottom:8px">Expected outcomes</h3><ul style="font-size:20px">
        <li>High accuracy and recall on unseen websites</li><li>An explained verdict for any URL in 1&ndash;2 seconds</li>
        <li>Web page, public website and Chrome extension</li></ul></div></div>""", 3)
    S["dataset"] = page("""<div class="kicker">4 &middot; Dataset description</div><h1>UCI Phishing Websites dataset</h1>
      <table style="margin-top:16px;font-size:20px">
        <tr><td>Source</td><td>UCI Machine Learning Repository (id 327), Mohammad, Thabtah &amp; McCluskey</td></tr>
        <tr><td>Records</td><td>11,055 websites</td></tr>
        <tr><td>Input attributes</td><td>30 features about the URL, page content, domain and reputation, each -1 / 0 / 1</td></tr>
        <tr><td>Output attribute</td><td><b>result</b>: phishing or legitimate (4,898 phishing = 44.3%)</td></tr>
        <tr><td>Examples of inputs</td><td>sslfinal_state (HTTPS), url_of_anchor (links), having_ip_address, prefix_suffix, web_traffic</td></tr>
      </table>""", 4)
    S["sample"] = page(f"""<div class="kicker">4 &middot; Dataset description</div><h1>Sample records (first 8 rows, 11 of 31 columns)</h1>
      <div style="margin-top:14px">{sample_table()}</div>
      <div class="sub" style="margin-top:14px;font-size:17px">Original coding: result = -1 phishing, 1 legitimate.
        We recode it to phishing = 1 (positive class), legitimate = 0.</div>""", 5)
    S["viz"] = page(f"""<div class="kicker">4 &middot; Dataset visualisation</div><h1>Phishing rate for each feature value</h1>
      {mv.img('eda_phishing_rate.png', 450)}""", 6)
    S["preprocess"] = page("""<div class="kicker">5 &middot; Data preprocessing</div><h1>Steps before training</h1>
      <table style="margin-top:14px;font-size:19px">
        <tr><td><b>Data cleaning</b></td><td>Validated every column and value; found 47% exact duplicate rows (used for an extra test)</td></tr>
        <tr><td><b>Missing values</b></td><td>None &mdash; checked automatically by the code</td></tr>
        <tr><td><b>Encoding</b></td><td>Features already numeric (-1/0/1); only the label recoded to phishing = 1</td></tr>
        <tr><td><b>Feature scaling</b></td><td>StandardScaler for the SVM (inside a pipeline); not needed for tree models</td></tr>
        <tr><td><b>Train-test split</b></td><td>Stratified 80 / 20: 8,844 training and 2,211 test websites, seed 42</td></tr>
      </table>""", 7)
    S["code"] = page(f"""<div class="kicker">5 &middot; Data preprocessing &middot; code (src/data.py)</div><h1>Validation and train-test split</h1>
      <pre style="margin-top:12px">{code_of(data.validate, data.split)}</pre>""", 8)
    S["workflow"] = page("""<div class="kicker">6 &middot; System architecture / workflow</div><h1>Complete workflow</h1>
      <div class="flow">
        <div class="step"><h3>Input</h3><p>Training: 11,055 labelled websites<br><br>App: a website URL typed by the user</p></div>
        <div class="arrow">&#10140;</div>
        <div class="step"><h3>Preprocessing</h3><p>Validate, recode label, split 80/20<br><br>App: compute 21 features from the URL, DNS, certificate and HTML</p></div>
        <div class="arrow">&#10140;</div>
        <div class="step"><h3>Model Training</h3><p>4 models &times; grid search with 5-fold cross-validation; best model saved</p></div>
        <div class="arrow">&#10140;</div>
        <div class="step"><h3>Prediction</h3><p>XGBoost outputs a phishing probability; &ge; 50% means phishing</p></div>
        <div class="arrow">&#10140;</div>
        <div class="step"><h3>Output</h3><p>Verdict + probability + top 5 reasons (SHAP), shown in the web page or extension</p></div>
      </div>""", 9)
    S["models"] = page("""<div class="kicker">7 &middot; Machine learning model</div><h1>Why XGBoost? Alternatives compared</h1>
      <div class="cols" style="margin-top:14px"><div><h3 style="font-size:21px;margin-bottom:8px">Alternatives compared</h3><ul style="font-size:19px">
        <li><b>Decision Tree</b>: simple and readable, but overfits</li><li><b>Random Forest</b>: bagging of many trees</li>
        <li><b>SVM (RBF kernel)</b>: maximum-margin kernel method</li><li><b>XGBoost</b>: gradient boosting</li></ul></div>
      <div><h3 style="font-size:21px;margin-bottom:8px">Why XGBoost was selected</h3><ul style="font-size:19px">
        <li>Ties for best CV F1 on the deployable features</li><li>About 8&times; faster prediction than Random Forest</li>
        <li>Best phishing recall and ROC-AUC (full features)</li><li>Most robust without duplicate rows</li>
        <li>Explains every prediction (SHAP)</li></ul></div></div>""", 10)
    S["training"] = page(f"""<div class="kicker">8 &middot; Model training</div><h1>Training process</h1>
      <div class="cols" style="margin-top:10px"><div style="flex:.62"><ul style="font-size:19px">
        <li>Grid search over hyper-parameters</li><li>5-fold stratified cross-validation on the training set</li>
        <li>Scored by F1 of the phishing class</li><li>Best configuration refit on all 8,844 rows and saved</li></ul></div>
      <div><pre style="font-size:13.5px">{training_log()}</pre>
        <div class="sub" style="font-size:13px;margin-top:6px">Summary saved by our training run (results/cv_results.json)</div></div></div>""", 11)
    S["testing"] = page(f"""<div class="kicker">8 &middot; Testing and evaluation</div><h1>Test-set metrics (2,211 unseen websites)</h1>
      <div class="sub" style="margin:10px 0 6px;font-size:18px">All 30 features:</div>{mv.results_table('full')}
      <div class="sub" style="margin:14px 0 6px;font-size:18px">Deployable model, 21 features:</div>{mv.results_table('lite')}""", 12)
    S["confusion"] = page(f"""<div class="kicker">8 &middot; Testing and evaluation</div><h1>Confusion matrices (all 30 features)</h1>
      {mv.img('confusion_full.png', 330)}
      <div class="sub" style="margin-top:12px;text-align:center">XGBoost: 1,208 + 950 correct &middot; 23 false alarms &middot; 30 missed phishing sites</div>""", 13)
    S["conclusion"] = page("""<div class="kicker">Conclusion</div><h1>Summary</h1><ul style="margin-top:20px">
        <li>About <b>97.6%</b> accuracy on the benchmark; XGBoost has the best recall and ROC-AUC</li>
        <li>Working, explained app: web page, public website and Chrome extension</li>
        <li>Limitation: training data from about 2015; modern, cleanly built phishing pages can be missed</li>
        <li>Future work: fresh data and modern reputation signals</li></ul>
      <div class="sub" style="text-align:center;margin-top:40px;font-size:30px;color:#0b0b0b"><b>Thank you!</b></div>""", 15)
    return S


SCENES = [
    ("title", "slide", [
        "This is our machine learning project, Phishing Website Detection using Machine Learning, by Parth Kinage, "
        "Ronak Boddu and Vedangi Pokharkar, under the guidance of Doctor Kanchan Dhuri.",
    ]),
    ("overview", "slide", [
        "In brief, our project detects phishing websites using machine learning.",
        "We trained and compared four models on eleven thousand labelled websites, chose the best one, and built a "
        "working app that checks any website address and explains its answer.",
    ]),
    ("problem", "slide", [
        "The problem: phishing websites imitate trusted sites, like banks and email providers, to steal passwords and "
        "card details.",
        "This matters because phishing is one of the most common ways that accounts are hacked, and new phishing sites "
        "appear faster than blocklists can track them.",
        "Our target users are everyday internet users, students and employees who receive suspicious links, and IT and "
        "security teams who need a quick, explained check.",
    ]),
    ("objectives", "slide", [
        "Our objectives were to study the dataset, to build and compare four machine learning models, to find which "
        "warning signs matter most, and to deploy the best model in an app that explains its answer.",
        "The expected outcome is high accuracy and recall on unseen websites, and an explained verdict for any web "
        "address in one to two seconds.",
    ]),
    ("dataset", "slide", [
        "Our dataset is the Phishing Websites dataset from the U C I Machine Learning Repository.",
        "It has eleven thousand and fifty five records and thirty input features, which describe the web address, the "
        "page content, the domain, and the site's reputation. Each feature is coded as minus one, zero, or one.",
        "The output attribute is the result: phishing or legitimate. Forty four percent of the websites are phishing.",
    ]),
    ("sample", "slide", [
        "Here are the first records of the dataset. Each row is one website, each column is a feature, and the last "
        "column is the output.",
        "In the original file, minus one in the result column means phishing. We recoded it so that phishing is one, "
        "the positive class.",
    ]),
    ("viz", "slide", [
        "This visualisation shows the share of phishing websites for each value of eight key features.",
        "For example, when most of a page's links point nowhere, ninety nine percent of those websites are phishing, "
        "while a valid HTTPS certificate usually means a legitimate site.",
    ]),
    ("preprocess", "slide", [
        "Before training, we prepared the data in five steps.",
        "Data cleaning: we validated every column and value, and found that forty seven percent of the rows are exact "
        "duplicates, which we used for an extra, stricter test.",
        "Missing values: there are none, and our code checks this automatically.",
        "Encoding: all features are already numeric, so no categorical encoding was needed. Only the label was recoded.",
        "Feature scaling: tree models do not need it, but for the S V M we standardise the features inside a pipeline.",
        "And the train test split is stratified: eighty percent for training, and twenty percent for testing.",
    ]),
    ("code", "slide", [
        "This is the actual code from our project. The validate function checks the columns, missing values, and the "
        "allowed values, and the split function makes the stratified eighty twenty split with a fixed random seed.",
    ]),
    ("workflow", "slide", [
        "This flowchart shows our complete workflow.",
        "The input is the labelled dataset for training, and in the app, a website address typed by the user.",
        "Preprocessing prepares the data, and in the app, computes twenty one features from the address, its DNS "
        "record, its certificate and its HTML.",
        "Model training tunes four models with grid search and five-fold cross-validation, and saves the best one.",
        "For prediction, XGBoost outputs a phishing probability, and the output is the verdict with its top reasons.",
    ]),
    ("models", "slide", [
        "We compared four algorithms: a decision tree, a random forest, a support vector machine, and XGBoost. "
        "Together they cover a single tree, bagging, a kernel method and boosting.",
        "We selected XGBoost for the app. It tied for the best cross-validation F1-score on the deployable features, "
        "and it predicts about eight times faster than random forest.",
        "It also has the best phishing recall and ROC A U C, it is the most robust when duplicate rows are removed, "
        "and it explains every prediction.",
    ]),
    ("training", "slide", [
        "Training process: for each model, grid search tries every combination of settings, and five-fold "
        "cross-validation scores each one on the training data.",
        "The best configuration is then retrained on all eight thousand eight hundred and forty four training records "
        "and saved. On the right is the training summary from our run.",
    ]),
    ("testing", "slide", [
        "Testing process: every saved model is evaluated once on the test set: two thousand two hundred and eleven "
        "websites it has never seen.",
        "With all thirty features, XGBoost reaches ninety seven point six percent accuracy, ninety seven point six "
        "precision, ninety six point nine recall, and an F1-score of ninety seven point three. Random forest is the "
        "most accurate at ninety seven point seven.",
        "The deployable model, which uses twenty one features, reaches about ninety four point six percent accuracy.",
    ]),
    ("confusion", "slide", [
        "The confusion matrices show the exact errors on the test set.",
        "For XGBoost, one thousand two hundred and eight legitimate and nine hundred and fifty phishing websites are "
        "correct. Only twenty three legitimate sites were wrongly flagged, and thirty phishing sites were missed.",
    ]),
    ("demo_legit", "demo", [
        "Now the live demonstration. This is our application, running on the internet.",
        "For input, we type the address of a real website, Wikipedia, and click Check.",
        "The app generates the prediction in about a second: likely legitimate, with a phishing probability of almost "
        "zero.",
        "To interpret the output, the top reasons show which features pushed the decision. Green arrows mean towards "
        "legitimate, and red arrows mean towards phishing.",
    ]),
    ("fake", "demo", [
        "Next, a phishing example. We built a harmless fake phishing page that runs only on our computer, and its "
        "form sends nothing.",
        "It asks you to verify your account, its password form submits nowhere, its links lead nowhere, and right "
        "click is disabled.",
    ]),
    ("fake_check", "demo", [
        "We enter its address in the app, and click Check.",
        "The prediction is likely phishing, with ninety nine point five percent probability.",
        "The output explains why: most links lead nowhere, there is no HTTPS, images come from other sites, and right "
        "click is disabled.",
    ]),
    ("popup", "demo", [
        "The same prediction is available in our Chrome extension. One click sends the current tab's address to our "
        "service.",
        "Here the current tab is the fake page, and the popup warns: likely phishing, with the top reasons.",
    ]),
    ("conclusion", "slide", [
        "To conclude, our models reach about ninety seven point six percent accuracy on the benchmark, and our app "
        "gives an explained verdict for real websites.",
        "Its main limitation is the older training data, so modern, cleanly built phishing pages can be missed. Our "
        "future work is to train on fresh data. Thank you.",
    ]),
]


def d_demo_legit(page, at):
    page.goto(mv.LIVE, wait_until="networkidle")
    at(1)
    mv.type_slowly(page, "#url", "https://www.wikipedia.org/")
    page.wait_for_timeout(400)
    page.click("#go")
    page.wait_for_selector(".badge", timeout=60_000)
    at(3)
    page.hover("ul.reasons")


def d_fake(page, at):
    app.show_page(page, "demo/account-verify")
    at(1)
    page.hover("button:has-text('Verify')")
    page.wait_for_timeout(1500)
    page.hover("a:has-text('Help')")


DEMOS = {"demo_legit": d_demo_legit, "fake": d_fake, "fake_check": app.d_fake_check, "popup": app.d_popup}


def main(work, output):
    work = Path(work)
    S = slides(work)
    mv.SCENES = SCENES
    mv.DEMOS = DEMOS
    mv.slides = lambda: S
    mv.main(work, output)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
