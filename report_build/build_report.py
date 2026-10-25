"""Fills the college report template (Sem V ML project) with this project.

Opens the template .docx, fills the cover, certificate and abstract in place,
and rebuilds the chapters by cloning the template's own blocks (headings,
body text, lists, tables, figures, captions, equations, references), so the
formatting matches the template. Result numbers are read from results/.

Run: python -m report_build.build_report <template.docx> <output.docx>
Then open the output in Word and update fields (the script
report_build/update_fields.ps1 does this and exports a PDF).
"""
import copy
import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TITLE = "Phishing Website Detection using Machine Learning"
MEMBERS = ["Kinage Parth (24101B0036)", "Boddu Ronak (24101B0044)", "Pokharkar Vedangi (24101B0051)"]
GUIDE = "Dr. Kanchan Dhuri"
GITHUB = "https://github.com/ParthKinage/Phishing-website-"
LIVE = "https://phishing-website-detector-wheat.vercel.app"
MODEL_LABELS = {"decision_tree": "Decision Tree", "random_forest": "Random Forest",
                "svm": "SVM (RBF)", "xgboost": "XGBoost"}

W = qn


# ----------------------------------------------------------------- helpers

def text_of(el):
    return "".join(t.text or "" for t in el.iter(W("w:t")))


def style_of(p):
    s = p.find(f"{W('w:pPr')}/{W('w:pStyle')}")
    return s.get(W("w:val")) if s is not None else None


def make_run(text, bold=False, italic=False, rpr=None):
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    if bold or italic:
        pr = r.find(W("w:rPr"))
        if pr is None:
            pr = OxmlElement("w:rPr")
            r.insert(0, pr)
        if bold:
            pr.append(OxmlElement("w:b"))
        if italic:
            pr.append(OxmlElement("w:i"))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def add_marked_runs(p, text, rpr=None):
    """Append runs to paragraph `p`; **bold** and *italic* markup supported."""
    for part in re.split(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            p.append(make_run(part[2:-2], bold=True, rpr=rpr))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            p.append(make_run(part[1:-1], italic=True, rpr=rpr))
        else:
            p.append(make_run(part, rpr=rpr))


def strip_runs(p, keep_breaks=False):
    for el in list(p):
        if el.tag in (W("w:bookmarkStart"), W("w:bookmarkEnd"), W("w:hyperlink"), W("w:proofErr")):
            p.remove(el)
        elif el.tag == W("w:r"):
            if keep_breaks and el.find(W("w:br")) is not None and el.find(W("w:t")) is None:
                continue
            p.remove(el)
    for math in p.findall(qn("m:oMath")):
        p.remove(math)


def set_text(p, text, rpr_from_first=True):
    """Replace a paragraph's text, keeping the first text run's formatting."""
    first = next((r for r in p.iter(W("w:r")) if r.find(W("w:t")) is not None), None)
    rpr = first.find(W("w:rPr")) if (first is not None and rpr_from_first) else None
    rpr = copy.deepcopy(rpr) if rpr is not None else None
    strip_runs(p)
    add_marked_runs(p, text, rpr)


# ------------------------------------------------------------------ builder

class Builder:
    """Clones the template's prototype blocks and inserts them in order."""

    def __init__(self, doc, protos, anchor):
        self.doc, self.p, self.anchor = doc, protos, anchor
        self.fig = {}
        self.tab = {}

    def _put(self, el):
        self.anchor.addnext(el)
        self.anchor = el
        return el

    def _clone(self, name):
        return copy.deepcopy(self.p[name])

    def h1(self, text):
        p = self._clone("h1")
        strip_runs(p, keep_breaks=True)
        p.append(make_run(text))
        return self._put(p)

    def h2(self, text):
        p = self._clone("h2")
        strip_runs(p)
        p.append(make_run(text))
        return self._put(p)

    def h3(self, text):
        p = self._clone("h3")
        strip_runs(p)
        p.append(make_run(text))
        return self._put(p)

    def para(self, text):
        p = self._clone("body")
        strip_runs(p)
        add_marked_runs(p, text)
        return self._put(p)

    def bullets(self, items, kind="bullet"):
        for item in items:
            p = self._clone(kind)
            strip_runs(p)
            add_marked_runs(p, item)
            self._put(p)

    def caption(self, kind, number, text):
        p = self._clone("fcap" if kind == "Figure" else "tcap")
        strip_runs(p)
        p.append(make_run(f"{kind} {number}: {text}"))
        return self._put(p)

    def figure(self, number, image, caption, width=6.0):
        p = self._clone("img")
        strip_runs(p)
        self._put(p)
        Paragraph(p, self.doc._body).add_run().add_picture(str(image), width=Inches(width))
        self.caption("Figure", number, caption)

    def table(self, number, caption, headers, rows, widths):
        self.caption("Table", number, caption)
        tbl = self._clone("tbl")
        assert abs(sum(widths) - 8666) <= 2, sum(widths)
        tbl.find(f"{W('w:tblPr')}/{W('w:tblW')}").set(W("w:w"), str(sum(widths)))
        grid = tbl.find(W("w:tblGrid"))
        for g in list(grid):
            grid.remove(g)
        for wdt in widths:
            gc = OxmlElement("w:gridCol")
            gc.set(W("w:w"), str(wdt))
            grid.append(gc)
        for tr in tbl.findall(W("w:tr")):
            tbl.remove(tr)
        for cells, kind in [(headers, "hdr")] + [(r, "row") for r in rows]:
            tr = copy.deepcopy(self.p[f"{kind}_tr"])
            for tc in tr.findall(W("w:tc")):
                tr.remove(tc)
            for value, wdt in zip(cells, widths):
                tc = copy.deepcopy(self.p[f"{kind}_tc"])
                tc.find(f"{W('w:tcPr')}/{W('w:tcW')}").set(W("w:w"), str(wdt))
                cp = tc.find(W("w:p"))
                rpr = self.p[f"{kind}_rpr"]
                strip_runs(cp)
                add_marked_runs(cp, str(value), rpr)
                tr.append(tc)
            tbl.append(tr)
        self._put(tbl)
        # small gap after the table
        gap = self._clone("body")
        strip_runs(gap)
        self._put(gap)

    def equation(self, lhs, num, den, number):
        p = self._clone("eq")
        ts = p.findall(f".//{qn('m:t')}")
        ts[0].text, ts[1].text, ts[2].text = f"{lhs}=", num, den
        for t in p.iter(W("w:t")):
            if t.text and t.text.startswith("("):
                t.text = f"({number})"
        return self._put(p)

    def references(self, refs):
        for ref in refs:
            p = self._clone("ref")
            strip_runs(p)
            add_marked_runs(p, ref)
            self._put(p)

    def element(self, el):
        return self._put(el)


def field_paragraph(instr, placeholder):
    p = OxmlElement("w:p")
    for kind in ("begin",):
        r = OxmlElement("w:r")
        fc = OxmlElement("w:fldChar")
        fc.set(W("w:fldCharType"), kind)
        r.append(fc)
        p.append(r)
    r = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r.append(it)
    p.append(r)
    r = OxmlElement("w:r")
    fc = OxmlElement("w:fldChar")
    fc.set(W("w:fldCharType"), "separate")
    r.append(fc)
    p.append(r)
    p.append(make_run(placeholder))
    r = OxmlElement("w:r")
    fc = OxmlElement("w:fldChar")
    fc.set(W("w:fldCharType"), "end")
    r.append(fc)
    p.append(r)
    return p


# --------------------------------------------------------------- the data

def pct(x):
    return f"{100 * x:.2f}"


def load_results():
    full = pd.read_csv(RES / "metrics_full.csv")
    lite = pd.read_csv(RES / "metrics_lite.csv")
    dedup = pd.read_csv(RES / "dedup_sensitivity.csv")
    cv = json.loads((RES / "cv_results.json").read_text())
    summary = json.loads((RES / "data_summary.json").read_text())
    serving = json.loads((ROOT / "models" / "serving.json").read_text())
    ablation = pd.read_csv(RES / "ablation.csv")
    gap = pd.read_csv(RES / "lite_gap.csv")
    live = json.loads((RES / "live_check.json").read_text())
    return full, lite, dedup, cv, summary, serving, ablation, gap, live


def metric_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([MODEL_LABELS[r.model], pct(r.accuracy), pct(r.precision), pct(r.recall), pct(r.f1),
                     f"{r.roc_auc:.4f}", f"{r.cv_f1_mean:.4f} ± {r.cv_f1_std:.4f}",
                     f"{r.train_time_s:.2f}", f"{r.inference_ms:.2f}"])
    return rows


def params_text(p):
    return ", ".join(f"{k.replace('svc__', '')}={v}" for k, v in p.items())


# ----------------------------------------------------------- the content

def write_body(b, R):
    full, lite, dedup, cv, summary, serving, ablation, gap, live = R
    fbest = full.loc[full.accuracy.idxmax()]
    fx = full[full.model == "xgboost"].iloc[0]
    lx = lite[lite.model == "xgboost"].iloc[0]
    dfx = dedup[(dedup.feature_set == "full") & (dedup.model == "xgboost")].iloc[0]
    drops = dedup.f1_change * 100
    k15 = ablation[ablation.k == 15].iloc[0]

    # ------------------------------------------------------- Chapter 1
    b.h1("Introduction")
    b.para("Phishing is a form of online fraud in which an attacker builds a website that imitates a "
           "trusted organisation, such as a bank, an e-mail provider or an online shop, in order to steal "
           "passwords, card numbers and other personal data. Phishing pages are cheap to create and are usually "
           "online for only a short time, so lists of known bad sites are always one step behind the attackers.")
    b.para("This project builds a machine learning (ML) system that decides whether a website is **phishing** or "
           "**legitimate** from measurable properties of its address and page. Four supervised classifiers are "
           "trained and compared on a public benchmark, the most important warning signs are identified with "
           "several feature-importance methods, and the best deployable model is turned into a working "
           "application: a web service, a web page and a Chrome extension that check any URL and explain the verdict.")

    b.h2("Background")
    b.para("Mohammad, Thabtah and McCluskey [1] proposed a set of 30 phishing indicators and published the "
           "UCI Phishing Websites dataset [2]. The indicators fall into four groups: address-bar features "
           "(for example an IP address used as the host, a very long URL, an '@' symbol or a '-' in the domain), "
           "HTML and JavaScript features (where links, forms, scripts and images point to), domain features "
           "(domain age, DNS record, registration length) and reputation features (web traffic rank, PageRank, "
           "Google index, number of backlinks). Each indicator is coded as -1 (phishing-like), 0 (suspicious) or "
           "1 (legitimate-like).")
    b.para("Detecting phishing from these indicators is a supervised binary classification problem. Tree-based "
           "ensembles such as Random Forest [6] and gradient-boosted trees (XGBoost) [7] usually perform well on "
           "such tabular data, while kernel methods such as the Support Vector Machine (SVM) [9] are a strong "
           "classical baseline. Explanation methods such as SHAP [8] and permutation importance [11] show which "
           "features drive the predictions.")

    b.h2("Motivation")
    b.para("Many published models report very high accuracy on this dataset, but they are rarely turned into a "
           "tool that can check a real website. Several of the original features can no longer be computed: Alexa "
           "traffic rank and Google PageRank have been discontinued, and WHOIS and search-engine lookups are slow "
           "or rate-limited. We wanted to find out how much accuracy remains when a model uses only features that "
           "can be computed live, which features matter most, and whether the high reported accuracy survives a "
           "stricter evaluation. A working, explainable demo also makes the work useful beyond a single table "
           "of numbers.")

    b.h2("Problem Statement")
    b.para("Given a website URL, classify the website as phishing or legitimate. The model is trained on the 30 "
           "pre-computed features of the UCI Phishing Websites dataset; for deployment, it must work from features "
           "that a program can compute from the URL and its static HTML within a few seconds, and it must explain "
           "each decision. Missing a phishing site is costlier than a false alarm, so recall and F1-score on the "
           "phishing class are the key metrics.")

    b.h2("Objectives")
    b.para("The objectives of this project are:")
    b.bullets([
        "To study the UCI Phishing Websites dataset, its features and its quality issues (exploratory data analysis).",
        "To design and implement a reproducible training pipeline for Decision Tree, Random Forest, SVM and XGBoost "
        "classifiers on two feature sets: all 30 features (full) and the 21 features computable live (lite).",
        "To evaluate the performance of these models with cross-validation and a held-out test set, and compare it "
        "with existing methods.",
        "To identify the most important features with built-in, permutation and SHAP importance, and to explain the "
        "accuracy gap between the full and lite feature sets.",
        "To deploy the best lite model as a web service with a web page and a Chrome extension that give an "
        "explained verdict for any URL.",
    ])

    b.h2("Scope of the Work")
    b.para("The work covers the 11,055-website UCI dataset, four classical ML models, CPU-only training on a "
           "laptop, and a live demo that analyses the static HTML of a page. It does not cover deep learning on raw "
           "page content, screenshots or e-mails, JavaScript-rendered pages, or a large-scale evaluation on today's "
           "phishing websites. The online version is a course demonstration, not a hardened security product.")

    b.h2("Organization of the Report")
    b.para("Chapter 2 reviews existing work and identifies research gaps. Chapter 3 describes the proposed system: "
           "its architecture, the dataset with exploratory data analysis, the models, the training procedure and the "
           "evaluation metrics. Chapter 4 presents the experimental setup, the results, the feature-importance "
           "analysis, a comparison with existing methods and the deployment. Chapter 5 concludes and suggests future "
           "work, and Chapter 6 lists publications.")

    # ------------------------------------------------------- Chapter 2
    b.h1("Literature Survey")
    b.para("This chapter reviews work on ML-based phishing website detection that uses the same benchmark data, so "
           "that results can be compared.")
    b.h2("Review of Existing Work")
    b.para("Mohammad, Thabtah and McCluskey [1] introduced the 30 phishing features used in this project and "
           "classified websites with a self-structuring neural network. Their dataset, published in the UCI "
           "repository [2], has since become a standard benchmark.")
    b.para("Awasthi and Goel [3] (2022) compared six base classifiers (logistic regression, naive Bayes, decision "
           "tree, SVM, k-nearest neighbours, linear discriminant analysis) with six ensembles (bagging, AdaBoost, "
           "gradient boosting, voting, Extra Trees and XGBoost) on the same 11,055-instance dataset, which they "
           "obtained from Kaggle. Extra Trees performed best with 98.73% accuracy and XGBoost reached 98.37%. "
           "These best results were obtained without cross-validation, using all 30 features.")
    b.para("Almujahid, Haq and Alshehri [4] (2024) compared logistic regression, k-nearest neighbours, decision "
           "tree, random forest, SVM, XGBoost, a convolutional neural network (CNN) and a dense neural network on two "
           "datasets, with feature selection (SelectKBest) and GridSearchCV tuning. On the UCI dataset, the tuned CNN "
           "achieved 0.99 accuracy, 0.99 precision, 0.98 recall and 0.99 F1-score. The authors note that the CNN "
           "was not interpreted and that real-world deployment was not evaluated.")
    b.para("Newaz, Haq and Ahmed [5] (2024) built a stacking ensemble: recursive feature elimination, ten base "
           "classifiers combined by a greedy search, and a multi-layer perceptron as the meta-classifier, trained "
           "with 10-fold cross-validation on four datasets. On the UCI dataset they report 97.49% accuracy, 97.11% "
           "precision, 98.43% recall and 97.76% F1-score, with Random Forest (97.28%) and XGBoost (97.29%) as the "
           "strongest single classifiers. The base learners were not tuned.")

    b.h2("Summary of Literature")
    b.para("Table 2.1 summarizes the literature reviewed.")
    b.table("2.1", "Summary of literature survey",
            ["Sr. No.", "Author(s) & Year", "Method / Technique", "Dataset", "Result", "Limitation"],
            [["1", "Awasthi & Goel, 2022 [3]", "6 base + 6 ensemble classifiers; 10-fold CV for ensembles",
              "UCI/Kaggle, 11,055 sites, 30 features", "Extra Trees: 98.73% accuracy",
              "Best results without CV; all 30 features; no deployment"],
             ["2", "Almujahid et al., 2024 [4]", "LR, KNN, DT, RF, SVM, XGBoost, CNN, DNN; SelectKBest, GridSearchCV",
              "Mendeley (10,000) and UCI (11,055)", "CNN: 0.99 accuracy, 0.98 recall on UCI",
              "No interpretability; no real-world deployment"],
             ["3", "Newaz et al., 2024 [5]", "RFECV + 10 base learners, stacking with MLP meta-classifier",
              "UCI + 3 Mendeley datasets", "97.49% accuracy, 97.76% F1 on UCI",
              "Untuned base learners; preprint; no live use"]],
            [700, 1400, 1900, 1500, 1500, 1666])

    b.h2("Research Gaps")
    b.para("The reviewed work leaves four gaps that this project addresses:")
    b.bullets([
        "**Evaluation on duplicated data.** Almost half of the dataset's rows are exact duplicates (Section 3.2.1), "
        "yet the reviewed studies evaluate on random splits without checking how duplicates inflate the scores.",
        "**Deployability.** The models use features (traffic rank, PageRank, Google index, backlinks, WHOIS) that "
        "cannot be computed for a new URL today, and none of the studies deploys a model that checks live websites.",
        "**Interpretability.** Few studies explain which features drive the predictions or whether different "
        "importance methods agree.",
        "**Full-vs-deployable gap.** No study measures how much accuracy is lost when only live-computable features "
        "are used, or which missing features cause the loss.",
    ])

    # ------------------------------------------------------- Chapter 3
    b.h1("Proposed Methodology")
    b.para("The proposed system has two parts: an offline pipeline that prepares the data, trains, tunes, evaluates "
           "and interprets four classifiers on two feature sets, and an online service that computes the "
           "deployable features for a live URL and returns an explained prediction.")

    b.h2("System Architecture")
    b.para("The block diagram of the proposed system is shown in Figure 3.1. In the offline pipeline (A), the "
           "dataset is validated and split, the models are tuned with cross-validation, evaluated once on the test "
           "set and analysed with feature-importance methods. The best lite model is saved. In the online part (B), a "
           "client sends a URL to the FastAPI service, the feature extractor computes the 21 lite features, the "
           "XGBoost model returns a phishing probability, and SHAP values give the top reasons.")
    b.figure("3.1", RES / "architecture.png", "Block diagram of the proposed system", width=6.0)

    b.h2("Dataset Description")
    b.para("The details of the dataset used are given in Table 3.1.")
    b.table("3.1", "Dataset description", ["Parameter", "Details"], [
        ["Name of dataset", "Phishing Websites (Mohammad, Thabtah and McCluskey)"],
        ["Source", "UCI Machine Learning Repository, dataset id 327 [2]"],
        ["Number of samples", f"{summary['rows']:,} websites ({summary['unique_rows']:,} unique rows)"],
        ["Number of features", "30, each coded -1 / 0 / 1 (no missing values)"],
        ["Number of classes", f"2: phishing {summary['phishing']:,} ({summary['phishing_share']:.1%}), "
                              f"legitimate {summary['legitimate']:,}"],
        ["Train / Test split", "Stratified 80 / 20, random_state = 42: 8,844 training, 2,211 test websites"],
    ], [2600, 6066])
    b.para("The label is mapped to **phishing = 1** (the positive class) and legitimate = 0. The two feature sets "
           "are listed in Table 3.2. The lite set keeps the features that the extractor can compute from a URL and "
           "its static HTML; the nine excluded features need WHOIS, discontinued services, search-engine queries, "
           "blacklists or JavaScript execution.")
    from src.features import EXCLUDED_FROM_LITE, PAGE, URL_ONLY
    b.table("3.2", "Feature sets used", ["Feature set", "Features"], [
        ["URL features (9, lite)", ", ".join(URL_ONLY)],
        ["Page / network features (12, lite)", ", ".join(PAGE)],
        ["Excluded from lite (9)", ", ".join(EXCLUDED_FROM_LITE)],
    ], [2600, 6066])

    b.h3("Exploratory Data Analysis (EDA) Results")
    b.para(f"The classes are moderately balanced: {summary['phishing_share']:.1%} of the websites are phishing "
           f"(Figure 3.2). The most important quality issue is duplication: {summary['duplicate_rows']:,} of the "
           f"{summary['rows']:,} rows ({summary['duplicate_rows'] / summary['rows']:.0%}) are exact copies of "
           f"another row, and {summary['conflicting_feature_vectors']} feature vectors occur with both labels. "
           "Because a random split can put a row in the training set and its copy in the test set, the test scores "
           "can be optimistic; Section 4.2 measures this effect.")
    b.figure("3.2", RES / "eda_class_duplicates.png", "Class distribution and duplicate rows", width=6.0)
    b.para("Figure 3.3 shows the share of phishing websites for each value of eight key features. Websites with "
           "*url_of_anchor* = -1 (most links point nowhere or to other domains) are 99% phishing, and websites "
           "without valid HTTPS (*sslfinal_state* = -1 or 0) are mostly phishing, while a valid certificate "
           "(+1) means only 11% phishing. Figure 3.4 ranks all features by their correlation with the label: "
           "*sslfinal_state* (r = -0.72) and *url_of_anchor* (r = -0.69) are by far the strongest.")
    b.figure("3.3", RES / "eda_phishing_rate.png", "Phishing rate for each value of eight key features", width=6.0)
    b.figure("3.4", RES / "eda_label_correlation.png",
             "Correlation of each feature with the phishing label (bold: not in the lite set)", width=4.6)
    b.para("EDA also showed that the dataset's coding does not always follow its documentation. For example, "
           "*prefix_suffix* = -1 (\"a '-' in the domain\") occurs in 87% of all rows, and for five features "
           "(such as *shortining_service* and *domain_registration_length*) the documented phishing-like value is "
           "actually less common among phishing sites. This matters for the live demo, whose extractor follows the "
           "documented rules.")

    b.h2("Algorithm / Model Description")
    b.para("The workflow is implemented in Python (src/ folder) and runs with one command "
           "(python -m src.run_all):")
    b.bullets([
        "Load the dataset (ucimlrepo, with an ARFF download as fallback), validate the columns and values, "
        "map the label to phishing = 1 and summarise it (class balance, value counts, duplicates).",
        "Split the data into stratified 80% training and 20% test sets with a fixed seed. The test set is "
        "used only once, for the final evaluation.",
        "For each feature set (full, lite) and each model, tune the hyper-parameters with GridSearchCV and "
        "5-fold stratified cross-validation on the training set, scoring the F1-score of the phishing class.",
        "Refit the best configuration on the whole training set, measure training and single-prediction "
        "time, and save the model.",
        "Evaluate every model on the test set and repeat the experiment on de-duplicated data.",
        "Analyse feature importance and select the lite model with the best CV F1-score for deployment.",
    ], kind="step")

    b.h3("Models Used")
    b.bullets([
        "**Decision Tree:** a single tree of if-then splits chosen to reduce impurity (Gini or entropy). Simple and "
        "interpretable, but prone to overfitting.",
        "**Random Forest [6]:** many decision trees trained on bootstrap samples with random feature subsets; their "
        "votes are averaged (bagging), which reduces variance.",
        "**SVM with RBF kernel [9]:** finds a maximum-margin boundary in a kernel-transformed space; inputs are "
        "standardised, and probability estimates are enabled for ROC curves.",
        "**XGBoost [7]:** gradient boosting that adds trees one at a time, each fitted to the gradient of the "
        "log-loss of the current ensemble, with shrinkage, row/column subsampling and L2 regularisation. The final "
        "score is the sum of the trees' outputs passed through a sigmoid.",
    ])

    b.h3("Model Development and Hyper-parameter Tuning")
    b.para("Table 3.3 lists the hyper-parameter grids and the best configurations found by 5-fold cross-validation. "
           "Grids were kept small so that the whole pipeline runs in about seven minutes on a laptop CPU. All random "
           "seeds are fixed (42), so the results are reproducible.")
    grids = {
        "decision_tree": "criterion {gini, entropy}; max_depth {None, 10, 20}; min_samples_leaf {1, 2, 5}",
        "random_forest": "n_estimators {200, 400}; max_depth {None, 20}; max_features {sqrt, 0.5}",
        "svm": "C {1, 10, 100}; gamma {scale, 0.1, 0.01}",
        "xgboost": "n_estimators {200, 400}; max_depth {4, 6, 8}; learning_rate {0.05, 0.1}",
    }
    best = {(r["feature_set"], r["model"]): r["best_params"] for r in cv}
    b.table("3.3", "Hyper-parameter grids and best configurations (5-fold CV)",
            ["Model", "Grid searched", "Best (full)", "Best (lite)"],
            [[MODEL_LABELS[m], grids[m], params_text(best[("full", m)]), params_text(best[("lite", m)])]
             for m in MODEL_LABELS],
            [1300, 2966, 2200, 2200])

    b.h3("Feature Importance Methods")
    b.bullets([
        "**Built-in importance:** impurity decrease (Decision Tree, Random Forest) and gain (XGBoost).",
        "**Permutation importance [11]:** the drop in test F1-score when one feature's values are shuffled "
        "(10 repeats). It works for every model, including the SVM.",
        "**SHAP [8]:** TreeExplainer computes each feature's contribution to each individual prediction; it is "
        "used for the summary plots and for the explanations returned by the API.",
        "**Ablation:** the best model is retrained on its top-k features (k = 5, 10, 15, 20, 30), ranked by SHAP "
        "on training data.",
        "**Full-vs-lite gap analysis:** the nine excluded features are added back to the lite set one at a time "
        "and cumulatively, measuring how much of the CV F1 gap each recovers.",
    ])

    b.h3("Performance Metrics")
    b.para("The accuracy of the classifier is computed using Equation (3.1), and precision, recall and F1-score "
           "for the phishing class using Equations (3.2) to (3.4):")
    b.equation("Accuracy", "TP+TN", "TP+TN+FP+FN", "3.1")
    b.equation("Precision", "TP", "TP+FP", "3.2")
    b.equation("Recall", "TP", "TP+FN", "3.3")
    b.equation("F1", "2×Precision×Recall", "Precision+Recall", "3.4")
    b.para("where TP, TN, FP and FN denote true positives, true negatives, false positives and false negatives "
           "respectively, with phishing as the positive class. ROC-AUC (the area under the receiver operating "
           "characteristic curve) measures how well the predicted probabilities rank phishing above legitimate "
           "websites across all thresholds. For stability, we report the mean and standard deviation of the "
           "5-fold CV F1-score.")

    b.h2("Tools and Technologies Used")
    b.bullets([
        "Programming language: Python 3.13; HTML, CSS and JavaScript for the web page and extension.",
        "Libraries / frameworks: scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, pandas, NumPy, SciPy, Matplotlib, "
        "joblib; requests, BeautifulSoup4 and tldextract for feature extraction; FastAPI, Pydantic and Uvicorn "
        "for the API; pytest for testing; Chrome Extension Manifest V3; Vercel for hosting.",
        "Hardware: laptop with a 13th Gen Intel Core i9-13980HX CPU (24 cores, 32 threads) and 15.6 GB RAM; "
        "no GPU was used.",
    ])

    # ------------------------------------------------------- Chapter 4
    b.h1("Results and Discussion")
    b.para("This chapter reports the results of the experiments. All numbers are produced by the code in the "
           "GitHub repository and are read from its results/ folder.")

    b.h2("Experimental Setup")
    b.para("All experiments ran on Windows 11 with Python 3.13 on the laptop described in Section 3.4, using the "
           "CPU only. The data was split once into 8,844 training and 2,211 test websites (stratified, seed 42). "
           "Hyper-parameters were tuned with GridSearchCV and 5-fold stratified cross-validation on the training "
           "set (scoring: F1 of the phishing class), with at most 8 parallel worker processes. The decision threshold "
           "is 0.5. Training time is the refit time of the best configuration; inference time is the median of 200 "
           "single-website predictions. The complete pipeline, including the importance analysis, took about seven "
           "minutes, and a second run from an empty folder reproduced every metric exactly.")

    b.h2("Results")
    b.h3("Model Comparison on the Full Feature Set")
    hdr = ["Model", "Acc. (%)", "Prec. (%)", "Recall (%)", "F1 (%)", "ROC-AUC", "CV F1 (mean ± std)",
           "Train (s)", "Pred. (ms)"]
    widths = [1250, 800, 800, 850, 750, 900, 1516, 900, 900]
    b.table("4.1", "Test-set performance on the full feature set (30 features, 2,211 test websites)",
            hdr, metric_rows(full), widths)
    b.para(f"Random Forest, SVM and XGBoost perform almost identically on the full feature set (Table 4.1): their "
           f"accuracies lie within {100 * (full.accuracy.max() - full[full.model != 'decision_tree'].accuracy.min()):.2f} "
           f"percentage points, a difference of only one to three test websites. Random Forest has the highest "
           f"accuracy ({pct(fbest.accuracy)}%), while XGBoost has the highest phishing recall ({pct(fx.recall)}%), "
           f"the highest ROC-AUC ({fx.roc_auc:.4f}) and the most stable cross-validation score. The single Decision "
           f"Tree is clearly weaker. Figures 4.1 and 4.2 show the ROC curves and confusion matrices.")
    b.figure("4.1", RES / "roc_full.png", "ROC curves on the test set (full feature set)", width=6.0)
    b.figure("4.2", RES / "confusion_full.png", "Confusion matrices on the test set (full feature set)", width=6.0)

    b.h3("Model Comparison on the Lite (Deployable) Feature Set")
    b.table("4.2", "Test-set performance on the lite feature set (21 features)", hdr, metric_rows(lite), widths)
    b.para(f"With only the 21 live-computable features, all models lose about three percentage points of accuracy "
           f"(Table 4.2, Figure 4.3). Random Forest and XGBoost tie on CV F1-score ({lx.cv_f1_mean:.4f}); XGBoost "
           f"was selected for deployment because a single prediction takes {lx.inference_ms:.1f} ms instead of "
           f"{lite[lite.model == 'random_forest'].iloc[0].inference_ms:.1f} ms. The choice is made on "
           "cross-validation, not on the test set.")
    b.figure("4.3", RES / "roc_lite.png", "ROC curves on the test set (lite feature set)", width=6.0)

    b.h3("Sensitivity to Duplicate Rows")
    b.para("To measure how much duplicates inflate the scores, each tuned configuration was retrained and "
           "re-tested after removing exact duplicates before splitting (Table 4.3).")
    b.table("4.3", "Accuracy and F1-score on the standard and de-duplicated splits",
            ["Features", "Model", "Train / test rows", "Acc. standard (%)", "Acc. dedup (%)",
             "F1 standard (%)", "F1 dedup (%)", "F1 change"],
            [[r.feature_set, MODEL_LABELS[r.model], f"{r.n_train:,} / {r.n_test:,}", pct(r.accuracy_standard),
              pct(r.accuracy_dedup), pct(r.f1_standard), pct(r.f1_dedup), f"{100 * r.f1_change:+.2f}"]
             for r in dedup.itertuples()],
            [850, 1300, 1300, 1050, 1050, 1050, 1050, 1016])
    b.para(f"Every model loses {-drops.max():.1f} to {-drops.min():.1f} F1 points on unique data, so part of "
           f"the headline accuracy comes from memorising repeated rows. XGBoost is the most robust: it has the "
           f"best full-set accuracy on de-duplicated data ({pct(dfx.accuracy_dedup)}%) and the smallest F1 drop.")

    b.h3("Feature Importance")
    b.para("Permutation importance (Figure 4.4) and SHAP (Figure 4.5) agree that *sslfinal_state* (valid HTTPS) "
           "and *url_of_anchor* (links that point nowhere or to other domains) dominate every model, followed by "
           "*web_traffic*, *prefix_suffix*, *links_in_tags* and *having_sub_domain*. About ten features, such as "
           "*iframe*, *rightclick* and *port*, have almost no effect. Across all eight importance rankings the mean "
           "Spearman rank correlation is 0.82 (Figure 4.6); the only outlier is XGBoost's built-in gain "
           "importance, which is known to over-reward features that split small, pure groups of rows.")
    b.figure("4.4", RES / "report_permutation.png",
             "Permutation importance on the test set (top 15 features per model)", width=6.0)
    b.figure("4.5", RES / "shap_summary.png", "SHAP summary for the Random Forest (500 test websites)", width=4.4)
    b.figure("4.6", RES / "importance_agreement.png", "Agreement between importance methods (Spearman correlation)", width=5.0)
    b.para(f"The ablation in Figure 4.7 shows that five features already give 93% accuracy and fifteen features "
           f"reach {pct(k15.test_accuracy)}%, about 0.7 points below all thirty. Figure 4.8 explains the full-vs-lite "
           "gap: no single excluded feature recovers more than about a fifth of the gap (links_pointing_to_page "
           "21%, web_traffic 18%), but together the excluded reputation features (backlinks, traffic rank, domain "
           "age, Google index, PageRank) recover all of it. The missing information is the site's reputation.")
    b.figure("4.7", RES / "ablation.png", "Ablation: Random Forest on its top-k features", width=4.8)
    b.figure("4.8", RES / "lite_gap.png", "Share of the full-vs-lite gap recovered by each excluded feature", width=4.8)

    b.h3("Sample Results of the Deployed Application")
    b.para(f"Figures 4.9 to 4.11 show the web application. A real legitimate site checked on the public "
           f"deployment is classified as legitimate with its main reasons (Figure 4.9). A harmless fake phishing page "
           f"built for testing (an \"account verification\" form that submits nowhere, links that go nowhere, "
           f"a hidden iframe and disabled right-click) is classified as phishing with 99.5% probability "
           f"(Figure 4.10). A modern, cleanly built fake page is **missed** (Figure 4.11): it looks technically "
           f"clean, and this shows the main limitation of a model trained on 2015 data. In a check of "
           f"{live['n_sites']} popular legitimate websites, none was flagged as phishing.")
    b.figure("4.9", RES / "screenshot_live_legit.png",
             "Sample output: a legitimate website checked on the public deployment", width=5.0)
    b.figure("4.10", RES / "screenshot_local_phishing.png",
             "Sample output: the fake phishing test page is detected (99.5%)", width=5.0)
    b.figure("4.11", RES / "screenshot_local_modern.png",
             "Sample output: a modern, clean fake phishing page is missed (limitation)", width=5.0)

    b.h2("Comparison with Existing Methods")
    b.para("Table 4.4 compares the proposed method with existing methods on the same dataset. The numbers for "
           "existing methods are taken from the papers; they use different splits and protocols, so small "
           "differences should not be over-interpreted. Values reported with two decimals in the source papers "
           "are shown as percentages.")
    fr = full[full.model == "random_forest"].iloc[0]
    b.table("4.4", "Performance comparison of the proposed method with existing methods",
            ["Method", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"],
            [["Extra Trees, Awasthi & Goel [3]", "98.73", "98", "97", "97"],
             ["Tuned CNN, Almujahid et al. [4]", "99", "99", "98", "99"],
             ["Stacking ensemble, Newaz et al. [5]", "97.49", "97.11", "98.43", "97.76"],
             ["Proposed: Random Forest (full, 30 features)", pct(fr.accuracy), pct(fr.precision), pct(fr.recall), pct(fr.f1)],
             ["Proposed: XGBoost (full, 30 features)", pct(fx.accuracy), pct(fx.precision), pct(fx.recall), pct(fx.f1)],
             ["Proposed: XGBoost (lite, 21 live features, deployed)", pct(lx.accuracy), pct(lx.precision), pct(lx.recall), pct(lx.f1)]],
            [3466, 1300, 1300, 1300, 1300])
    b.para("On the full feature set, the proposed models are within about one point of the best published results, "
           "with simple, tuned models and a stricter protocol (a held-out test set used once). The project's main "
           "contributions lie elsewhere: it measures the effect of duplicate rows (which the compared studies do "
           "not), explains the predictions, quantifies the cost of using only deployable features, and delivers a "
           "working, explainable application.")

    b.h2("Deployment Details")
    b.bullets([
        "**Feature extractor (src/extract.py):** computes the 21 lite features for a URL with the dataset authors' "
        "encoding rules: 9 from the URL string, a DNS lookup, a TLS certificate check and 10 from the page's static "
        "HTML (no JavaScript is executed). It uses 5-second timeouts and a 2 MB size limit; if the page cannot be "
        "fetched, the missing features are filled with their most common training value and the result is marked "
        "as partial.",
        "**Prediction API (FastAPI):** POST /predict returns the label, the phishing probability, all 21 features "
        "and the top five reasons (per-prediction SHAP values computed by XGBoost's built-in TreeSHAP); GET /health "
        "and a try-it web page are also provided. Inputs are validated (http/https only).",
        "**Chrome extension (Manifest V3):** a popup that sends the current tab's URL to the local API and shows the "
        "verdict and reasons. It requests only the activeTab permission and access to the local API.",
        f"**Public deployment (Vercel):** a slim version of the API is hosted at {LIVE}. It refuses private and "
        "internal addresses (including through redirects) so that it cannot be used to probe the server's network, "
        "does not include the fake test pages, and limits each client to 20 checks per minute.",
        "**Testing:** 118 automated tests (pytest) cover every feature rule, the API, the public-mode safeguards and "
        "the saved models.",
    ])

    b.h2("Link of GitHub Repository")
    b.para(f"The source code, trained models, results and documentation are available at: {GITHUB}")
    b.para(f"The deployed web application is available at: {LIVE}")

    # ------------------------------------------------------- Chapter 5
    b.h1("Conclusion and Future Scope")
    b.h2("Conclusion")
    b.para(f"This project built and evaluated an ML system for phishing website detection. On the UCI Phishing "
           f"Websites dataset, Random Forest, SVM and XGBoost reach about 97.6% test accuracy, with Random Forest "
           f"highest at {pct(fbest.accuracy)}% and XGBoost best on phishing recall and ROC-AUC; the Decision Tree is "
           f"about one point lower. Using only the 21 features that can be computed live costs about three points "
           f"(94.5-94.8%), and this gap is explained by the missing site-reputation features as a group. Built-in, "
           f"permutation and SHAP importance agree that valid HTTPS and the destination of a page's links are the "
           f"strongest signals. Removing the 47% duplicate rows lowers F1 by 2-4 points, which gives a more honest "
           f"estimate of performance on unseen websites. The best lite model, XGBoost, was deployed as an explained "
           f"web service with a web page, a Chrome extension and a public website, meeting all the objectives of "
           f"Section 1.4.")
    b.h2("Future Scope")
    b.bullets([
        "Train on a recent, larger dataset collected with the same extractor that is used in deployment, so that "
        "training and live features match exactly.",
        "Add modern reputation signals that can be computed today, such as domain age from RDAP, popularity from "
        "the Tranco list and certificate-transparency logs.",
        "Render pages in a headless browser to analyse JavaScript-built content, and add visual similarity to known "
        "brands to catch modern, cleanly built phishing kits.",
        "Evaluate the system on a live feed of verified phishing URLs, and combine the model with blocklists such as "
        "Google Safe Browsing.",
        "Publish the Chrome extension with a configurable API address and add user feedback for continual learning.",
    ])

    # ------------------------------------------------------- Chapter 6
    b.h1("List of Publications")
    b.h2("List of Published Papers")
    b.para("No paper has been published from this work so far.")
    b.h2("Details of Participation in Project Competitions")
    b.para("The project has not yet been presented at a project competition.")


REFERENCES = [
    "R. M. Mohammad, F. Thabtah, and L. McCluskey, “Predicting phishing websites based on self-structuring "
    "neural network,” *Neural Computing and Applications*, vol. 25, no. 2, pp. 443–458, 2014.",
    "R. M. Mohammad, F. Thabtah, and L. McCluskey, “Phishing Websites,” UCI Machine Learning Repository, 2015. "
    "[Online]. Available: https://archive.ics.uci.edu/dataset/327/phishing+websites",
    "A. Awasthi and N. Goel, “Phishing website prediction using base and ensemble classifier techniques with "
    "cross-validation,” *Cybersecurity*, vol. 5, art. 22, 2022, doi: 10.1186/s42400-022-00126-9.",
    "N. F. Almujahid, M. A. Haq, and M. Alshehri, “Comparative evaluation of machine learning algorithms for "
    "phishing site detection,” *PeerJ Computer Science*, vol. 10, e2131, 2024, doi: 10.7717/peerj-cs.2131.",
    "A. Newaz, F. S. Haq, and N. Ahmed, “A sophisticated framework for the accurate detection of phishing "
    "websites,” arXiv preprint arXiv:2403.09735, 2024.",
    "L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. "
    "Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
    "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural "
    "Information Processing Systems 30*, 2017.",
    "C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, "
    "vol. 12, pp. 2825–2830, 2011.",
    "A. Altmann, L. Toloşi, O. Sander, and T. Lengauer, “Permutation importance: a corrected feature importance "
    "measure,” *Bioinformatics*, vol. 26, no. 10, pp. 1340–1347, 2010.",
]

ABBREVIATIONS = [
    ("API", "Application Programming Interface"), ("AUC", "Area Under the Curve"),
    ("CNN", "Convolutional Neural Network"), ("CV", "Cross-Validation"), ("DNS", "Domain Name System"),
    ("DT", "Decision Tree"), ("EDA", "Exploratory Data Analysis"), ("FN / FP", "False Negative / False Positive"),
    ("HTML", "HyperText Markup Language"), ("HTTPS", "HyperText Transfer Protocol Secure"),
    ("ML", "Machine Learning"), ("RBF", "Radial Basis Function"), ("RF", "Random Forest"),
    ("ROC", "Receiver Operating Characteristic"), ("SHAP", "SHapley Additive exPlanations"),
    ("SSRF", "Server-Side Request Forgery"), ("SVM", "Support Vector Machine"),
    ("TLS / SSL", "Transport Layer Security / Secure Sockets Layer"), ("TN / TP", "True Negative / True Positive"),
    ("UCI", "University of California, Irvine (ML repository)"), ("URL", "Uniform Resource Locator"),
    ("WHOIS", "Domain registration lookup protocol"), ("XGBoost", "eXtreme Gradient Boosting"),
]

ABSTRACT = (
    "Phishing websites imitate trusted organisations to steal passwords and payment details, and they appear "
    "faster than blocklists can track them. This project develops and deploys a machine learning system that "
    "classifies a website as phishing or legitimate. Using the UCI Phishing Websites dataset (11,055 websites, "
    "30 features), we train and tune four classifiers, namely Decision Tree, Random Forest, Support Vector Machine "
    "and XGBoost, with 5-fold cross-validation on two feature sets: all 30 features, and the 21 features that a "
    "program can compute from a live URL and its HTML. On the held-out test set, Random Forest reaches 97.69% "
    "accuracy and XGBoost 97.60% with the best recall and ROC-AUC (0.9969); the deployable feature set reaches "
    "94.5–94.8%. Exploratory analysis revealed that 47% of the rows are duplicates; removing them lowers the "
    "F1-score by 2–4 points, a more realistic estimate. Permutation importance and SHAP agree that valid HTTPS "
    "and the destination of a page's links are the strongest signals, and that the deployable model's gap comes "
    "from missing site-reputation features. The best deployable model, XGBoost, is served by a FastAPI service "
    "with a web page, a Chrome extension and a public website, and explains every decision with SHAP values. A "
    "modern, cleanly built phishing page is still missed, which highlights the need for fresher training data."
)
KEYWORDS = "Phishing detection, Machine learning, XGBoost, Random Forest, SHAP, Web security"


# ------------------------------------------------------------------ main

def main(template, output):
    doc = Document(template)
    body = doc.element.body
    paras = [el for el in body if el.tag == W("w:p")]

    # Cover and certificate
    for p in paras:
        t = text_of(p).strip()
        if t == "Title of the Project":
            set_text(p, TITLE)
        elif t == "“Title of the Project”":
            set_text(p, f"“{TITLE}”")
        elif t.startswith("Dr. / Prof."):
            set_text(p, GUIDE)
    name_paras = [p for p in paras if text_of(p).strip() == "Surname Firstname Middlename (Roll No.)"]
    assert len(name_paras) == 8, len(name_paras)
    for group in (name_paras[:4], name_paras[4:]):
        for p, name in zip(group, MEMBERS):
            set_text(p, name)
        body.remove(group[3])  # three members, template has four lines
    cert_guide = [p for p in paras if text_of(p).strip() == "Project Guide"][-1]
    nxt = cert_guide.getnext()
    if not text_of(nxt).strip():
        # centred like the "Project Guide" line above it, but not bold
        old = nxt.find(W("w:pPr"))
        if old is not None:
            nxt.remove(old)
        nxt.insert(0, copy.deepcopy(cert_guide.find(W("w:pPr"))))
        strip_runs(nxt)
        nxt.append(make_run(GUIDE))

    # Abstract and keywords
    abstract = next(p for p in paras if text_of(p).startswith("[Write the abstract"))
    set_text(abstract, ABSTRACT)
    kw = next(p for p in paras if text_of(p).startswith("Keywords:"))
    strip_runs(kw)
    kw.append(make_run("Keywords: ", bold=True))
    kw.append(make_run(KEYWORDS))

    # Prototypes (copied before the body is cleared)
    def find(pred):
        return next(el for el in body if pred(el))

    h1 = find(lambda el: style_of(el) == "Heading1" and text_of(el).strip() == "Introduction")
    tbl = find(lambda el: el.tag == W("w:tbl"))
    trs = tbl.findall(W("w:tr"))
    hdr_tc, row_tc = trs[0].find(W("w:tc")), trs[1].find(W("w:tc"))

    def first_rpr(tc):
        r = next((r for r in tc.iter(W("w:r")) if r.find(W("w:t")) is not None), None)
        return copy.deepcopy(r.find(W("w:rPr"))) if r is not None and r.find(W("w:rPr")) is not None else None

    fcap = find(lambda el: style_of(el) == "FigureCaption")
    protos = {
        "h1": copy.deepcopy(h1),
        "h2": copy.deepcopy(find(lambda el: style_of(el) == "Heading2")),
        "h3": copy.deepcopy(find(lambda el: style_of(el) == "Heading3")),
        "body": copy.deepcopy(h1.getnext()),
        "bullet": copy.deepcopy(find(lambda el: text_of(el).startswith("To study"))),
        "step": copy.deepcopy(find(lambda el: text_of(el).startswith("Step 1"))),
        "ref": copy.deepcopy(find(lambda el: text_of(el).startswith("D. Xue"))),
        "tbl": copy.deepcopy(tbl),
        "hdr_tr": copy.deepcopy(trs[0]), "row_tr": copy.deepcopy(trs[1]),
        "hdr_tc": copy.deepcopy(hdr_tc), "row_tc": copy.deepcopy(row_tc),
        "hdr_rpr": first_rpr(hdr_tc), "row_rpr": first_rpr(row_tc),
        "tcap": copy.deepcopy(find(lambda el: style_of(el) == "TableCaption")),
        "fcap": copy.deepcopy(fcap),
        "img": copy.deepcopy(fcap.getprevious()),
        "eq": copy.deepcopy(find(lambda el: el.find(f".//{qn('m:oMath')}") is not None)),
    }
    refs_heading = find(lambda el: style_of(el) == "FrontMatterTitleinTOC" and text_of(el).strip() == "References")
    hosted = find(lambda el: "Hosted website link" in text_of(el))
    refs_heading_copy, hosted_copy = copy.deepcopy(refs_heading), copy.deepcopy(hosted)

    # Lists of figures / tables -> Word fields built from the caption styles
    lof = find(lambda el: style_of(el) == "FrontMatterTitleinTOC" and text_of(el).strip() == "List of Figures")
    lot = find(lambda el: style_of(el) == "FrontMatterTitleinTOC" and text_of(el).strip() == "List of Tables")
    sect_para = find(lambda el: el.tag == W("w:p") and el.find(f"{W('w:pPr')}/{W('w:sectPr')}") is not None)
    for start, stop in ((lof, lot), (lot, sect_para)):
        el = start.getnext()
        while el is not stop:
            nxt_el = el.getnext()
            body.remove(el)
            el = nxt_el
    # \t takes style *names*, not style ids
    lof.addnext(field_paragraph('TOC \\h \\z \\t "Figure Caption,1"', "Update field to build the list of figures."))
    lot.addnext(field_paragraph('TOC \\h \\z \\t "Table Caption,1"', "Update field to build the list of tables."))

    # Abbreviations page (listed in the template's Contents), before the section break
    abbr_heading = copy.deepcopy(lot)
    set_text(abbr_heading, "Abbreviations")
    sect_para.addprevious(abbr_heading)
    front = Builder(doc, protos, abbr_heading)
    front.table("A.1", "Abbreviations used in this report", ["Abbreviation", "Meaning"],
                [list(a) for a in ABBREVIATIONS], [2400, 6266])
    # The abbreviations table caption is not a numbered report table: make it a plain bold line.
    cap = abbr_heading.getnext()
    body.remove(cap)

    # Clear the template's chapters and rebuild them
    final_sect = body.find(W("w:sectPr"))
    el = h1
    while el is not None and el is not final_sect:
        nxt_el = el.getnext()
        body.remove(el)
        el = nxt_el

    b = Builder(doc, protos, sect_para)
    write_body(b, load_results())
    b.element(refs_heading_copy)
    b.references(REFERENCES)
    set_text(hosted_copy, f"Hosted website link: {LIVE}")
    b.element(hosted_copy)

    doc.core_properties.title = TITLE
    doc.core_properties.author = ", ".join(n.split(" (")[0] for n in MEMBERS)
    doc.save(output)
    print("Saved", output)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
