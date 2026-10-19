// Sends the active tab's URL to the local API and shows the verdict.
// Nothing is sent anywhere except http://127.0.0.1:8000, and only when the
// popup is opened.
const API = "http://127.0.0.1:8000/predict";
const TIMEOUT_MS = 30000;

const out = document.getElementById("out");

function el(tag, cls, ...children) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const c of children) e.append(c);
  return e;
}

function showError(...parts) {
  out.className = "error";
  out.replaceChildren(...parts);
}

function render(r) {
  const phishing = r.label === "phishing";
  const parts = [
    el("div", "",
      el("span", "badge " + r.label, phishing ? "⚠ Likely phishing" : "✓ Likely legitimate"),
      el("span", "prob", `${(r.phishing_probability * 100).toFixed(1)}% phishing`)),
  ];
  if (r.partial_features) {
    parts.push(el("div", "warn",
      `Partial result: ${r.unavailable_features.length} page features could not be computed.`));
  }
  const ul = el("ul");
  for (const x of r.top_reasons) {
    const up = x.contribution > 0;
    ul.append(el("li", "",
      el("span", up ? "up" : "down", up ? "▲ phishing" : "▼ legitimate"),
      ` ${x.checks}: ${x.meaning}`));
  }
  parts.push(el("h2", "", "Top reasons"), ul);
  parts.push(el("p", "muted", `Model: ${r.model} (lite features, static HTML only).`));
  out.className = "";
  out.replaceChildren(...parts);
}

async function main() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab && tab.url;
  document.getElementById("url").textContent = url || "";
  if (!url || !/^https?:\/\//i.test(url)) {
    showError("Only http:// and https:// pages can be checked.");
    return;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") {
      showError("The API took too long to answer.");
    } else {
      showError("Cannot reach the API at 127.0.0.1:8000. Start it from the project folder with ",
        el("code", "", "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"));
    }
    return;
  } finally {
    clearTimeout(timer);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = Array.isArray(body.detail) ? body.detail.map(d => d.msg).join("; ") : body.detail;
    showError(`API error ${res.status}: ${msg || res.statusText}`);
    return;
  }
  render(body);
}

main();
