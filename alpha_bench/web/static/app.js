import { divergingBars, lineChart } from "./charts.js";

// =====================================================================
// utilities
// =====================================================================
const $ = (sel, root = document) => root.querySelector(sel);

/** DOM builder. Strings become text nodes (never HTML), so untrusted data is safe. */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "style") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    else if (k === "value") el.value = v;
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function api(path, { method = "GET", body } = {}) {
  const resp = await fetch(path, {
    method, credentials: "same-origin",
    headers: { "X-Alpha-Bench": "1", ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await resp.json(); } catch { /* empty */ }
  if (!resp.ok) throw new Error((data && data.error) || `Request failed (${resp.status})`);
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3500);
}

const isNum = (v) => v != null && isFinite(v);
const num = (v, d = 2) => (isNum(v) ? Number(v).toFixed(d) : "–");
const pct = (v, d = 1) => (isNum(v) ? (v * 100).toFixed(d) + "%" : "–");
const signed = (v, f = num) => (isNum(v) ? (v > 0 ? "+" : "") + f(v) : "–");
const money = (v) => (isNum(v) ? (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + "M" : (v / 1e3).toFixed(0) + "K") : "–");
const PCT_METRICS = new Set(["returns", "turnover", "drawdown", "max_weight", "coverage", "max_sector_net"]);
const fmtMetric = (k, v) => (PCT_METRICS.has(k) ? pct(v) : k === "margin" ? num(v, 1) + " bps" :
  k.endsWith("_count") ? num(v, 0) : num(v));

/** replaceChildren that skips null/false parts (replaceChildren would print them as text). */
function fill(el, ...kids) {
  el.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));
}

function badge(result) {
  const r = String(result || "PENDING").toUpperCase();
  return h("span", { class: `badge ${r}`, text: r });
}

function busy(text) { return h("span", { class: "busy", text }); }

function errorBox(err) { return h("div", { class: "err-box", role: "alert", text: err.message || String(err) }); }

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied."); } catch { toast("Copy failed; select and copy manually."); }
}

function download(name, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// =====================================================================
// app state
// =====================================================================
const S = { meta: null, state: null, authState: null, verifyTimer: null, tab: "create", resultViews: new Set(),
  history: [], lastJobs: {} };

// ---------------------------------------------------------------- theme
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("ab-theme"); } catch { /* storage blocked */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("ab-theme", next); } catch { /* ignore */ }
  });
}

// ----------------------------------------------------------------- tabs
function initTabs() {
  for (const btn of document.querySelectorAll(".tabs button")) {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  }
}
function showTab(name) {
  S.tab = name;
  for (const btn of document.querySelectorAll(".tabs button")) btn.setAttribute("aria-selected", btn.dataset.tab === name);
  for (const p of document.querySelectorAll(".tab-panel")) p.hidden = p.id !== `tab-${name}`;
  if (name === "library") refreshLibrary();
  if (name === "correlation") refreshPool();
  if (name === "settings") renderSettings();
}

// =====================================================================
// polling: auth pill, jobs, banner
// =====================================================================
async function poll() {
  try {
    const st = await api("/api/state");
    S.state = st;
    updatePill(st.auth);
    updateJobs(st.jobs);
    const banner = $("#banner");
    if (st.load_error) { banner.hidden = false; banner.textContent = st.load_error; }
    else if (!st.engine_ready) { banner.hidden = false; banner.textContent = "Preparing the offline simulation engine (first launch builds the synthetic market, about 10 seconds)..."; }
    else banner.hidden = true;
    if (st.auth.state !== S.authState) {
      S.authState = st.auth.state;
      if ($("#auth-dialog").open) renderAuthDialog(st.auth);
      if (st.auth.state === "verification_needed") startVerifyPolling();
      if (st.auth.state === "session_expired" && st.jobs.some((j) => j.state === "paused")) {
        toast("Session expired: live jobs are paused. Sign in to resume them.");
      }
    }
  } catch (e) {
    const banner = $("#banner");
    banner.hidden = false;
    banner.textContent = "Lost contact with the Alpha Bench server. Is it still running? " + e.message;
  }
}

function updatePill(auth) {
  const pill = $("#auth-pill");
  pill.dataset.state = auth.state;
  let label = auth.label;
  if (auth.state === "signed_in") label = `Signed in${auth.user_id ? " · " + auth.user_id : ""}`;
  $(".label", pill).textContent = label;
  pill.title = auth.message || "";
}

function updateJobs(jobs) {
  const pill = $("#jobs-pill");
  const live = jobs.filter((j) => !["done", "error", "cancelled"].includes(j.state));
  pill.hidden = jobs.length === 0;
  pill.textContent = live.length ? `BRAIN jobs · ${live.length} running` : `BRAIN jobs · ${jobs.length}`;
  for (const j of jobs) {
    const prev = S.lastJobs[j.id];
    if (prev && prev !== j.state && ["done", "error"].includes(j.state)) {
      toast(j.state === "done" ? `BRAIN simulation finished (${j.brain_alpha_id}).` : `BRAIN job failed: ${j.error}`);
      for (const view of S.resultViews) view.onJob(j);
    }
    S.lastJobs[j.id] = j.state;
  }
  if ($("#jobs-dialog").open) renderJobsDialog(jobs);
}

// =====================================================================
// sign-in dialog
// =====================================================================
function openAuth() {
  const dlg = $("#auth-dialog");
  renderAuthDialog(S.state ? S.state.auth : { state: "signed_out", label: "Signed out" });
  if (!dlg.open) dlg.showModal();
}

async function authCall(path, body, msgEl) {
  try {
    const status = await api(path, { method: "POST", body });
    S.state && (S.state.auth = status);
    S.authState = status.state;
    updatePill(status);
    renderAuthDialog(status);
    if (status.state === "verification_needed") startVerifyPolling();
    return status;
  } catch (e) {
    if (msgEl) msgEl.replaceChildren(errorBox(e));
    else toast(e.message);
    poll();
    return null;
  }
}

function startVerifyPolling() {
  if (S.verifyTimer) return;
  const prefs = (S.state && S.state.prefs) || {};
  const every = (prefs.verification_poll_seconds || 7) * 1000;
  const cap = (prefs.verification_poll_cap_seconds || 240) * 1000;
  const started = Date.now();
  S.verifyTimer = setInterval(async () => {
    if (Date.now() - started > cap || S.authState !== "verification_needed") {
      clearInterval(S.verifyTimer); S.verifyTimer = null;
      if ($("#auth-dialog").open && S.authState === "verification_needed") {
        const note = $("#verify-note");
        if (note) note.textContent = "Stopped checking automatically. Click “I've completed verification” when you're done.";
      }
      return;
    }
    try {
      const st = await api("/api/auth/verify-check", { method: "POST" });
      S.authState = st.state;
      updatePill(st);
      if (st.state !== "verification_needed") {
        clearInterval(S.verifyTimer); S.verifyTimer = null;
        if ($("#auth-dialog").open) renderAuthDialog(st);
        if (st.state === "signed_in") toast("Verified. Signed in to BRAIN.");
      } else {
        const note = $("#verify-note");
        if (note) note.textContent = st.message;
      }
    } catch (e) {
      clearInterval(S.verifyTimer); S.verifyTimer = null;
      poll();
      const note = $("#verify-note");
      if (note) note.textContent = e.message;
    }
  }, every);
}

function renderAuthDialog(auth) {
  const dlg = $("#auth-dialog");
  const msg = h("div");
  const close = h("button", { class: "btn", type: "button", text: "Close", on: { click: () => dlg.close() } });
  let body;
  const st = auth.state;

  if (st === "signed_in") {
    body = [
      h("h2", { id: "auth-title", text: "BRAIN account" }),
      h("div", { class: "ok-box", text: auth.message }),
      auth.token_expiry_seconds ? h("p", { class: "muted", text: `Session token valid for about ${Math.round(auth.token_expiry_seconds / 60)} more minutes.` }) : null,
      h("p", { class: "sub", text: "Live simulations, pulling your alphas and field/operator sync use this session. Alpha Bench never submits alphas: you submit on the BRAIN website yourself." }),
      msg,
      h("div", { class: "actions" },
        h("button", { class: "btn", type: "button", text: "Sync fields & operators", on: { click: () => { dlg.close(); startTask("/api/meta/sync", "Syncing fields and operators"); } } }),
        h("button", { class: "btn", type: "button", text: "Load my active alphas", on: { click: () => { dlg.close(); startTask("/api/correlation/load-active", "Loading your active alphas"); } } }),
        h("button", { class: "btn danger", type: "button", text: "Sign out and wipe local session",
          on: { click: () => authCall("/api/auth/signout", undefined, msg) } }),
        close),
    ];
  } else if (st === "verification_needed") {
    const url = auth.verification_url || "";
    body = [
      h("h2", { id: "auth-title", text: "Verification required" }),
      h("p", { text: "BRAIN wants to confirm it's really you with a quick face check (run by BRAIN's identity provider, Persona). You complete it yourself in your own browser. Alpha Bench never sees your camera, face or ID." }),
      h("ul", { class: "tips" },
        h("li", { text: "Use good, even lighting and face the camera directly." }),
        h("li", { text: "Have your ID document ready in case it's requested." }),
        h("li", { text: "On a computer without a camera, copy the link and open it on your phone." })),
      h("div", { class: "link-box" },
        h("input", { type: "text", readOnly: true, value: url, "aria-label": "Verification link" }),
        h("button", { class: "btn", type: "button", text: "Copy link", on: { click: () => copy(url) } })),
      h("p", { id: "verify-note", class: "muted", text: "Alpha Bench checks every few seconds whether you've finished." }),
      msg,
      h("div", { class: "actions" },
        h("button", { class: "btn", type: "button", text: "Restart sign-in", on: { click: () => authCall("/api/auth/restart", undefined, msg) } }),
        h("button", { class: "btn", type: "button", text: "Open verification in browser",
          on: { click: () => window.open(url, "_blank", "noopener,noreferrer") } }),
        h("button", { class: "btn primary", type: "button", text: "I've completed verification",
          on: { click: () => authCall("/api/auth/verify-check", undefined, msg) } })),
    ];
  } else if (st === "verification_expired") {
    body = [
      h("h2", { id: "auth-title", text: "Verification link expired" }),
      h("p", { text: auth.message }),
      msg,
      h("div", { class: "actions" }, close,
        h("button", { class: "btn primary", type: "button", text: "Restart sign-in for a fresh link",
          on: { click: () => authCall("/api/auth/restart", undefined, msg) } })),
    ];
  } else if (st === "locked") {
    body = [
      h("h2", { id: "auth-title", text: "Sign-in paused" }),
      h("div", { class: "err-box", text: auth.message }),
      h("p", { class: "muted", text: "This protects your BRAIN account from being locked out by repeated wrong passwords. Check your password on the BRAIN website meanwhile." }),
      h("div", { class: "actions" }, close),
    ];
  } else {
    const email = h("input", { type: "email", name: "email", autocomplete: "username", required: true,
      value: auth.remembered_email || "" });
    const pw = h("input", { type: "password", name: "password", autocomplete: "current-password", required: true });
    const remember = h("input", { type: "checkbox", checked: !!auth.can_remember, disabled: !auth.can_remember });
    const form = h("form", { on: { submit: async (e) => {
      e.preventDefault();
      msg.replaceChildren(busy("Signing in..."));
      const res = await authCall("/api/auth/signin", { email: email.value, password: pw.value, remember: remember.checked }, msg);
      pw.value = "";
      if (res && res.state === "signed_in") toast(res.message);
    } } },
      h("label", { class: "field" }, "BRAIN email", email),
      h("label", { class: "field" }, "Password", pw),
      h("label", { class: "check-row" }, remember,
        h("span", { text: auth.can_remember ? "Remember me on this computer (password in the OS keyring, session cookies encrypted). Untick for “don't remember me”."
          : "Remember me (unavailable: no OS keyring found, so you'll sign in each launch)" })),
      msg,
      h("div", { class: "actions" },
        auth.has_saved_password ? h("button", { class: "btn", type: "button", text: "Use saved credentials",
          on: { click: () => authCall("/api/auth/signin-saved", undefined, msg) } }) : null,
        close,
        h("button", { class: "btn primary", type: "submit", text: "Sign in" })));
    body = [
      h("h2", { id: "auth-title", text: st === "session_expired" ? "Session expired" : "Sign in to BRAIN" }),
      h("p", { class: "sub", text: auth.message }),
      st === "offline" ? h("div", { class: "warn-box", text: "Offline tools keep working. Live features need BRAIN." }) : null,
      form,
      h("p", { class: "muted", text: "Your password goes only to api.worldquantbrain.com over HTTPS. It is never logged or written to disk (unless you choose Remember me, which uses the OS keyring)." }),
    ];
  }
  dlg.replaceChildren(...body.filter(Boolean));
}

// =====================================================================
// background tasks
// =====================================================================
async function startTask(path, label, onDone) {
  let task;
  try { task = await api(path, { method: "POST" }); } catch (e) { toast(e.message); return; }
  toast(`${label}...`);
  const tick = async () => {
    let t;
    try { t = await api(`/api/tasks/${task.id}`); } catch { return; }
    for (const el of document.querySelectorAll(`[data-task-kind="${t.kind}"]`)) {
      el.replaceChildren(h("div", { class: "muted", text: t.message }),
        h("div", { class: "progress" }, h("div", { style: { width: `${Math.round(t.progress * 100)}%` } })));
    }
    if (t.state === "running") { setTimeout(tick, 1500); return; }
    if (t.state === "done") toast(`${label}: done.`);
    else toast(`${label} failed: ${t.error}`);
    for (const el of document.querySelectorAll(`[data-task-kind="${t.kind}"]`)) {
      el.replaceChildren(t.state === "done" ? h("div", { class: "ok-box", text: `${label}: done. ${summarizeResult(t.result)}` })
        : errorBox(new Error(t.error)));
    }
    S.meta = null;
    if (onDone) onDone(t);
  };
  setTimeout(tick, 800);
}

function summarizeResult(r) {
  if (!r) return "";
  if (r.fields != null) return `${r.fields} fields and ${r.operators.available} operators available to your account.`;
  if (r.alphas != null) return `${r.alphas} active alphas loaded (${r.with_pnl} with PnL).`;
  if (r.checks) return `${r.checks.length} checks returned.`;
  return "";
}

// =====================================================================
// jobs dialog
// =====================================================================
function renderJobsDialog(jobs) {
  const dlg = $("#jobs-dialog");
  const rows = jobs.map((j) => h("tr", {},
    h("td", {}, h("div", { class: "mono", text: j.expression.slice(0, 70) + (j.expression.length > 70 ? "…" : "") }),
      j.error ? h("div", { class: "why", text: j.error }) : null),
    h("td", { text: j.state }),
    h("td", {}, j.brain_alpha_id ? h("a", { href: `https://platform.worldquantbrain.com/alpha/${encodeURIComponent(j.brain_alpha_id)}`,
      target: "_blank", rel: "noopener noreferrer", text: j.brain_alpha_id }) : "–"),
    h("td", {}, ["queued", "starting", "running", "paused"].includes(j.state)
      ? h("button", { class: "btn small", text: "Cancel", on: { click: async () => { await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); poll(); } } }) : null)));
  dlg.replaceChildren(
    h("h2", { id: "jobs-title", text: "Live BRAIN simulations" }),
    h("p", { class: "sub", text: `At most ${S.state ? S.state.prefs.max_concurrent_simulations : 3} run at once (change in Settings). Polling follows BRAIN's Retry-After guidance.` }),
    h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Expression", "State", "BRAIN alpha", ""].map((t) => h("th", { text: t })))), h("tbody", {}, rows))),
    h("div", { class: "actions" }, h("button", { class: "btn", text: "Close", on: { click: () => dlg.close() } })));
}

// =====================================================================
// shared: settings form
// =====================================================================
async function meta() {
  if (!S.meta) S.meta = await api("/api/meta");
  return S.meta;
}

function settingsForm(initial = {}, { compact = false } = {}) {
  const m = S.meta;
  const d = { ...m.defaults, ...initial };
  const sel = (name, opts, val) => h("select", { name }, opts.map((o) => h("option", { value: String(o), selected: String(o) === String(val), text: String(o) })));
  const f = {
    region: sel("region", m.regions, d.region),
    universe: sel("universe", m.universes, d.universe),
    delay: sel("delay", [1, 0], d.delay),
    decay: h("input", { type: "number", min: 0, max: 60, step: 1, value: d.decay, style: { width: "80px" } }),
    neutralization: h("select", { name: "neutralization" },
      h("optgroup", { label: "Group" }, m.neutralizations.map((o) => h("option", { value: o, selected: o === d.neutralization, text: o }))),
      h("optgroup", { label: "Risk model (BRAIN; approximated offline)" },
        (m.risk_neutralizations || []).map((o) => h("option", { value: o, selected: o === d.neutralization, text: o })))),
    test_years: h("input", { type: "number", min: 0, max: 5, step: 1, style: { width: "70px" },
      value: String(parseInt((String(d.test_period || "P0Y").match(/P(\d+)Y/) || [0, 0])[1], 10)) }),
    truncation: h("input", { type: "number", min: 0, max: 1, step: 0.01, value: d.truncation, style: { width: "90px" } }),
    pasteurization: sel("pasteurization", ["ON", "OFF"], d.pasteurization),
    nan_handling: sel("nan_handling", ["OFF", "ON"], d.nan_handling),
  };
  const main = h("div", { class: "row" },
    h("label", { class: "field" }, "Region", f.region),
    h("label", { class: "field" }, "Universe", f.universe),
    h("label", { class: "field" }, "Delay", f.delay),
    compact ? null : h("label", { class: "field" }, "Decay", f.decay),
    compact ? null : h("label", { class: "field" }, "Neutralization", f.neutralization),
    compact ? null : h("label", { class: "field" }, "Truncation (max stock weight)", f.truncation));
  const adv = h("details", { class: "adv" }, h("summary", { text: "More settings" }),
    h("div", { class: "row" },
      compact ? h("label", { class: "field" }, "Decay", f.decay) : null,
      compact ? h("label", { class: "field" }, "Neutralization", f.neutralization) : null,
      compact ? h("label", { class: "field" }, "Truncation", f.truncation) : null,
      h("label", { class: "field" }, "Pasteurization", f.pasteurization),
      h("label", { class: "field" }, "NaN handling", f.nan_handling),
      h("label", { class: "field", title: "Learn2Quant's settings dialog uses a 1-year test period" }, "Test period (years)", f.test_years)));
  return {
    el: h("div", {}, main, adv),
    get: () => ({ region: f.region.value, universe: f.universe.value, delay: +f.delay.value, decay: +f.decay.value,
      neutralization: f.neutralization.value, truncation: +f.truncation.value, pasteurization: f.pasteurization.value,
      nan_handling: f.nan_handling.value, test_period: `P${Math.max(0, parseInt(f.test_years.value || "0", 10))}Y0M0D` }),
  };
}

// =====================================================================
// shared: result view
// =====================================================================
const CHECK_TO_METRIC = { RISK_TURNOVER: "turnover", RISK_WEIGHT: "max_weight", RISK_SECTOR: "max_sector_net",
  LOW_SHARPE: "sharpe", LOW_FITNESS: "fitness", TURNOVER: "turnover", HIGH_TURNOVER: "turnover",
  LOW_TURNOVER: "turnover", CONCENTRATED_WEIGHT: "max_weight", LOW_SUB_UNIVERSE_SHARPE: "sub_universe_sharpe",
  SELF_CORRELATION: "self_correlation", DRAWDOWN: "drawdown", LOW_2Y_SHARPE: "sharpe_2y", PROD_CORRELATION: "prod_correlation" };

class ResultView {
  constructor(container, res, { title, onReplace } = {}) {
    this.c = container;
    this.res = res;
    this.title = title;
    this.onReplace = onReplace;
    this.brain = null;
    this.jobId = null;
    S.resultViews.add(this);
    this.render();
    if (res.library_id) this.loadBrain();
  }

  destroy() { S.resultViews.delete(this); if (this.resize) this.resize.disconnect(); }

  async loadBrain() {
    try {
      const d = await api(`/api/library/${this.res.library_id}`);
      if (d.item.brain || d.item.brain_alpha_id) {
        this.brain = { ...(d.item.brain || {}), alpha_id: d.item.brain_alpha_id, url: d.brain_url, status: d.item.status };
        this.renderChecks();
        this.renderBrainBar();
      }
    } catch { /* not saved */ }
  }

  onJob(job) {
    if (job.library_id === this.res.library_id) this.loadBrain();
  }

  render() {
    const r = this.res;
    const sim = r.sim || {};
    this.checksEl = h("div");
    this.brainBar = h("div");
    this.panel = h("div");
    const warnings = [...(r.warnings || []), ...((sim.warnings) || [])];
    if (sim.proxied && sim.proxied.length) warnings.push(`Offline stand-ins used for ${sim.proxied.join(", ")} (no offline data): treat offline Sharpe as rough.`);
    const settings = r.settings || {};
    const setRows = [["Region", settings.region], ["Universe", settings.universe], ["Delay", settings.delay],
      ["Decay", settings.decay], ["Neutralization", settings.neutralization], ["Truncation (max stock weight)", settings.truncation],
      ["Pasteurization", settings.pasteurization], ["NaN handling", settings.nan_handling], ["Unit handling", settings.unit_handling]];

    this.c.replaceChildren(h("div", { class: "card" },
      h("div", { class: "summary-line" },
        h("h2", { text: this.title || r.thesis || "Result" }),
        r.family_label ? h("span", { class: "chip", text: r.family_label }) : null,
        r.library_id ? h("span", { class: "chip", text: `Library #${r.library_id}` }) : null),
      r.summary ? h("div", { class: "summary-line" }, badge(r.summary.passed ? "PASS" : "FAIL"),
        h("span", { class: "headline", text: r.summary.headline }), h("span", { class: "label-sim", text: "(simulated offline)" })) : null,
      isSummary(sim.metrics || {}),
      outlookBox(r.outlook),
      h("h3", { text: "Expression" }),
      h("div", { class: "expr mono" }, r.expression,
        h("button", { class: "btn small copy", text: "Copy", on: { click: () => copy(r.expression) } })),
      h("div", { class: "grid-2" },
        h("div", {},
          h("h3", { text: "Simulation settings" }),
          h("table", {}, h("tbody", {}, setRows.map(([k, v]) => h("tr", {}, h("td", { class: "muted", text: k }), h("td", { text: v ?? "–" }))))),
          h("button", { class: "btn small", style: { "margin-top": "8px" }, text: "Copy settings as JSON",
            on: { click: () => copy(JSON.stringify(settings, null, 2)) } })),
        r.rationale ? this.rationaleEl(r.rationale) : null),
      warnings.length ? h("div", { class: "warn-box" }, h("ul", { class: "fix-list" }, warnings.map((w) => h("li", { text: w })))) : null,
      h("h3", { text: "Submission checks" }),
      h("p", { class: "muted", text: "Each metric is shown with the threshold that governs it and where that threshold came from. “Simulated” = Alpha Bench's offline engine; “Verified on BRAIN” = your account's real results." }),
      this.checksEl,
      this.brainBar,
      h("div", { class: "row", style: { "margin-top": "12px" } },
        h("button", { class: "btn primary", text: "Why did this fail? / How to improve", on: { click: () => this.why() } }),
        h("button", { class: "btn", text: "Optimize settings", on: { click: () => this.optimizeForm() } }),
        h("button", { class: "btn", text: "Simulate on BRAIN", on: { click: () => this.live() } })),
      this.panel,
      h("h3", { text: "Backtest (simulated offline)" }),
      this.chartsEl = h("div", { class: "grid-charts" }),
      h("h3", { text: "Yearly" }),
      this.yearlyEl(sim.yearly || []),
      h("p", { class: "muted", text: "Offline figures come from a synthetic market: use them to rank and screen, then verify on BRAIN. See Settings & rules → Simulation fidelity." })));
    this.renderChecks();
    this.renderBrainBar();
    this.renderCharts();
    let lastW = this.chartsEl.clientWidth;
    this.resize = new ResizeObserver(() => {
      const w = this.chartsEl.clientWidth;
      if (Math.abs(w - lastW) < 30) return;
      lastW = w;
      clearTimeout(this._rt);
      this._rt = setTimeout(() => this.renderCharts(), 120);
    });
    this.resize.observe(this.chartsEl);
  }

  rationaleEl(rat) {
    if (typeof rat === "string") return h("div", { class: "rationale" }, h("h3", { text: "Rationale" }), h("p", { text: rat }));
    return h("div", { class: "rationale" }, h("h3", { text: "Rationale" }),
      h("dl", {},
        rat.why_it_works ? [h("dt", { text: "Why it should work" }), h("dd", { text: rat.why_it_works })] : null,
        rat.construction ? [h("dt", { text: "How it's built" }), h("dd", { text: rat.construction })] : null,
        rat.description ? [h("dt", { text: "Structure" }), h("dd", { text: rat.description })] : null,
        rat.inspiration ? [h("dt", { text: "Inspiration" }), h("dd", { text: rat.inspiration })] : null,
        rat.settings ? [h("dt", { text: "Why these settings" }), h("dd", {}, h("ul", { class: "fix-list" }, rat.settings.map((s) => h("li", { text: s }))))] : null,
        rat.repairs && rat.repairs.length ? [h("dt", { text: "Repairs applied" }), h("dd", {}, h("ul", { class: "fix-list" }, rat.repairs.map((s) => h("li", { text: s }))))] : null,
        rat.changes && rat.changes.length ? [h("dt", { text: "Changes applied" }), h("dd", {}, h("ul", { class: "fix-list" }, rat.changes.map((c) => h("li", { text: `${c.change}: ${c.detail}` }))))] : null));
  }

  brainFor(check) {
    if (!this.brain) return { value: null, result: null, limit: null };
    const metrics = this.brain.metrics || {};
    const checks = this.brain.checks || [];
    const metric = CHECK_TO_METRIC[check.name] || check.metric;
    let bc;
    if (check.name === "TURNOVER" || check.name === "HIGH_TURNOVER" || check.name === "LOW_TURNOVER") {
      const both = checks.filter((c) => c.name === "HIGH_TURNOVER" || c.name === "LOW_TURNOVER");
      bc = both.find((c) => String(c.result).toUpperCase() === "FAIL") || both[0];
    } else bc = checks.find((c) => c.name === check.name);
    const value = bc && bc.value != null ? bc.value : metrics[metric];
    return { value, result: bc ? bc.result : null, limit: bc ? bc.limit : null };
  }

  renderChecks() {
    const r = this.res;
    const checks = r.checks || [];
    const metrics = (r.sim && r.sim.metrics) || {};
    const hasBrain = !!(this.brain && (this.brain.metrics || this.brain.checks));
    const head = ["Check", "Simulated", "Threshold", "Result", ...(hasBrain ? ["Verified on BRAIN", "BRAIN result"] : [])];
    const rows = [];
    for (const c of checks) {
      const metric = CHECK_TO_METRIC[c.name] || c.metric;
      const b = this.brainFor(c);
      const detail = h("details", {}, h("summary", { class: "muted", text: "why" }),
        h("div", { class: "why", text: c.why }),
        c.fixes && c.fixes.length ? h("ul", { class: "fix-list" }, c.fixes.map((f) => h("li", { text: f }))) : null,
        h("div", { class: "src", text: `Threshold source: ${c.source}` }));
      rows.push(h("tr", {},
        h("td", {}, h("div", { text: c.label }), detail),
        h("td", { class: "num", text: c.value == null ? "–" : fmtMetric(metric, c.value) }),
        h("td", { class: "num", text: c.limit }),
        h("td", {}, badge(c.result)),
        hasBrain ? h("td", { class: "num", text: b.value == null ? "–" : fmtMetric(metric, b.value) }) : null,
        hasBrain ? h("td", {}, b.result ? badge(b.result) : h("span", { class: "muted", text: "–" }),
          b.limit != null ? h("div", { class: "src", text: `BRAIN limit ${fmtMetric(metric, b.limit)}` }) : null) : null));
    }
    if (hasBrain) {
      const known = new Set(checks.map((c) => c.name).concat(["HIGH_TURNOVER", "LOW_TURNOVER"]));
      for (const bc of (this.brain.checks || []).filter((c) => !known.has(c.name))) {
        rows.push(h("tr", {}, h("td", { text: bc.name }), h("td", { class: "num muted", text: "n/a" }),
          h("td", { class: "num", text: bc.limit ?? "–" }), h("td", { class: "muted", text: "BRAIN only" }),
          h("td", { class: "num", text: bc.value ?? "–" }), h("td", {}, badge(bc.result))));
      }
    }
    const stats = [["Returns", "returns"], ["Drawdown", "drawdown"], ["Margin", "margin"], ["Long count", "long_count"],
      ["Short count", "short_count"], ["Coverage (share of universe held)", "coverage"], ["PnL (IS)", "pnl_total"]];
    const bm = (this.brain && this.brain.metrics) || {};
    const cal = r.calibrated || {};
    const statRows = stats.map(([label, k]) => h("tr", {},
      h("td", { text: label }),
      h("td", { class: "num", text: k === "pnl_total" ? money(metrics[k]) : fmtMetric(k, metrics[k]) }),
      Object.keys(cal).length ? h("td", { class: "num", text: cal[k] ? fmtMetric(k, cal[k].value) : "–" }) : null,
      hasBrain ? h("td", { class: "num", text: bm[k] == null ? "–" : (k === "pnl_total" ? money(bm[k]) : fmtMetric(k, bm[k])) }) : null));
    const calNote = Object.keys(cal).length
      ? h("p", { class: "muted", text: `“Calibrated” shifts offline Sharpe/returns/turnover by the median gap seen in ${cal.sharpe ? cal.sharpe.n : "your"} past reconciliations with BRAIN.` }) : null;
    fill(this.checksEl,
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, head.map((t) => h("th", { text: t })))), h("tbody", {}, rows))),
      h("h3", { text: "Other statistics" }),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", { text: "Metric" }), h("th", { class: "num", text: "Simulated" }),
        Object.keys(cal).length ? h("th", { class: "num", text: "Calibrated estimate" }) : null,
        hasBrain ? h("th", { class: "num", text: "Verified on BRAIN" }) : null)), h("tbody", {}, statRows))),
      calNote,
      r.correlation && r.correlation.max != null ? h("p", { class: "muted",
        text: `Highest simulated correlation: ${num(r.correlation.max)} with ${r.correlation.with_name} (${r.correlation.compared} alphas compared).` }) : null);
  }

  renderBrainBar() {
    const b = this.brain;
    if (!b || !b.alpha_id) { this.brainBar.replaceChildren(); return; }
    const taskEl = h("div", { "data-task-kind": "brain-check" });
    this.brainBar.replaceChildren(h("div", { class: "ok-box" },
      h("div", {}, h("span", { class: "label-brain", text: "Verified on BRAIN: " }), `alpha ${b.alpha_id} · status ${b.status || ""}`),
      h("div", { class: "row", style: { "margin-top": "8px" } },
        h("button", { class: "btn", text: "Check submission eligibility on BRAIN (does not submit)",
          on: { click: () => startTask(`/api/brain/check/${encodeURIComponent(b.alpha_id)}`, "BRAIN submission check", () => this.loadBrain()) } }),
        h("a", { class: "btn", href: b.url, target: "_blank", rel: "noopener noreferrer", text: "Open on BRAIN to submit manually ↗" }),
        h("button", { class: "btn", text: "Reconcile with Alpha Bench's prediction", on: { click: () => gotoReconcile({ library_id: this.res.library_id, brain_alpha_id: b.alpha_id }) } })),
      taskEl,
      b.verified_correlation && b.verified_correlation.max != null ? h("div", { class: "why",
        text: `Verified self-correlation on daily PnL: ${num(b.verified_correlation.max)} with ${b.verified_correlation.with_name}.` }) : null));
  }

  renderCharts() {
    const s = (this.res.sim && this.res.sim.series) || {};
    const el = this.chartsEl;
    el.replaceChildren();
    if (!s.dates || !s.dates.length) return;
    const m = (this.res.sim && this.res.sim.metrics) || {};
    const rules = (S.meta && S.meta.rules) || {};
    const delay = (this.res.settings || {}).delay ?? 1;
    const sharpeLimit = rules.sharpe ? rules.sharpe[`delay${delay}`].value : 1.25;
    const toMax = rules.turnover ? rules.turnover.max.value : 0.7;
    const boxes = [h("div", {}), h("div", {}), h("div", {}), h("div", {}), h("div", {})];
    el.append(...boxes);
    lineChart(boxes[0], { title: "Cumulative PnL", subtitle: "$ on a $20M book, in-sample", dates: s.dates,
      series: [{ name: "Simulated PnL", values: s.cum_pnl, color: "var(--series-1)" }], fmt: money, area: false, zero: true });
    lineChart(boxes[1], { title: "Drawdown", subtitle: `% of half the book · worst ${pct(m.drawdown)}`, dates: s.dates,
      series: [{ name: "Drawdown", values: s.drawdown, color: "var(--series-1)" }], fmt: (v) => pct(v, 0), area: true });
    const to = rolling(s.turnover, 10);
    const refs = [{ y: 0.125, label: "12.5%" }];
    if (Math.max(...to.filter(isNum)) > toMax * 0.6) refs.push({ y: toMax, label: `max ${pct(toMax, 0)}` });
    lineChart(boxes[2], { title: "Turnover", subtitle: "rolling average; fitness gives no extra credit below 12.5%", dates: s.dates,
      series: [{ name: "Turnover", values: to, color: "var(--series-1)" }], fmt: (v) => pct(v, 0), refs, zero: true });
    lineChart(boxes[3], { title: "Rolling Sharpe", subtitle: "126-day window", dates: s.dates,
      series: [{ name: "Rolling Sharpe", values: s.rolling_sharpe, color: "var(--series-1)" }], fmt: (v) => num(v, 1),
      refs: [{ y: sharpeLimit, label: `floor ${sharpeLimit}` }], zero: true });
    const ex = (this.res.sim && this.res.sim.exposures) || {};
    const items = Object.entries(ex).map(([k, v]) => ({ label: k, value: v.net, extra: `gross ${pct(v.gross)}` }));
    if (items.length && Math.max(...items.map((d) => Math.abs(d.value))) < 5e-4) {
      fill(boxes[4], h("div", { class: "chart" }, h("h4", { text: "Sector exposure" }),
        h("div", { class: "chart-sub", text: "average net weight (share of book)" }),
        h("p", { class: "muted", text: `Net sector exposure is zero by construction: ${(this.res.settings || {}).neutralization} neutralization removes it. Gross weight per sector: ` +
          items.map((d) => `${d.label} ${d.extra.replace("gross ", "")}`).join(", ") + "." })));
    } else {
      divergingBars(boxes[4], { title: "Sector exposure", subtitle: "average net weight (share of book)", items, fmt: (v) => pct(v, 1) });
    }
  }

  yearlyEl(yearly) {
    if (!yearly.length) return h("p", { class: "muted", text: "–" });
    const cols = [["Year", "year", (v) => v], ["Sharpe", "sharpe", num], ["Fitness", "fitness", num], ["Returns", "returns", pct],
      ["Turnover", "turnover", pct], ["Drawdown", "drawdown", pct], ["Margin (bps)", "margin", (v) => num(v, 1)],
      ["Long", "long_count", (v) => num(v, 0)], ["Short", "short_count", (v) => num(v, 0)]];
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, cols.map(([t], i) => h("th", { class: i ? "num" : "", text: t })))),
      h("tbody", {}, yearly.map((y) => h("tr", {}, cols.map(([, k, f], i) => h("td", { class: i ? "num" : "", text: f(y[k]) })))))));
  }

  // --------------------------------------------------------- actions
  async why() {
    this.panel.replaceChildren(h("div", { class: "card" }, busy("Diagnosing and testing minimal fixes offline...")));
    try {
      const d = await api("/api/why", { method: "POST", body: { expression: this.res.expression, settings: this.res.settings } });
      const failing = d.failing.length
        ? d.failing.map((c) => h("div", { class: c.result === "FAIL" ? "err-box" : "warn-box" },
          h("div", {}, badge(c.result), " ", h("strong", { text: c.label })), h("div", { class: "why", text: c.why })))
        : [h("div", { class: "ok-box", text: d.fixes.length
          ? "Nothing fails offline. The changes below are optional improvements."
          : "Nothing fails offline, so there's nothing to repair. Try Optimize settings for a better Sharpe/fitness trade-off, then verify on BRAIN." })];
      fill(this.panel, h("div", { class: "card" },
        h("h3", { text: "Why it fails, and the smallest changes that fix it" }),
        failing,
        this.fixTable(d.fixes)));
    } catch (e) { this.panel.replaceChildren(errorBox(e)); }
  }

  fixTable(fixes) {
    if (!fixes.length) return null;
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Fix", "Sharpe", "Fitness", "Turnover", "Clears", ""].map((t, i) => h("th", { class: i && i < 4 ? "num" : "", text: t })))),
      h("tbody", {}, fixes.map((f) => h("tr", {},
        h("td", {}, h("div", { text: f.title }), h("div", { class: "why", text: `${f.change} · ${f.detail}` }),
          f.introduced && f.introduced.length ? h("div", { class: "src", text: `Introduces: ${f.introduced.join(", ")}` }) : null),
        h("td", { class: "num", text: `${num(f.metrics && f.metrics.sharpe)} (${signed(f.delta && f.delta.sharpe)})` }),
        h("td", { class: "num", text: `${num(f.metrics && f.metrics.fitness)} (${signed(f.delta && f.delta.fitness)})` }),
        h("td", { class: "num", text: pct(f.metrics && f.metrics.turnover) }),
        h("td", {}, f.passes_all ? badge("PASS") : null, h("div", { class: "src", text: (f.cleared || []).join(", ") || "–" })),
        h("td", {}, h("button", { class: "btn small", text: "Apply", on: { click: () => this.apply(f.expression, f.settings, "fix", f.change) } })))))));
  }

  async apply(expression, settings, mode = "fix", note) {
    this.panel.replaceChildren(h("div", { class: "card" }, busy("Applying and re-simulating...")));
    try {
      const sim = await api("/api/simulate", { method: "POST", body: { expression, settings } });
      const saved = await api("/api/library", { method: "POST", body: { expression, settings, mode, parent_id: this.res.library_id,
        rationale: note ? { changes: [{ change: note, detail: "applied from the fix list" }] } : null,
        offline: { metrics: sim.sim.metrics, summary: sim.summary } } });
      const next = { ...this.res, ...sim, expression, settings, library_id: saved.library_id,
        rationale: note ? { ...(typeof this.res.rationale === "object" ? this.res.rationale : {}), repairs: [...((this.res.rationale && this.res.rationale.repairs) || []), `${note} (applied)`] } : this.res.rationale };
      this.destroy();
      if (this.onReplace) this.onReplace(next);
      else new ResultView(this.c, next, { title: this.title });
      toast(`Saved as library #${saved.library_id} (child of #${this.res.library_id ?? "–"}).`);
    } catch (e) { this.panel.replaceChildren(errorBox(e)); }
  }

  optimizeForm() {
    const list = (v) => v.split(",").map((x) => x.trim()).filter(Boolean);
    const decay = h("input", { type: "text", value: "0, 2, 4, 6, 8, 12" });
    const trunc = h("input", { type: "text", value: "0.01, 0.05, 0.08, 0.1" });
    const neuts = S.meta.neutralizations.filter((n) => n !== "NONE").map((n) => h("input", { type: "checkbox", value: n, checked: true }));
    const unis = S.meta.universes.map((u) => h("input", { type: "checkbox", value: u, checked: u === this.res.settings.universe }));
    const delays = [1, 0].map((d) => h("input", { type: "checkbox", value: String(d), checked: d === this.res.settings.delay }));
    const out = h("div");
    const go = async () => {
      out.replaceChildren(busy("Sweeping settings (the expression is evaluated once per universe/delay)..."));
      try {
        const grid = { decay: list(decay.value).map(Number), truncation: list(trunc.value).map(Number),
          neutralization: neuts.filter((c) => c.checked).map((c) => c.value),
          universe: unis.filter((c) => c.checked).map((c) => c.value), delay: delays.filter((c) => c.checked).map((c) => +c.value) };
        const d = await api("/api/optimize", { method: "POST", body: { expression: this.res.expression, settings: this.res.settings, grid } });
        out.replaceChildren(
          h("p", { class: "muted", text: `${d.runs} combinations in ${d.elapsed_s}s · ${d.passing} pass every offline check · ranked by overall score (failures first, then Sharpe and fitness margins). ◆ = on the Sharpe/fitness Pareto front.` }),
          h("div", { class: "table-wrap" }, h("table", {},
            h("thead", {}, h("tr", {}, ["", "Universe", "Delay", "Decay", "Neutralization", "Trunc.", "Sharpe", "Fitness", "Turnover", "Returns", "Fails", ""].map((t, i) => h("th", { class: i >= 6 && i <= 9 ? "num" : "", text: t })))),
            h("tbody", {}, d.rows.slice(0, 25).map((r) => h("tr", {},
              h("td", { text: r.pareto ? "◆" : "" }), h("td", { text: r.settings.universe }), h("td", { text: r.settings.delay }),
              h("td", { text: r.settings.decay }), h("td", { text: r.settings.neutralization }), h("td", { text: r.settings.truncation }),
              h("td", { class: "num", text: num(r.sharpe) }), h("td", { class: "num", text: num(r.fitness) }),
              h("td", { class: "num", text: pct(r.turnover) }), h("td", { class: "num", text: pct(r.returns) }),
              h("td", {}, r.fails.length ? h("span", { class: "src", text: r.fails.join(", ") }) : badge("PASS")),
              h("td", {}, h("button", { class: "btn small", text: "Use", on: { click: () => this.apply(this.res.expression, r.settings, "sweep", `settings → ${r.settings.neutralization}, decay ${r.settings.decay}, truncation ${r.settings.truncation}, ${r.settings.universe}, delay ${r.settings.delay}`) } }))))))));
      } catch (e) { out.replaceChildren(errorBox(e)); }
    };
    this.panel.replaceChildren(h("div", { class: "card" },
      h("h3", { text: "Settings sweep" }),
      h("div", { class: "row" },
        h("label", { class: "field" }, "Decay values", decay),
        h("label", { class: "field" }, "Truncation values", trunc)),
      h("div", { class: "row", style: { "margin-top": "10px" } },
        h("div", {}, h("div", { class: "muted", text: "Neutralization" }), h("div", { class: "checks-inline" }, neuts.map((c) => h("label", {}, c, c.value)))),
        h("div", {}, h("div", { class: "muted", text: "Universe" }), h("div", { class: "checks-inline" }, unis.map((c) => h("label", {}, c, c.value)))),
        h("div", {}, h("div", { class: "muted", text: "Delay" }), h("div", { class: "checks-inline" }, delays.map((c) => h("label", {}, c, c.value))))),
      h("div", { class: "row", style: { "margin-top": "10px" } }, h("button", { class: "btn primary", text: "Run sweep", on: { click: go } })),
      out));
  }

  async live() {
    if (!S.state || S.state.auth.state !== "signed_in") { openAuth(); return; }
    try {
      let lid = this.res.library_id;
      const d = await api("/api/brain/simulate", { method: "POST", body: lid ? { library_id: lid } : { expression: this.res.expression, settings: this.res.settings } });
      this.res.library_id = d.library_id;
      this.jobId = d.job.id;
      toast("Queued on BRAIN. You'll see the verified results here when it finishes.");
      this.brainBar.replaceChildren(h("div", { class: "warn-box" }, busy("Simulating on BRAIN... (see the Jobs button for progress)")));
      poll();
    } catch (e) { toast(e.message); }
  }
}

/** How an offline pass is likely to land on BRAIN. Never trust a bare offline pass. */
function outlookBox(o) {
  if (!o || o.level === "fail") return null;
  const kind = { comfortable: ["ok-box", "BRAIN outlook: good — clears the floors with margin"],
    borderline: ["warn-box", "BRAIN outlook: borderline — a live run may come back “Needs Improvement”"],
    "likely-fail": ["err-box", "BRAIN outlook: likely “Needs Improvement” — calibration puts it below a floor"] }[o.level];
  if (!kind) return null;
  const tail = o.calibration_n ? ` Based on ${o.calibration_n} reconciled offline-vs-BRAIN results.`
    : " No calibration data yet, so this uses a safety margin; reconcile live results to sharpen it.";
  return h("div", { class: kind[0] },
    h("div", {}, h("strong", { text: kind[1] }), h("span", { class: "muted", text: tail })),
    (o.reasons || []).length ? h("ul", { class: "fix-list" }, o.reasons.map((t) => h("li", { text: t }))) : null);
}

/** IS summary in BRAIN's own order, so it reads like the platform's results panel. */
function isSummary(m) {
  const cells = [["Sharpe", num(m.sharpe)], ["Turnover", pct(m.turnover, 2)], ["Fitness", num(m.fitness)],
    ["Returns", pct(m.returns, 2)], ["Drawdown", pct(m.drawdown, 2)], ["Margin", isNum(m.margin) ? `${num(m.margin)}‱` : "–"],
    ["Returns / drawdown", isNum(m.returns_drawdown) ? num(m.returns_drawdown) : "–"]];
  const chips = [];
  if (m.turnover_bucket) chips.push(`${m.turnover_bucket} turnover`);
  if (isNum(m.train_sharpe)) chips.push(`train Sharpe ${num(m.train_sharpe)} · last ${m.test_months} months (test) ${num(m.test_sharpe)}`);
  return h("div", { class: "is-summary" },
    h("div", { class: "is-cells" }, cells.map(([k, v]) => h("div", { class: "is-cell" }, h("div", { class: "is-k", text: k }), h("div", { class: "is-v", text: v })))),
    chips.length ? h("div", { class: "row" }, chips.map((c) => h("span", { class: "chip", text: c }))) : null);
}

function rolling(arr, w) {
  const out = [];
  let sum = 0, cnt = 0;
  const q = [];
  for (const v of arr || []) {
    q.push(v);
    if (isNum(v)) { sum += v; cnt++; }
    if (q.length > w) { const o = q.shift(); if (isNum(o)) { sum -= o; cnt--; } }
    out.push(cnt ? sum / cnt : null);
  }
  return out;
}

function mountResult(host, res, title) {
  for (const v of [...S.resultViews]) if (v.c === host) v.destroy();
  return new ResultView(host, res, { title, onReplace: (next) => mountResult(host, next, title) });
}

// =====================================================================
// Create
// =====================================================================
function renderCreate() {
  const m = S.meta;
  const panel = $("#tab-create");
  const fams = Object.entries(m.families).map(([k, label]) => h("input", { type: "checkbox", value: k, checked: true, "data-label": label }));
  const form = settingsForm({}, { compact: true });
  const auto = h("input", { type: "checkbox" });
  const byCat = {};
  for (const op of m.operators.filter((o) => !o.platform_only)) (byCat[op.category] ||= []).push(op);
  const opBoxes = [];
  const opsEl = h("div", {}, Object.entries(byCat).map(([cat, ops]) => h("div", { style: { "margin-top": "8px" } },
    h("div", { class: "muted", text: cat }),
    h("div", { class: "checks-inline" }, ops.map((o) => {
      const cb = h("input", { type: "checkbox", value: o.name, checked: o.available, disabled: !o.available });
      opBoxes.push(cb);
      return h("label", { title: `${o.signature} · ${o.description}${o.available ? "" : " (not available on your account)"}` }, cb, o.name);
    })))));
  const out = h("div");
  const hist = h("div");
  const btn = h("button", { class: "btn primary", text: "Create alpha" });
  btn.addEventListener("click", async () => {
    const families = fams.filter((c) => c.checked).map((c) => c.value);
    if (!families.length) { toast("Pick at least one idea family."); return; }
    const allChecked = opBoxes.every((c) => c.checked || c.disabled);
    btn.disabled = true;
    out.replaceChildren(h("div", { class: "card" }, busy("Reasoning about a fresh idea, building candidates, screening them offline and repairing the best...")));
    try {
      const res = await api("/api/create", { method: "POST", body: { families, settings: form.get(), auto_universe: auto.checked,
        allowed_ops: allChecked ? null : opBoxes.filter((c) => c.checked).map((c) => c.value) } });
      mountResult(out, res, res.thesis);
      S.history.unshift(res);
      renderHistory(hist, out);
    } catch (e) { out.replaceChildren(errorBox(e)); }
    btn.disabled = false;
  });
  panel.replaceChildren(
    h("div", { class: "card" },
      h("h2", { text: "Create a new alpha" }),
      h("p", { class: "sub", text: "Every click reasons from scratch: a different economic thesis, different fields and a different structure. Ideas already generated this session (or saved in your library) are never re-derived." }),
      h("h3", { text: "1 · Idea space" }),
      h("div", { class: "checks-inline" }, fams.map((c) => h("label", {}, c, c.dataset.label))),
      h("h3", { text: "2 · Where it will trade" }),
      form.el,
      h("label", { class: "checks-inline", style: { "margin-top": "8px" } }, auto, "Let Alpha Bench choose the universe"),
      h("details", { class: "adv" }, h("summary", { text: "Operator-level control" }),
        h("p", { class: "muted", text: "Untick operators you don't want in generated alphas. Greyed-out ones aren't available on your BRAIN account (after syncing)." }),
        opsEl),
      h("div", { class: "row", style: { "margin-top": "16px" } }, btn,
        h("span", { class: "muted", text: "Takes about 5-20 seconds: candidates are simulated, screened for correlation with your alphas, and repaired." }))),
    out,
    hist);
}

function renderHistory(el, out) {
  if (!S.history.length) { el.replaceChildren(); return; }
  el.replaceChildren(h("div", { class: "card" }, h("h3", { text: "Created this session" }),
    h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, S.history.map((r) => h("tr", { class: "clickable", on: { click: () => mountResult(out, r, r.thesis) } },
      h("td", { text: r.thesis }), h("td", { class: "mono", text: r.expression.slice(0, 80) }),
      h("td", {}, badge(r.summary.passed ? "PASS" : "FAIL")), h("td", { class: "num", text: `S ${num(r.sim.metrics.sharpe)} · F ${num(r.sim.metrics.fitness)}` }))))))));
}

// =====================================================================
// Recreate / Modify
// =====================================================================
function renderRecreate(prefill) {
  const panel = $("#tab-recreate");
  const expr = h("textarea", { class: "mono", rows: 4, spellcheck: "false", placeholder: "Paste an alpha expression, e.g.  rank(-ts_delta(close, 5))", value: prefill ? prefill.expression : "" });
  const form = settingsForm(prefill ? prefill.settings : {});
  const modeMod = h("input", { type: "radio", name: "rmode", value: "modify", checked: true });
  const modeReb = h("input", { type: "radio", name: "rmode", value: "rebuild" });
  const out = h("div");
  const res = h("div");
  const btn = h("button", { class: "btn primary", text: "Analyse and improve" });
  btn.addEventListener("click", async () => {
    if (!expr.value.trim()) { toast("Paste an expression first."); return; }
    btn.disabled = true;
    out.replaceChildren(h("div", { class: "card" }, busy(modeReb.checked ? "Rebuilding the idea as a new alpha..." : "Diagnosing and patching...")));
    res.replaceChildren();
    try {
      const d = await api("/api/recreate", { method: "POST", body: { expression: expr.value, settings: form.get(), mode: modeReb.checked ? "rebuild" : "modify" } });
      out.replaceChildren(recreateSummary(d));
      const result = { ...d.result, library_id: d.library_id,
        rationale: { description: d.description, inspiration: d.inspiration, changes: d.changes } };
      mountResult(res, result, d.mode === "rebuild" ? "Rebuilt alpha" : (d.unchanged ? "Already passes: no changes needed" : "Modified alpha"));
    } catch (e) { out.replaceChildren(errorBox(e)); }
    btn.disabled = false;
  });
  const gallery = h("div");
  courseGallery(gallery, (ex, mode) => {
    expr.value = ex.expression;
    const f2 = settingsForm(ex.settings);
    form.el.replaceWith(f2.el);
    form.el = f2.el; form.get = f2.get;
    (mode === "rebuild" ? modeReb : modeMod).checked = true;
    btn.click();
    btn.scrollIntoView({ behavior: "smooth", block: "center" });
  });
  panel.replaceChildren(
    gallery,
    h("div", { class: "card" },
      h("h2", { text: "Recreate or modify an existing alpha" }),
      h("p", { class: "sub", text: "Alpha Bench parses the expression, classifies its structure, simulates it, and then either patches it with the smallest changes that fix its failures, or rebuilds the idea as a genuinely new, decorrelated alpha." }),
      h("label", { class: "field" }, "Alpha expression", expr),
      h("h3", { text: "Settings it was simulated with" }),
      form.el,
      h("h3", { text: "What should Alpha Bench do?" }),
      h("div", { class: "checks-inline" },
        h("label", {}, modeMod, "Modify it: keep the alpha, apply minimal fixes"),
        h("label", {}, modeReb, "Rebuild it: new alpha inspired by the same idea")),
      h("div", { class: "row", style: { "margin-top": "14px" } }, btn)),
    out, res);
}

async function courseGallery(host, onPick) {
  let d;
  try { d = await api("/api/course"); } catch (e) { host.replaceChildren(errorBox(e)); return; }
  const rows = d.examples.map((ex) => {
    const b = ex.brain || {};
    const res = isNum(b.sharpe) ? `Sharpe ${num(b.sharpe)} · turnover ${pct(b.turnover)} · fitness ${isNum(b.fitness) ? num(b.fitness) : "–"} · returns ${pct(b.returns)}` : "not backtested in the lesson";
    const failing = ex.verdict.startsWith("Fails");
    return h("tr", {},
      h("td", {}, h("div", { class: "src", text: ex.lesson }), h("strong", { text: ex.title }), h("div", { class: "why", text: ex.idea })),
      h("td", { class: "mono", text: ex.expression }),
      h("td", {}, h("div", { text: res }), h("div", {}, ex.verdict.startsWith("Passes") ? badge("PASS") : failing ? badge("FAIL") : badge("PENDING"), " ", h("span", { class: "why", text: ex.verdict }))),
      h("td", {}, h("div", { class: "row" },
        h("button", { class: "btn small", text: "Modify", on: { click: () => onPick(ex, "modify") } }),
        h("button", { class: "btn small", text: "Rebuild", on: { click: () => onPick(ex, "rebuild") } }),
        isNum(b.sharpe) ? h("button", { class: "btn small", text: "Reconcile", on: { click: () => gotoReconcile({ expression: ex.expression, settings: ex.settings, pasted_text: ex.pasted_text }) } }) : null,
        h("a", { class: "btn small", href: ex.video, target: "_blank", rel: "noopener noreferrer", text: "Lesson ↗" }))));
  });
  host.replaceChildren(h("details", { class: "card", open: false },
    h("summary", {}, h("strong", { text: "Learn2Quant examples" }), h("span", { class: "muted", text: ` · the alphas WorldQuant builds in its lessons, with the BRAIN results shown on screen` })),
    h("p", { class: "sub", style: { "margin-top": "10px" }, text: "Load one to see why BRAIN passed or failed it and how Alpha Bench would fix it. Several lesson alphas fail on fitness; that's what makes them good practice." }),
    h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Idea", "Expression", "BRAIN result (from the lesson)", ""].map((t) => h("th", { text: t })))), h("tbody", {}, rows)))));
}

function recreateSummary(d) {
  const st = d.structure;
  const orig = d.original;
  const nr = d.result;
  const cmp = [["Sharpe", "sharpe", num], ["Fitness", "fitness", num], ["Turnover", "turnover", pct], ["Returns", "returns", pct], ["Max weight", "max_weight", pct]];
  return h("div", { class: "card" },
    h("h3", { text: "What this alpha is" }),
    h("p", { text: d.description }),
    h("div", { class: "row" },
      h("span", { class: "chip", text: `family: ${st.family.replace("_", " ")}` }),
      h("span", { class: "chip", text: `thesis: ${st.thesis}` }),
      h("span", { class: "chip", text: `horizon: ${st.horizon}` }),
      h("span", { class: "chip", text: `fields: ${st.fields.join(", ") || "–"}` })),
    st.notes.length ? h("ul", { class: "fix-list" }, st.notes.map((n) => h("li", { text: n }))) : null,
    h("h3", { text: "Before and after (simulated)" }),
    h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "Metric" }), h("th", { class: "num", text: "Original" }), h("th", { class: "num", text: d.mode === "rebuild" ? "Rebuilt" : "Modified" }))),
      h("tbody", {}, cmp.map(([l, k, f]) => h("tr", {}, h("td", { text: l }), h("td", { class: "num", text: f(orig.sim.metrics[k]) }), h("td", { class: "num", text: f(nr.sim.metrics[k]) }))),
        h("tr", {}, h("td", { text: "Checks" }), h("td", {}, badge(orig.summary.passed ? "PASS" : "FAIL"), h("div", { class: "src", text: orig.summary.failures.join(", ") })),
          h("td", {}, badge(nr.summary.passed ? "PASS" : "FAIL"), h("div", { class: "src", text: nr.summary.failures.join(", ") })))))),
    d.mode === "rebuild" ? h("p", { class: d.decorrelated ? "ok-box" : "warn-box",
      text: `PnL correlation with the original: ${num(d.correlation_with_original)} (${d.decorrelated ? "below 0.7, so it won't fail self-correlation against the original" : "still high: it may fail self-correlation against the original"}).` }) : null,
    d.mode === "modify" && d.options && d.options.length ? [h("h3", { text: "Other minimal fixes you could apply instead" }),
      h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, d.options.map((o) => h("tr", {},
        h("td", {}, h("div", { text: o.title }), h("div", { class: "why", text: `${o.change} · ${o.detail}` })),
        h("td", { class: "num", text: `S ${num(o.metrics && o.metrics.sharpe)} · F ${num(o.metrics && o.metrics.fitness)} · TO ${pct(o.metrics && o.metrics.turnover)}` }),
        h("td", {}, o.passes_all ? badge("PASS") : h("span", { class: "src", text: `clears ${(o.cleared || []).join(", ") || "nothing"}` })),
        h("td", {}, h("button", { class: "btn small", text: "Copy", on: { click: () => copy(o.expression) } }))))))) ] : null,
    d.mode === "rebuild" && d.alternatives && d.alternatives.length ? [h("h3", { text: "Other variants considered" }),
      h("ul", { class: "fix-list" }, d.alternatives.map((a) => h("li", {}, h("span", { class: "mono", text: a.expression }), ` · ${a.why} · corr ${num(a.corr_with_original)}`)))] : null);
}

// =====================================================================
// Reconcile
// =====================================================================
let reconcilePrefill = null;
function gotoReconcile(pre) {
  reconcilePrefill = pre;
  showTab("reconcile");
  renderReconcile();
}

async function renderReconcile() {
  const panel = $("#tab-reconcile");
  const pre = reconcilePrefill || {};
  reconcilePrefill = null;
  let items = [];
  try { items = (await api("/api/library")).items; } catch { /* ignore */ }
  const libSel = h("select", {}, h("option", { value: "", text: "— none —" }),
    items.map((i) => h("option", { value: i.id, selected: pre.library_id === i.id, text: `#${i.id} ${i.thesis || i.mode} · ${i.expression.slice(0, 50)}` })));
  const idIn = h("input", { type: "text", placeholder: "e.g. aBc1234", value: pre.brain_alpha_id || "" });
  const expr = h("textarea", { class: "mono", rows: 3, spellcheck: "false", value: pre.expression || "", placeholder: "Only needed if the alpha isn't in the library and you're not pulling it by ID" });
  const form = settingsForm(pre.settings || {});
  const pasted = h("textarea", { rows: 5, value: pre.pasted_text || "", placeholder: "Paste BRAIN's results here, e.g. the IS summary row (Sharpe  Turnover  Fitness  Returns  Drawdown  Margin) and any failed checks." });
  const file = h("input", { type: "file", accept: "image/*" });
  const ocrMsg = h("div");
  file.addEventListener("change", async () => {
    const f = file.files[0];
    if (!f) return;
    ocrMsg.replaceChildren(busy("Reading the screenshot locally (Tesseract OCR)..."));
    const b64 = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f); });
    try {
      const d = await api("/api/reconcile/ocr", { method: "POST", body: { image_b64: b64 } });
      pasted.value = (pasted.value ? pasted.value + "\n" : "") + (d.text || "");
      ocrMsg.replaceChildren(h("div", { class: "ok-box", text: `Read ${Object.keys(d.metrics).length} metrics from the screenshot. Check them below before reconciling.` }));
    } catch (e) { ocrMsg.replaceChildren(errorBox(e)); }
  });
  const out = h("div");
  const btn = h("button", { class: "btn primary", text: "Reconcile" });
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    out.replaceChildren(h("div", { class: "card" }, busy("Comparing Alpha Bench's prediction with BRAIN and testing fixes...")));
    const body = { library_id: libSel.value ? +libSel.value : null, brain_alpha_id: idIn.value.trim() || null,
      expression: expr.value.trim() || null, settings: expr.value.trim() ? form.get() : {}, pasted_text: pasted.value.trim() || null };
    try { out.replaceChildren(reconcileView(await api("/api/reconcile", { method: "POST", body }))); }
    catch (e) { out.replaceChildren(errorBox(e)); }
    btn.disabled = false;
  });
  const signedIn = S.state && S.state.auth.state === "signed_in";
  panel.replaceChildren(
    h("div", { class: "card" },
      h("h2", { text: "Reconcile with real BRAIN results" }),
      h("p", { class: "sub", text: "Tell Alpha Bench which alpha, give it BRAIN's actual results, and it will diagnose the gap, pin each failure to a specific metric, and propose minimal fixes calibrated to BRAIN's numbers. Every comparison is logged to recalibrate the offline engine." }),
      h("h3", { text: "1 · Which alpha" }),
      h("div", { class: "row" },
        h("label", { class: "field" }, "From your library", libSel),
        h("label", { class: "field" }, `BRAIN alpha ID${signedIn ? " (results are pulled automatically)" : " (sign in to pull results)"}`, idIn)),
      h("details", { class: "adv", open: !!pre.expression }, h("summary", { text: "Or enter the expression and settings" }), h("label", { class: "field" }, "Expression", expr), form.el),
      h("h3", { text: "2 · BRAIN's results (if not pulled by ID)" }),
      h("label", { class: "field" }, "Paste metrics", pasted),
      h("div", { class: "row", style: { "margin-top": "8px" } }, h("label", { class: "field" }, "…or upload a screenshot (read locally, never uploaded)", file)),
      ocrMsg,
      h("div", { class: "row", style: { "margin-top": "14px" } }, btn)),
    out);
}

function reconcileView(d) {
  const rows = d.rows.map((r) => h("tr", {},
    h("td", { text: r.label }),
    h("td", { class: "num", text: r.percent ? pct(r.offline) : num(r.offline) }),
    h("td", { class: "num", text: r.brain == null ? "–" : (r.percent ? pct(r.brain) : num(r.brain)) }),
    h("td", { class: "num", text: r.delta == null ? "–" : (r.percent ? signed(r.delta, (v) => pct(v)) : signed(r.delta)) })));
  const fixRows = d.fixes.map((f) => h("tr", {},
    h("td", {}, h("div", { text: f.title }), h("div", { class: "why", text: `${f.change} · ${f.detail}` })),
    h("td", { class: "num", text: num(f.predicted_brain.sharpe) }), h("td", { class: "num", text: num(f.predicted_brain.fitness) }),
    h("td", { class: "num", text: pct(f.predicted_brain.turnover) }),
    h("td", {}, f.predicted_clears.length ? f.predicted_clears.map((c) => h("div", {}, badge("PASS"), " ", c)) : h("span", { class: "muted", text: "–" })),
    h("td", {}, h("button", { class: "btn small", text: "Open", on: { click: async () => {
      const host = $("#reconcile-apply");
      host.replaceChildren(h("div", { class: "card" }, busy("Simulating the fix...")));
      try {
        const sim = await api("/api/simulate", { method: "POST", body: { expression: f.expression, settings: f.settings } });
        const saved = await api("/api/library", { method: "POST", body: { expression: f.expression, settings: f.settings, mode: "fix",
          parent_id: d.library_id, rationale: { changes: [{ change: f.change, detail: `reconcile fix for ${d.failing.join(", ")}` }] },
          offline: { metrics: sim.sim.metrics, summary: sim.summary } } });
        mountResult(host, { ...sim, expression: f.expression, settings: f.settings, library_id: saved.library_id,
          rationale: { changes: [{ change: f.change, detail: f.detail }] } }, `Repaired: ${f.title}`);
        host.scrollIntoView({ behavior: "smooth" });
      } catch (e) { host.replaceChildren(errorBox(e)); }
    } } }))));
  return h("div", {},
    h("div", { class: "card" },
      h("h2", { text: "Diagnosis" }),
      h("p", { class: "muted", text: d.source === "brain" ? "BRAIN results pulled from your account (verified on BRAIN)." : "BRAIN results from pasted text/screenshot." }),
      d.diagnosis.length ? d.diagnosis.map((t) => h("div", { class: "err-box", text: t })) : h("div", { class: "ok-box", text: "BRAIN reports no failing checks." }),
      d.learned_limits.length ? h("p", { class: "muted", text: `Learned BRAIN's thresholds: ${d.learned_limits.join("; ")}. Offline checks now use them.` }) : null,
      h("h3", { text: "Simulated vs verified" }),
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", { text: "Metric" }), h("th", { class: "num", text: "Alpha Bench (simulated)" }), h("th", { class: "num", text: "BRAIN (verified)" }), h("th", { class: "num", text: "Gap" }))),
        h("tbody", {}, rows))),
      h("h3", { text: "Why the numbers differ" }),
      h("ul", { class: "fix-list" }, d.gap_reasons.map((g) => h("li", { text: g })))),
    h("div", { class: "card" },
      h("h2", { text: "Minimal fixes, predicted on BRAIN's numbers" }),
      h("p", { class: "muted", text: "Each fix's effect is measured offline and applied to BRAIN's actual baseline (a shift for Sharpe/fitness/returns, a ratio for turnover). Verify the chosen fix with a live simulation." }),
      d.fixes.length ? h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["Fix", "Sharpe", "Fitness", "Turnover", "Predicted to clear", ""].map((t, i) => h("th", { class: i && i < 4 ? "num" : "", text: t })))),
        h("tbody", {}, fixRows))) : h("p", { class: "muted", text: "No fixes needed or none found." }),
      d.calibration && Object.keys(d.calibration).length ? h("p", { class: "muted",
        text: "Calibration log: " + Object.entries(d.calibration).map(([k, v]) => `${k} median gap ${signed(v.median_delta)} over ${v.n}`).join(" · ") }) : null),
    h("div", { id: "reconcile-apply" }));
}

// =====================================================================
// Library
// =====================================================================
async function refreshLibrary() {
  const panel = $("#tab-library");
  const search = h("input", { type: "text", placeholder: "Search expressions, theses, BRAIN IDs" });
  const list = h("div");
  const detail = h("div");
  const load = async () => {
    try {
      const d = await api("/api/library" + (search.value ? `?search=${encodeURIComponent(search.value)}` : ""));
      list.replaceChildren(d.items.length ? h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["#", "Created", "Mode", "Idea", "Expression", "Simulated", "BRAIN"].map((t) => h("th", { text: t })))),
        h("tbody", {}, d.items.map((i) => {
          const om = (i.offline && i.offline.metrics) || {};
          const bm = (i.brain && i.brain.metrics) || {};
          return h("tr", { class: "clickable", on: { click: () => showLibraryItem(i.id, detail) } },
            h("td", { text: i.id }), h("td", { text: new Date(i.created_at * 1000).toLocaleString() }),
            h("td", {}, h("span", { class: "chip", text: i.mode })), h("td", { text: i.name || i.thesis || "–" }),
            h("td", { class: "mono", text: i.expression.slice(0, 60) + (i.expression.length > 60 ? "…" : "") }),
            h("td", { class: "num", text: om.sharpe != null ? `S ${num(om.sharpe)} · F ${num(om.fitness)}` : "–" }),
            h("td", {}, i.brain_alpha_id ? h("span", { text: `${i.brain_alpha_id} · S ${num(bm.sharpe)}` }) : h("span", { class: "muted", text: i.status })));
        })))) : h("p", { class: "muted", text: "Nothing saved yet. Everything you create, modify, rebuild or reconcile lands here automatically." }));
    } catch (e) { list.replaceChildren(errorBox(e)); }
  };
  search.addEventListener("input", () => { clearTimeout(search._t); search._t = setTimeout(load, 250); });
  panel.replaceChildren(h("div", { class: "card" }, h("h2", { text: "Alpha library" }),
    h("p", { class: "sub", text: "Every alpha with its settings, simulated and verified metrics, and lineage (what it was derived from)." }),
    search, h("div", { style: { "margin-top": "12px" } }, list)), detail);
  load();
}

async function showLibraryItem(id, host) {
  host.replaceChildren(h("div", { class: "card" }, busy("Loading...")));
  try {
    const d = await api(`/api/library/${id}`);
    const it = d.item;
    const name = h("input", { type: "text", value: it.name || "", placeholder: "Name this alpha" });
    const lin = d.lineage;
    const resultHost = h("div");
    host.replaceChildren(h("div", { class: "card" },
      h("h2", { text: `#${it.id} ${it.name || it.thesis || ""}` }),
      h("div", { class: "expr mono" }, it.expression, h("button", { class: "btn small copy", text: "Copy", on: { click: () => copy(it.expression) } })),
      h("div", { class: "row", style: { "margin-top": "10px" } },
        name, h("button", { class: "btn", text: "Rename", on: { click: async () => { await api(`/api/library/${id}`, { method: "PATCH", body: { name: name.value } }); toast("Renamed."); } } }),
        h("button", { class: "btn primary", text: "Open (re-simulate offline)", on: { click: async () => {
          resultHost.replaceChildren(h("div", { class: "card" }, busy("Simulating...")));
          try {
            const sim = await api("/api/simulate", { method: "POST", body: { expression: it.expression, settings: it.settings } });
            mountResult(resultHost, { ...sim, expression: it.expression, settings: it.settings, library_id: it.id, rationale: it.rationale,
              thesis: it.thesis }, it.name || it.thesis || `Alpha #${it.id}`);
          } catch (e) { resultHost.replaceChildren(errorBox(e)); }
        } } }),
        h("button", { class: "btn", text: "Reconcile", on: { click: () => gotoReconcile({ library_id: it.id, brain_alpha_id: it.brain_alpha_id }) } }),
        h("button", { class: "btn", text: "Export JSON", on: { click: async () => download(`alpha-${it.id}.json`, await api(`/api/library/${id}/export`)) } }),
        h("button", { class: "btn danger", text: "Delete", on: { click: async () => {
          if (!confirm(`Delete library item #${id}? Its children are kept but lose this parent.`)) return;
          await api(`/api/library/${id}`, { method: "DELETE" }); host.replaceChildren(); refreshLibrary();
        } } })),
      h("h3", { text: "Lineage" }),
      lin.ancestors.length || lin.descendants.length ? h("ul", { class: "fix-list" },
        lin.ancestors.map((a) => h("li", {}, h("a", { href: "#", text: `#${a.id} (${a.mode})`, on: { click: (e) => { e.preventDefault(); showLibraryItem(a.id, host); } } }), ` ${a.expression.slice(0, 70)}`)),
        h("li", { text: `#${it.id} (this, ${it.mode})` }),
        lin.descendants.map((a) => h("li", {}, "↳ ", h("a", { href: "#", text: `#${a.id} (${a.mode})`, on: { click: (e) => { e.preventDefault(); showLibraryItem(a.id, host); } } }), ` ${a.expression.slice(0, 70)}`)))
        : h("p", { class: "muted", text: "No parents or children." }),
      it.brain_alpha_id ? h("p", {}, "BRAIN alpha: ", h("a", { href: d.brain_url, target: "_blank", rel: "noopener noreferrer", text: `${it.brain_alpha_id} ↗` }), ` · ${it.status}`) : null),
    resultHost);
  } catch (e) { host.replaceChildren(errorBox(e)); }
}

// =====================================================================
// Correlation
// =====================================================================
async function refreshPool() {
  const panel = $("#tab-correlation");
  const paste = h("textarea", { class: "mono", rows: 5, spellcheck: "false", placeholder: "One alpha per line. Optional name first:  My reversal | rank(-ts_delta(close, 5))" });
  const form = settingsForm({}, { compact: true });
  const checkExpr = h("textarea", { class: "mono", rows: 2, spellcheck: "false", placeholder: "Expression to test against the pool" });
  const checkForm = settingsForm({}, { compact: true });
  const checkOut = h("div");
  const list = h("div");
  const signedIn = S.state && S.state.auth.state === "signed_in";
  const load = async () => {
    const d = await api("/api/correlation/pool");
    fill(list,
      h("p", { class: "muted", text: `${d.counts.active} of your active BRAIN alphas · ${d.counts.pasted} pasted · ${d.counts.session} created this session.` }),
      d.items.length ? h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, d.items.map((k) => h("tr", {},
        h("td", {}, h("span", { class: "chip", text: k.source })), h("td", { text: k.name }), h("td", { class: "mono", text: (k.expression || "").slice(0, 70) }),
        h("td", { class: "num", text: k.sharpe != null ? `S ${num(k.sharpe)}` : "" }), h("td", { class: "muted", text: k.has_pnl ? "real PnL" : "simulated" })))))) : null);
  };
  panel.replaceChildren(
    h("div", { class: "card" },
      h("h2", { text: "Correlation checker" }),
      h("p", { class: "sub", text: "Self-correlation is measured on daily PnL changes. Your real active alphas give verified correlations for BRAIN-simulated candidates, and simulated correlations for offline screening; pasted alphas are simulated offline." }),
      h("p", { class: "muted", text: "How to read it: below 0.5 the alpha genuinely diversifies your pool (Learn2Quant Lesson 8); up to 0.7 it passes BRAIN's self-correlation check; above 0.7 it fails unless its Sharpe beats the correlated alpha by 10%." }),
      h("div", { class: "row" },
        h("button", { class: "btn primary", disabled: !signedIn, text: "Load my active alphas from BRAIN",
          on: { click: () => startTask("/api/correlation/load-active", "Loading your active alphas", load) } }),
        signedIn ? null : h("span", { class: "muted", text: "Sign in to load your real alphas." }),
        h("button", { class: "btn danger", text: "Clear pasted", on: { click: async () => { await api("/api/correlation/pasted", { method: "DELETE" }); load(); } } })),
      h("div", { "data-task-kind": "load-active" }),
      h("h3", { text: "Paste alphas" }), paste, form.el,
      h("div", { class: "row", style: { "margin-top": "8px" } }, h("button", { class: "btn", text: "Add to pool", on: { click: async () => {
        const items = paste.value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
          const i = l.indexOf("|");
          return i > 0 && !l.slice(0, i).includes("(") ? { name: l.slice(0, i).trim(), expression: l.slice(i + 1).trim(), settings: form.get() } : { expression: l, settings: form.get() };
        });
        try { const d = await api("/api/correlation/paste", { method: "POST", body: { items } }); toast(`Added ${d.added}.${d.errors.length ? " Errors: " + d.errors.join("; ") : ""}`); paste.value = ""; load(); }
        catch (e) { toast(e.message); }
      } } })),
      h("h3", { text: "Pool" }), list),
    h("div", { class: "card" }, h("h3", { text: "Check an expression" }), checkExpr, checkForm.el,
      h("div", { class: "row", style: { "margin-top": "8px" } }, h("button", { class: "btn primary", text: "Check correlation", on: { click: async () => {
        checkOut.replaceChildren(busy("Simulating and correlating..."));
        try {
          const d = await api("/api/correlation/check", { method: "POST", body: { expression: checkExpr.value, settings: checkForm.get() } });
          checkOut.replaceChildren(d.max == null ? h("p", { class: "muted", text: "The pool is empty." }) : h("div", {},
            h("p", {}, "Highest: ", h("strong", { text: num(d.max) }), ` with ${d.with_name} (simulated, daily PnL).`),
            h("table", {}, h("tbody", {}, d.rows.map((r) => h("tr", {}, h("td", { text: r.name }), h("td", {}, h("span", { class: "chip", text: r.kind })), h("td", { class: "num", text: num(r.corr) })))))));
        } catch (e) { checkOut.replaceChildren(errorBox(e)); }
      } } })), checkOut));
  load().catch((e) => list.replaceChildren(errorBox(e)));
}

// =====================================================================
// Settings & rules
// =====================================================================
async function renderSettings() {
  const panel = $("#tab-settings");
  S.meta = null;
  const m = await meta();
  const st = S.state || { prefs: {} };
  const p = st.prefs || {};
  const conc = h("input", { type: "number", min: 1, max: 10, value: p.max_concurrent_simulations || 3, style: { width: "80px" } });
  const region = h("select", {}, m.regions.map((r) => h("option", { value: r, selected: r === p.default_region, text: r })));
  const uni = h("select", {}, m.universes.map((u) => h("option", { value: u, selected: u === p.default_universe, text: u })));
  const delay = h("select", {}, [1, 0].map((d) => h("option", { value: d, selected: d === p.default_delay, text: String(d) })));
  const r = m.rules;
  const ruleRows = [["Sharpe (delay-1)", r.sharpe.delay1, ">="], ["Sharpe (delay-0)", r.sharpe.delay0, ">="],
    ["Fitness (delay-1)", r.fitness.delay1, ">="], ["Fitness (delay-0)", r.fitness.delay0, ">="],
    ["Turnover min", r.turnover.min, ">=", true], ["Turnover max", r.turnover.max, "<=", true],
    ["Max stock weight", r.weight.max, "<=", true], ["Sub-universe factor", r.sub_universe.factor, "x"],
    ["Self-correlation max", r.self_correlation.max, "<"], ["Self-corr. Sharpe improvement", r.self_correlation.sharpe_improvement, ">=", true],
    ["Returns / drawdown" + (r.returns_drawdown.enforced.value ? "" : " (advisory)"), r.returns_drawdown.min, ">="],
    ["Combine only below correlation", r.diversity.combine_max_corr, "<"],
    ...(r.drawdown.max.value != null ? [["Drawdown cap" + (r.drawdown.enforced.value ? "" : " (advisory)"), r.drawdown.max, "<=", true]] : [])];
  const opFilter = h("input", { type: "text", placeholder: "Filter operators" });
  const opTable = h("div");
  const drawOps = () => {
    const q = opFilter.value.toLowerCase();
    opTable.replaceChildren(h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, m.operators.filter((o) => !q || o.name.includes(q) || o.category.toLowerCase().includes(q)).map((o) => h("tr", {},
      h("td", { class: "mono", text: o.signature }), h("td", { text: o.category }), h("td", { class: "why", text: o.description }),
      h("td", {}, o.platform_only ? h("span", { class: "chip", text: "BRAIN only" }) : o.available ? badge("PASS") : h("span", { class: "muted", text: "not on your account" }))))))));
  };
  opFilter.addEventListener("input", drawOps);
  drawOps();
  const signedIn = st.auth && st.auth.state === "signed_in";
  panel.replaceChildren(
    h("div", { class: "card" }, h("h2", { text: "Preferences" }),
      h("div", { class: "row" },
        h("label", { class: "field" }, "Max concurrent BRAIN simulations", conc),
        h("label", { class: "field" }, "Region for field sync", region),
        h("label", { class: "field" }, "Universe for field sync", uni),
        h("label", { class: "field" }, "Delay for field sync", delay),
        h("button", { class: "btn primary", text: "Save", on: { click: async () => {
          try { await api("/api/prefs", { method: "POST", body: { max_concurrent_simulations: +conc.value, default_region: region.value, default_universe: uni.value, default_delay: +delay.value } }); toast("Saved."); poll(); }
          catch (e) { toast(e.message); }
        } } })),
      h("p", { class: "muted", text: "Community clients report about 3 concurrent simulations for most accounts; raise it only if your account allows more." })),
    courseCard(m),
    riskCard(m),
    h("div", { class: "card" }, h("h2", { text: "BRAIN data fields & operators" }),
      h("p", { class: "sub", text: `${m.fields.platform_count} fields synced from your account${m.fields.platform ? "" : " (none yet: Create uses offline placeholder names for alternative data)"}. Generated alphas only use synced fields once synced.` }),
      h("button", { class: "btn", disabled: !signedIn, text: "Sync from BRAIN", on: { click: () => startTask("/api/meta/sync", "Syncing fields and operators", () => renderSettings()) } }),
      h("div", { "data-task-kind": "sync-fields" })),
    h("div", { class: "card" }, h("h2", { text: "Submission rules" }),
      h("p", { class: "sub" }, "Thresholds used by every check. Put the exact figures from your course's “Read this First” document in ",
        h("code", { text: m.rules_path }), "; they override the defaults and BRAIN-learned limits. Then click Reload."),
      m.rule_conflicts.length ? h("div", { class: "warn-box" }, m.rule_conflicts.map((c) => h("div", { text: c }))) : null,
      h("table", {}, h("thead", {}, h("tr", {}, h("th", { text: "Rule" }), h("th", { class: "num", text: "Threshold" }), h("th", { text: "Source" }))),
        h("tbody", {}, ruleRows.map(([label, v, op, isPct]) => h("tr", {}, h("td", { text: label }),
          h("td", { class: "num", text: `${op} ${isPct ? pct(v.value) : v.value}` }), h("td", { class: "src", text: v.source }))))),
      h("p", { class: "muted", text: "Fitness = Sharpe × √(|Returns| / max(Turnover, 12.5%)). Sub-universe Sharpe must be ≥ factor × √(sub size / universe size) × Sharpe (community-reported formula; BRAIN's own limit is shown once verified). Self-correlation above the max passes only if Sharpe beats the correlated alpha by the improvement margin." }),
      h("button", { class: "btn", text: "Reload rules file", on: { click: async () => { await api("/api/rules/reload", { method: "POST" }); S.meta = null; renderSettings(); toast("Rules reloaded."); } } })),
    h("div", { class: "card" }, h("h2", { text: "Simulation fidelity" }),
      h("p", { class: "sub", text: "Where the offline engine approximates BRAIN:" }),
      h("ul", { class: "fix-list" }, m.fidelity.map((f) => h("li", { text: f })), Object.values(m.approximations).map((a) => h("li", { text: a })))),
    h("div", { class: "card" }, h("h2", { text: "Operators" }), opFilter, opTable));
}

function courseCard(m) {
  const out = h("div");
  return h("div", { class: "card" }, h("h2", { text: "What the course teaches" }),
    h("p", { class: "sub", text: `Rules and guidance from ${m.course.source}, built into the checks, Create and repair.` }),
    h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, m.course.principles.map(([k, v, src]) =>
      h("tr", {}, h("td", {}, h("strong", { text: k })), h("td", { text: v }), h("td", { class: "src", text: src })))))),
    h("div", { class: "row", style: { "margin-top": "10px" } },
      h("button", { class: "btn", text: "Calibrate the offline engine from the lesson results", on: { click: async () => {
        out.replaceChildren(busy("Simulating each lesson alpha offline and logging the gap to BRAIN's numbers..."));
        try {
          const d = await api("/api/course/calibrate", { method: "POST" });
          out.replaceChildren(h("div", { class: "table-wrap" }, h("table", {},
            h("thead", {}, h("tr", {}, ["Lesson alpha", "Sharpe offline / BRAIN", "Turnover offline / BRAIN", "Fitness offline / BRAIN"].map((t) => h("th", { text: t })))),
            h("tbody", {}, d.rows.map((r) => h("tr", {}, h("td", { text: r.title }),
              h("td", { class: "num", text: `${num(r.sharpe_offline)} / ${num(r.sharpe_brain)}` }),
              h("td", { class: "num", text: `${pct(r.turnover_offline)} / ${pct(r.turnover_brain)}` }),
              h("td", { class: "num", text: `${num(r.fitness_offline)} / ${num(r.fitness_brain)}` })))))),
            h("p", { class: "muted", text: "Logged to the calibration log: results now show a “Calibrated estimate” column based on these gaps." }));
        } catch (e) { out.replaceChildren(errorBox(e)); }
      } } })),
    out);
}

function riskCard(m) {
  const r = m.risk || {};
  const field = (v) => h("input", { type: "number", min: 0, max: 100, step: 0.5, value: v != null ? +(v * 100).toFixed(2) : "",
    placeholder: "off", style: { width: "90px" } });
  const to = field(r.max_turnover), w = field(r.max_weight), sec = field(r.max_sector_net);
  const val = (el) => (el.value === "" ? null : +el.value / 100);
  return h("div", { class: "card" }, h("h2", { text: "Risk controls" }),
    h("p", { class: "sub", text: "Your own limits, on top of BRAIN's rules. When set, every check table shows them, and Create and repair steer candidates to satisfy them. Leave blank for off." }),
    h("div", { class: "row" },
      h("label", { class: "field" }, "Turnover ceiling (%)", to),
      h("label", { class: "field" }, "Max position weight (%)", w),
      h("label", { class: "field" }, "Max net sector exposure (%)", sec),
      h("button", { class: "btn primary", text: "Save risk controls", on: { click: async () => {
        try { await api("/api/rules/risk", { method: "POST", body: { max_turnover: val(to), max_weight: val(w), max_sector_net: val(sec) } }); S.meta = null; toast("Risk controls saved."); }
        catch (e) { toast(e.message); }
      } } })));
}

// =====================================================================
// boot
// =====================================================================
async function boot() {
  initTheme();
  initTabs();
  $("#auth-pill").addEventListener("click", openAuth);
  $("#jobs-pill").addEventListener("click", () => { renderJobsDialog((S.state && S.state.jobs) || []); $("#jobs-dialog").showModal(); });
  await poll();
  try { await meta(); } catch (e) { $("#tab-create").replaceChildren(errorBox(e)); return; }
  renderCreate();
  renderRecreate();
  renderReconcile();
  setInterval(poll, 3000);
  if (S.state && ["session_expired", "verification_needed"].includes(S.state.auth.state)) openAuth();
}

boot();
