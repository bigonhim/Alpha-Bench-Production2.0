// Small SVG chart kit: line/area with crosshair tooltip, and diverging bars.
// Marks carry series colours; all text uses ink tokens. Labels go in via textContent.

const NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, style = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const [k, v] of Object.entries(style)) el.style.setProperty(k, v);
  return el;
}

function div(cls, text) {
  const el = document.createElement("div");
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

function niceTicks(lo, hi, count = 4) {
  if (!isFinite(lo) || !isFinite(hi)) return [0];
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count + 0.5) || 10 * mag;
  const start = Math.floor(lo / step) * step;
  const out = [];
  for (let v = start; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}

function placeTooltip(tip, container, x, y) {
  const w = container.clientWidth;
  const tw = tip.offsetWidth;
  tip.style.left = Math.min(Math.max(0, x + 14), Math.max(0, w - tw)) + "px";
  tip.style.top = Math.max(0, y - 10) + "px";
}

/**
 * Line chart. opts: {title, subtitle, dates[], series:[{name, values[], color}], refs:[{y,label}],
 *                    fmt(v), area:bool, height}
 */
export function lineChart(container, opts) {
  const { title, subtitle, dates, series, refs = [], fmt = (v) => v.toFixed(2), area = false } = opts;
  container.replaceChildren();
  // Draw at the container's real pixel width so text stays at its true size.
  const W = Math.max(300, Math.round(container.clientWidth || 640));
  const H = opts.height || 220, M = { t: 10, r: 64, b: 24, l: 56 };
  container.classList.add("chart");
  if (title) { const h = document.createElement("h4"); h.textContent = title; container.append(h); }
  if (subtitle) container.append(div("chart-sub", subtitle));
  if (series.length > 1) {
    const lg = div("legend");
    for (const s of series) {
      const item = document.createElement("span");
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = s.color;
      item.append(key, document.createTextNode(s.name));
      lg.append(item);
    }
    container.append(lg);
  }
  const n = dates.length;
  const vals = series.flatMap((s) => s.values.filter((v) => v != null && isFinite(v)));
  refs.forEach((r) => vals.push(r.y));
  if (!n || !vals.length) { container.append(div("muted", "No data.")); return; }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (area || opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const ticks = niceTicks(lo, hi, 4);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const x = (i) => M.l + (i / Math.max(1, n - 1)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - (v - lo) / (hi - lo || 1)) * (H - M.t - M.b);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${title || "Chart"}: ${series.map((s) => s.name).join(", ")}` });
  for (const t of ticks) {
    root.append(svg("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: t === 0 ? "baseline" : "gridline" }));
    const lab = svg("text", { x: M.l - 8, y: y(t) + 4, "text-anchor": "end", class: "tick" });
    lab.textContent = fmt(t);
    root.append(lab);
  }
  // Year ticks on the x axis.
  let lastYear = null;
  dates.forEach((d, i) => {
    const yr = String(d).slice(0, 4);
    if (yr !== lastYear) {
      lastYear = yr;
      if (i > 0 || n < 30) {
        const lab = svg("text", { x: x(i), y: H - 6, "text-anchor": "middle", class: "tick" });
        lab.textContent = yr;
        root.append(lab);
      }
    }
  });
  for (const r of refs) {
    root.append(svg("line", { x1: M.l, x2: W - M.r, y1: y(r.y), y2: y(r.y), class: "ref" }));
    // Reference labels sit at the left, just above their line: the right edge belongs to end labels.
    const lab = svg("text", { x: M.l + 4, y: y(r.y) - 4, class: "ref-label" });
    lab.textContent = r.label;
    root.append(lab);
  }
  for (const s of series) {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    if (area && d) {
      const first = s.values.findIndex((v) => v != null);
      let lastIdx = s.values.length - 1;
      while (lastIdx > 0 && s.values[lastIdx] == null) lastIdx--;
      root.append(svg("path", { d: `${d}L${x(lastIdx)},${y(0)}L${x(first)},${y(0)}Z` },
        { fill: s.color, "fill-opacity": "0.1", stroke: "none" }));
    }
    root.append(svg("path", { d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" },
      { stroke: s.color }));
    // End label + end dot for single-series charts (the story's number).
    let li = s.values.length - 1;
    while (li > 0 && (s.values[li] == null || !isFinite(s.values[li]))) li--;
    if (li >= 0 && s.values[li] != null && series.length <= 2) {
      root.append(svg("circle", { cx: x(li), cy: y(s.values[li]), r: 4, "stroke-width": 2 },
        { fill: s.color, stroke: "var(--surface)" }));
      const lab = svg("text", { x: x(li) + 8, y: y(s.values[li]) + 4, class: "end-label" });
      lab.textContent = fmt(s.values[li]);
      root.append(lab);
    }
  }
  const cross = svg("line", { y1: M.t, y2: H - M.b, class: "crosshair", visibility: "hidden" });
  root.append(cross);
  const dots = series.map((s) => {
    const c = svg("circle", { r: 4, "stroke-width": 2, visibility: "hidden" }, { fill: s.color, stroke: "var(--surface)" });
    root.append(c);
    return c;
  });
  const hit = svg("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent", tabindex: 0 });
  root.append(hit);
  container.append(root);

  const tip = div("tooltip");
  tip.hidden = true;
  container.append(tip);
  const show = (i, px, py) => {
    i = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(div("t-date", String(dates[i])));
    series.forEach((s, k) => {
      const v = s.values[i];
      if (v == null || !isFinite(v)) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", x(i)); dots[k].setAttribute("cy", y(v)); dots[k].setAttribute("visibility", "visible");
      const row = div("t-row");
      const key = document.createElement("span"); key.className = "t-key"; key.style.background = s.color;
      const val = document.createElement("span"); val.className = "t-val"; val.textContent = fmt(v);
      const name = document.createElement("span"); name.className = "t-name"; name.textContent = s.name;
      row.append(key, val, name);
      tip.append(row);
    });
    tip.hidden = false;
    placeTooltip(tip, container, px, py);
  };
  const hide = () => {
    tip.hidden = true; cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  };
  let focusIdx = n - 1;
  hit.addEventListener("pointermove", (e) => {
    const box = root.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((sx - M.l) / (W - M.l - M.r)) * (n - 1));
    const cbox = container.getBoundingClientRect();
    show(i, e.clientX - cbox.left, e.clientY - cbox.top);
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(focusIdx, container.clientWidth / 2, 40));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      focusIdx = Math.max(0, Math.min(n - 1, focusIdx + (e.key === "ArrowLeft" ? -5 : 5)));
      show(focusIdx, container.clientWidth / 2, 40);
      e.preventDefault();
    }
  });
}

/** Horizontal diverging bars around zero. opts: {title, subtitle, items:[{label,value}], fmt} */
export function divergingBars(container, opts) {
  const { title, subtitle, items, fmt = (v) => v.toFixed(2) } = opts;
  container.replaceChildren();
  const W = Math.max(300, Math.round(container.clientWidth || 640));
  container.classList.add("chart");
  if (title) { const h = document.createElement("h4"); h.textContent = title; container.append(h); }
  if (subtitle) container.append(div("chart-sub", subtitle));
  if (!items.length) { container.append(div("muted", "No data.")); return; }
  const band = 22, thick = 14, M = { t: 4, r: 12, b: 4, l: 84 }, LABEL = 48;
  const H = M.t + M.b + band * items.length;
  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);
  const zero = M.l + (W - M.l - M.r) / 2;
  // Leave room for the value label beyond each bar end so it never collides with category labels.
  const scale = Math.max(1, (W - M.l - M.r) / 2 - LABEL) / maxAbs;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title || "Bar chart" });
  root.append(svg("line", { x1: zero, x2: zero, y1: M.t, y2: H - M.b, class: "baseline" }));
  const tip = div("tooltip");
  tip.hidden = true;
  items.forEach((d, i) => {
    const cy = M.t + i * band + band / 2;
    const len = Math.abs(d.value) * scale;
    const pos = d.value >= 0;
    const r = Math.min(4, len);
    const x0 = zero, x1 = pos ? zero + len : zero - len;
    const top = cy - thick / 2, bot = cy + thick / 2;
    // Square at the baseline, 4px rounded at the data end.
    const path = pos
      ? `M${x0},${top}H${x1 - r}Q${x1},${top} ${x1},${top + r}V${bot - r}Q${x1},${bot} ${x1 - r},${bot}H${x0}Z`
      : `M${x0},${top}H${x1 + r}Q${x1},${top} ${x1},${top + r}V${bot - r}Q${x1},${bot} ${x1 + r},${bot}H${x0}Z`;
    const bar = svg("path", { d: path }, { fill: pos ? "var(--div-pos)" : "var(--div-neg)" });
    root.append(bar);
    const lab = svg("text", { x: M.l - 8, y: cy + 4, "text-anchor": "end", class: "tick" });
    lab.textContent = d.label;
    root.append(lab);
    const val = svg("text", { x: pos ? x1 + 6 : x1 - 6, y: cy + 4, "text-anchor": pos ? "start" : "end", class: "tick" });
    val.textContent = fmt(d.value);
    root.append(val);
    const hit = svg("rect", { x: M.l, y: cy - band / 2, width: W - M.l - M.r, height: band, fill: "transparent", tabindex: 0 });
    const on = (px, py) => {
      bar.style.setProperty("opacity", "0.8");
      tip.replaceChildren();
      const row = div("t-row");
      const v = document.createElement("span"); v.className = "t-val"; v.textContent = fmt(d.value);
      const nm = document.createElement("span"); nm.className = "t-name"; nm.textContent = d.label;
      row.append(v, nm);
      tip.append(row);
      if (d.extra) tip.append(div("t-date", d.extra));
      tip.hidden = false;
      placeTooltip(tip, container, px, py);
    };
    const off = () => { bar.style.removeProperty("opacity"); tip.hidden = true; };
    hit.addEventListener("pointermove", (e) => {
      const cbox = container.getBoundingClientRect();
      on(e.clientX - cbox.left, e.clientY - cbox.top);
    });
    hit.addEventListener("pointerleave", off);
    hit.addEventListener("focus", () => on(container.clientWidth / 2, cy));
    hit.addEventListener("blur", off);
    root.append(hit);
  });
  container.append(root, tip);
  // Table view: every value reachable without hovering.
  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "Table view";
  sum.className = "muted";
  const tbl = document.createElement("table");
  const head = tbl.createTHead().insertRow();
  for (const h of ["Group", "Value"]) { const th = document.createElement("th"); th.textContent = h; head.append(th); }
  const tb = tbl.createTBody();
  for (const d of items) {
    const tr = tb.insertRow();
    tr.insertCell().textContent = d.label;
    const c = tr.insertCell(); c.className = "num"; c.textContent = fmt(d.value) + (d.extra ? ` (${d.extra})` : "");
  }
  det.append(sum, tbl);
  container.append(det);
}
