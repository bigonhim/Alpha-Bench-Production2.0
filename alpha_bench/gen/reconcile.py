"""Reconcile: compare Alpha Bench's offline prediction with real BRAIN results,
localise what failed, propose minimal fixes calibrated to BRAIN's numbers, and
log the deltas so offline estimates improve over time.
"""
from __future__ import annotations

import io
import re
import statistics
from typing import Any

from ..engine.evaluator import FieldCatalog
from ..engine.simulate import FIDELITY_NOTES, OfflineEngine
from ..expr.classify import classify
from ..expr.parser import parse
from ..rules.engine import Check, Rules, evaluate, pct
from ..settings_model import SimSettings
from .repair import assess, propose, rank_fixes

METRICS = ["sharpe", "fitness", "returns", "turnover", "drawdown", "margin", "long_count", "short_count",
           "sub_universe_sharpe", "max_weight", "sharpe_2y", "self_correlation"]
PERCENT_METRICS = {"returns", "turnover", "drawdown", "max_weight"}
_LABELS = {"sharpe": "Sharpe", "fitness": "Fitness", "returns": "Returns", "turnover": "Turnover",
           "drawdown": "Drawdown", "margin": "Margin", "long_count": "Long count", "short_count": "Short count",
           "sub_universe_sharpe": "Sub-universe Sharpe", "max_weight": "Max weight", "sharpe_2y": "2-year Sharpe",
           "self_correlation": "Self-correlation"}
_ALIASES = {"sharpe": "sharpe", "fitness": "fitness", "returns": "returns", "return": "returns",
            "turnover": "turnover", "drawdown": "drawdown", "margin": "margin", "long count": "long_count",
            "short count": "short_count", "longcount": "long_count", "shortcount": "short_count",
            "self correlation": "self_correlation", "self-correlation": "self_correlation",
            "sub-universe sharpe": "sub_universe_sharpe", "sub universe sharpe": "sub_universe_sharpe"}
_NUM = r"(-?\d+(?:[.,]\d+)?)\s*(%|‱|bps|bp)?"


def _to_value(metric: str, num: str, unit: str | None) -> float:
    v = float(num.replace(",", "."))
    if unit == "%" or (metric in PERCENT_METRICS and unit is None and abs(v) > 1.5):
        return v / 100.0
    return v


def parse_pasted_metrics(text: str) -> dict[str, Any]:
    """Read metrics and check results from text copied out of the BRAIN UI (or OCR output)."""
    text = text.replace(" ", " ").replace("\xa0", " ")
    found: dict[str, float] = {}
    # Header row followed by a value row (BRAIN's IS summary table).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines[:-1]):
        heads = [h for h in re.split(r"\s{2,}|\t|\|", ln.lower()) if h.strip()]
        keys = [_ALIASES.get(h.strip()) for h in heads]
        if sum(k is not None for k in keys) >= 3:
            vals = re.findall(_NUM, lines[i + 1])
            if len(vals) >= sum(k is not None for k in keys):
                for k, (num, unit) in zip([k for k in keys if k], vals):
                    found.setdefault(k, _to_value(k, num, unit or None))
    # "Sharpe 1.23", "Turnover: 45.6%", "Self-correlation 0.81"
    alias_re = "|".join(sorted((re.escape(a) for a in _ALIASES), key=len, reverse=True))
    for m in re.finditer(rf"(?i)\b({alias_re})\b\s*[:=]?\s*{_NUM}", text):
        key = _ALIASES[m.group(1).lower()]
        found.setdefault(key, _to_value(key, m.group(2), m.group(3)))
    checks = []
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{5,})\b[^\n]*?\b(PASS|FAIL|WARNING|PENDING|ERROR)\b", text):
        checks.append({"name": m.group(1), "result": m.group(2)})
    return {"metrics": found, "checks": checks}


def ocr_image(data: bytes) -> tuple[str | None, str | None]:
    """Local OCR only (Tesseract). Screenshots are never sent to any online service."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None, ("Local OCR isn't installed. Install Tesseract OCR and `pip install pytesseract` to read "
                      "screenshots, or paste the numbers as text. Screenshots are never uploaded anywhere.")
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data))), None
    except Exception as exc:  # noqa: BLE001 - tesseract binary missing, bad image...
        return None, f"OCR failed locally ({type(exc).__name__}). Paste the numbers as text instead."


def _localise(check: dict, brain_m: dict, off_m: dict) -> str:
    name, value, limit = check.get("name"), check.get("value"), check.get("limit")
    def fmt(metric, v):
        if v is None:
            return "n/a"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)  # already a display string such as ">= 1.0"
        return pct(f) if metric in PERCENT_METRICS else f"{f:.2f}"
    metric = {"LOW_SHARPE": "sharpe", "LOW_FITNESS": "fitness", "LOW_TURNOVER": "turnover",
              "HIGH_TURNOVER": "turnover", "CONCENTRATED_WEIGHT": "max_weight",
              "LOW_SUB_UNIVERSE_SHARPE": "sub_universe_sharpe", "SELF_CORRELATION": "self_correlation",
              "LOW_2Y_SHARPE": "sharpe_2y", "PROD_CORRELATION": "prod_correlation"}.get(name)
    if value is None and metric:
        value = brain_m.get(metric)
    offline = off_m.get(metric) if metric else None
    bits = [f"{name}: BRAIN measured {fmt(metric, value)}"]
    if limit is not None:
        bits.append(f"against a limit of {fmt(metric, limit)}")
    if offline is not None:
        bits.append(f"(Alpha Bench predicted {fmt(metric, offline)})")
    return " ".join(bits) + "."


def reconcile(engine: OfflineEngine, rules: Rules, catalog: FieldCatalog, *, expression: str,
              settings: SimSettings, brain_metrics: dict, brain_checks: list[dict] | None = None,
              corr_fn=None, brain_correlation: dict | None = None) -> dict:
    node = parse(expression)
    base = assess(engine, rules, node, settings, corr_fn, full=True)
    off = base["sim"].metrics
    rows = []
    for k in METRICS:
        b, o = brain_metrics.get(k), off.get(k)
        if b is None and o is None:
            continue
        row = {"metric": k, "label": _LABELS[k], "offline": o, "brain": b, "percent": k in PERCENT_METRICS}
        if b is not None and o is not None:
            row["delta"] = float(b) - float(o)
            row["ratio"] = (float(b) / float(o)) if o else None
        rows.append(row)

    # Ground truth: BRAIN's own checks when available, otherwise our rules applied to BRAIN's numbers.
    if brain_checks:
        failing = [c for c in brain_checks if str(c.get("result")).upper() == "FAIL"]
        truth: list[Check] = [Check(c.get("name", "?"), c.get("name", "?"), "", c.get("value"),
                                    str(c.get("limit", "")), str(c.get("result")).upper(),
                                    _localise(c, brain_metrics, off), [], "BRAIN") for c in brain_checks]
        learned = rules.learn_from_brain(brain_checks, settings.delay, settings.region)
    else:
        bm = {**{k: v for k, v in off.items() if k not in brain_metrics}, **brain_metrics}
        truth = evaluate(bm, settings, rules, correlation=brain_correlation)
        failing = [{"name": c.name, "value": c.value, "limit": c.limit} for c in truth if c.result == "FAIL"]
        learned = []

    diagnosis = [_localise(c, brain_metrics, off) for c in failing]
    structure = classify(node, catalog.category)
    gaps = _gap_reasons(rows, base["sim"].proxied, settings)

    # Fixes: propose from BRAIN's failures, measure each fix's *effect* offline, and apply that effect
    # to BRAIN's baseline numbers (delta for Sharpe/fitness/returns, ratio for turnover).
    truth_checks = [Check(c["name"], c["name"], "", c.get("value"), str(c.get("limit")), "FAIL", "", [], "BRAIN")
                    for c in failing] if brain_checks else [c for c in truth if c.result == "FAIL"]
    fixes = propose(node, settings, truth_checks, {**off, **brain_metrics}, catalog, brain_correlation)
    ranked = rank_fixes(engine, rules, base, fixes, corr_fn, limit=8)
    options = []
    for f in ranked:
        d = f.result.get("delta", {})
        pred = {}
        for k in ("sharpe", "fitness", "returns"):
            if brain_metrics.get(k) is not None:
                pred[k] = float(brain_metrics[k]) + float(d.get(k, 0.0))
        if brain_metrics.get("turnover") is not None and off.get("turnover"):
            new_to = f.result["metrics"].get("turnover") or off["turnover"]
            pred["turnover"] = float(brain_metrics["turnover"]) * new_to / off["turnover"]
        clears = _predict_clears(pred, failing, rules, settings)
        options.append({"title": f.title, "change": f.change, "detail": f.detail, "size": f.size,
                        "expression": f.result["expression"], "settings": f.result["settings"],
                        "predicted_brain": pred, "predicted_clears": clears,
                        "offline_cleared": f.result.get("cleared", [])})
    options.sort(key=lambda o: (-len(o["predicted_clears"]), o["size"]))
    return {"rows": rows, "diagnosis": diagnosis, "failing": [c.get("name") for c in failing],
            "truth_checks": [c.to_dict() for c in truth], "structure": structure.to_dict(),
            "gap_reasons": gaps, "fixes": options, "learned_limits": learned,
            "offline": {"sim": base["sim"].to_dict(), "checks": [c.to_dict() for c in base["checks"]]}}


def _predict_clears(pred: dict, failing: list[dict], rules: Rules, s: SimSettings) -> list[str]:
    out = []
    for c in failing:
        name, limit = c.get("name"), c.get("limit")
        try:
            lim = float(limit) if limit is not None else None
        except (TypeError, ValueError):
            lim = None
        if name == "LOW_SHARPE" and "sharpe" in pred:
            if pred["sharpe"] >= (lim if lim is not None else rules.value("sharpe", s.delay, s.region)[0]):
                out.append(name)
        elif name == "LOW_FITNESS" and "fitness" in pred:
            if pred["fitness"] >= (lim if lim is not None else rules.value("fitness", s.delay, s.region)[0]):
                out.append(name)
        elif name == "HIGH_TURNOVER" and "turnover" in pred:
            if pred["turnover"] <= (lim if lim is not None else rules.value("turnover.max", region=s.region)[0]):
                out.append(name)
        elif name == "LOW_TURNOVER" and "turnover" in pred:
            if pred["turnover"] >= (lim if lim is not None else rules.value("turnover.min", region=s.region)[0]):
                out.append(name)
    return out


def _gap_reasons(rows: list[dict], proxied: list[str], s: SimSettings) -> list[str]:
    out = []
    by = {r["metric"]: r for r in rows}
    if proxied:
        out.append(f"{len(proxied)} field(s) had no offline data and used category proxies ({', '.join(proxied[:4])}"
                   f"{'...' if len(proxied) > 4 else ''}). Expect the largest gaps in Sharpe and returns.")
    t = by.get("turnover")
    if t and t.get("ratio") and (t["ratio"] > 1.3 or t["ratio"] < 0.77):
        out.append(f"Turnover differs by x{t['ratio']:.2f}: the real fields update "
                   f"{'more' if t['ratio'] > 1 else 'less'} often than their offline stand-ins.")
    sh = by.get("sharpe")
    if sh and sh.get("delta") is not None and abs(sh["delta"]) > 0.5:
        out.append("Sharpe gap > 0.5: offline Sharpe comes from a synthetic market, so it ranks ideas but can't "
                   "predict BRAIN's level. The calibration log narrows this over time.")
    if s.delay == 0:
        out.append("Delay-0: offline uses the same-day close, BRAIN uses an intraday snapshot.")
    if s.neutralization.upper() not in ("NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"):
        out.append(f"{s.neutralization} neutralization is approximated offline by SUBINDUSTRY.")
    out.append(FIDELITY_NOTES[0])
    return out


class Calibration:
    """Learns how BRAIN's numbers relate to offline ones from logged reconciliations."""

    def __init__(self, rows: list[dict]):
        self.rows = rows  # {metric, offline, brain, family}

    def _pairs(self, metric: str, family: str | None):
        pts = [r for r in self.rows if r["metric"] == metric and r["offline"] is not None and r["brain"] is not None]
        fam = [r for r in pts if family and r.get("family") == family]
        return (fam, "family") if len(fam) >= 5 else (pts, "all")

    def adjust(self, metrics: dict, family: str | None = None) -> dict:
        out = {}
        for m in ("sharpe", "fitness", "returns"):
            pts, scope = self._pairs(m, family)
            if len(pts) >= 3 and metrics.get(m) is not None:
                shift = statistics.median(r["brain"] - r["offline"] for r in pts)
                out[m] = {"value": metrics[m] + shift, "n": len(pts), "scope": scope, "method": "median shift"}
        pts, scope = self._pairs("turnover", family)
        pts = [r for r in pts if r["offline"]]
        if len(pts) >= 3 and metrics.get("turnover"):
            ratio = statistics.median(r["brain"] / r["offline"] for r in pts)
            out["turnover"] = {"value": metrics["turnover"] * ratio, "n": len(pts), "scope": scope,
                               "method": "median ratio"}
        if "sharpe" in out and "returns" in out:
            to = out.get("turnover", {}).get("value", metrics.get("turnover", 0.0))
            s, r = out["sharpe"]["value"], out["returns"]["value"]
            out["fitness"] = {"value": s * (abs(r) / max(to, 0.125)) ** 0.5, "n": out["sharpe"]["n"],
                              "scope": out["sharpe"]["scope"], "method": "recomputed"}
        return out

    def summary(self) -> dict:
        res = {}
        for m in ("sharpe", "fitness", "returns", "turnover"):
            pts = [r for r in self.rows if r["metric"] == m and r["offline"] is not None and r["brain"] is not None]
            if pts:
                res[m] = {"n": len(pts), "median_delta": statistics.median(r["brain"] - r["offline"] for r in pts)}
        return res
