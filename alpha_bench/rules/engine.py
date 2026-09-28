"""Submission rules: explicit, checkable thresholds with plain-language reasons.

Threshold precedence (highest first):
  1. rules.json in the app folder: your own figures (e.g. from a course document).
  2. Limits BRAIN itself reported in check results for your alphas
     (learned automatically when results are pulled; see `learn_from_brain`).
  3. Built-in defaults: rules taught in WorldQuant's Learn2Quant lessons (labelled with the lesson),
     otherwise BRAIN's publicly described criteria.
Every check reports which source its threshold came from.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..settings_model import SUB_UNIVERSE, UNIVERSE_SIZE, SimSettings

DEFAULT_RULES: dict[str, Any] = {
    "_about": ("Edit this file with the exact figures from your 'Read this First' document. Values here "
               "override both the built-in defaults and limits learned from BRAIN check results."),
    "sharpe": {"delay1": 1.25, "delay0": 2.0},
    "fitness": {"delay1": 1.0, "delay0": 1.3},
    "turnover": {"min": 0.01, "max": 0.70},
    "weight": {"max": 0.10},
    # Community-reported formula (not confirmed by WorldQuant documentation): sub-universe Sharpe must be
    # >= factor * sqrt(sub_size / universe_size) * Sharpe. BRAIN's own per-alpha limit is shown when available.
    "sub_universe": {"factor": 0.75},
    "self_correlation": {"max": 0.70, "sharpe_improvement": 0.10, "window_years": 4},
    # Learn2Quant Lesson 3: "target a return to drawdown ratio greater than one". Advisory (WARN) unless
    # enforced, because BRAIN's submission checks don't block on it.
    "returns_drawdown": {"min": 1.0, "enforced": False},
    # Optional absolute drawdown cap (off by default; the course's drawdown standard is the ratio above).
    "drawdown": {"max": None, "enforced": False},
    # Learn2Quant Lesson 8: combine alphas whose correlation is below 0.5.
    "diversity": {"combine_max_corr": 0.5},
    # LOW_2Y_SHARPE is a real BRAIN check, so it is enforced: a signal that faded in the last two
    # years comes back "Needs Improvement" even when the whole-period Sharpe looks fine.
    "two_year_sharpe": {"delay1": 1.25, "delay0": 2.0, "enforced": True},
    # BRAIN outlook: offline numbers run optimistic, so an offline pass within `margin` of the
    # Sharpe/fitness floors is treated as borderline until verified (or calibrated) on BRAIN.
    "outlook": {"margin": 1.3},
    # Your own risk limits (null = off). Stricter than BRAIN's rules; Create and repair respect them.
    "risk": {"max_turnover": None, "max_weight": None, "max_sector_net": None},
}

# BRAIN check name -> (rules path, delay-specific?). Limits are learned per region: BRAIN's thresholds
# differ by region (e.g. CHN requires a higher Sharpe).
BRAIN_CHECK_MAP = {
    "LOW_SHARPE": ("sharpe", True), "LOW_FITNESS": ("fitness", True),
    "LOW_TURNOVER": ("turnover.min", False), "HIGH_TURNOVER": ("turnover.max", False),
    "CONCENTRATED_WEIGHT": ("weight.max", False), "SELF_CORRELATION": ("self_correlation.max", False),
    "LOW_2Y_SHARPE": ("two_year_sharpe", True),
}


# Defaults that come straight from the lessons are labelled with their source.
COURSE_SOURCES = {
    "returns_drawdown.min": "Learn2Quant Lesson 3", "returns_drawdown.enforced": "Learn2Quant Lesson 3",
    "diversity.combine_max_corr": "Learn2Quant Lesson 8",
}


def _get(d: dict, path: str):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _set(d: dict, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


class Rules:
    def __init__(self, folder: Path | None = None):
        self.folder = folder
        self.user: dict = {}
        self.learned: dict = {}
        self.reload()

    @property
    def user_path(self) -> Path | None:
        return self.folder / "rules.json" if self.folder else None

    @property
    def learned_path(self) -> Path | None:
        return self.folder / "rules_learned_from_brain.json" if self.folder else None

    def reload(self) -> None:
        self.user = self._read(self.user_path)
        self.learned = self._read(self.learned_path)
        if self.user_path and not self.user_path.exists():
            try:
                self.user_path.write_text(json.dumps(DEFAULT_RULES, indent=2), encoding="utf-8")
            except OSError:
                pass
            self.user = {}  # a fresh copy of the defaults isn't a user override

    @staticmethod
    def _read(path: Path | None) -> dict:
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _user_overrides(self, path: str):
        """A user value counts only if it differs from the shipped default."""
        uv = _get(self.user, path)
        return None if uv is None or uv == _get(DEFAULT_RULES, path) else uv

    def value(self, path: str, delay: int | None = None, region: str = "USA") -> tuple[Any, str]:
        full = f"{path}.delay{delay}" if delay is not None and isinstance(_get(DEFAULT_RULES, path), dict) \
            and f"delay{delay}" in _get(DEFAULT_RULES, path) else path
        uv = self._user_overrides(full)
        if uv is not None:
            return uv, "your rules file"
        lv = _get(self.learned.get(region, {}), full)
        if lv is not None:
            return lv, f"BRAIN check limit ({region})"
        return _get(DEFAULT_RULES, full), COURSE_SOURCES.get(full, "default (public BRAIN criteria)")

    def effective(self, region: str = "USA") -> dict:
        out = copy.deepcopy(DEFAULT_RULES)
        for path in ("sharpe.delay1", "sharpe.delay0", "fitness.delay1", "fitness.delay0", "turnover.min",
                     "turnover.max", "weight.max", "sub_universe.factor", "self_correlation.max",
                     "self_correlation.sharpe_improvement", "drawdown.max", "drawdown.enforced",
                     "returns_drawdown.min", "returns_drawdown.enforced", "diversity.combine_max_corr",
                     "two_year_sharpe.delay1", "two_year_sharpe.delay0", "two_year_sharpe.enforced"):
            v, src = self.value(path, region=region)
            _set(out, path, {"value": v, "source": src})
        return out

    def risk(self) -> dict:
        out = dict(DEFAULT_RULES["risk"])
        out.update({k: v for k, v in (self.user.get("risk") or {}).items() if k in out})
        return out

    def set_risk(self, values: dict) -> dict:
        data = self._read(self.user_path) or json.loads(json.dumps(DEFAULT_RULES))
        risk = data.setdefault("risk", {})
        for k in DEFAULT_RULES["risk"]:
            if k in values:
                v = values[k]
                risk[k] = None if v in (None, "", 0) else float(v)
        if self.user_path:
            self.user_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.reload()
        return self.risk()

    def conflicts(self, region: str = "USA") -> list[str]:
        out = []
        for path in ("sharpe.delay1", "sharpe.delay0", "fitness.delay1", "fitness.delay0", "turnover.min",
                     "turnover.max", "weight.max", "self_correlation.max"):
            uv, lv = self._user_overrides(path), _get(self.learned.get(region, {}), path)
            if uv is not None and lv is not None and abs(float(uv) - float(lv)) > 1e-9:
                out.append(f"{path}: your rules file says {uv}, but BRAIN's latest {region} checks used {lv}.")
        return out

    def learn_from_brain(self, checks: list[dict], delay: int, region: str = "USA") -> list[str]:
        """Record the limits BRAIN reported so offline checks use BRAIN's real thresholds."""
        learned = []
        for c in checks or []:
            name, limit = c.get("name"), c.get("limit")
            if name not in BRAIN_CHECK_MAP or limit is None:
                continue
            try:
                limit = float(limit)
            except (TypeError, ValueError):
                continue
            path, by_delay = BRAIN_CHECK_MAP[name]
            full = f"{path}.delay{delay}" if by_delay else path
            bucket = self.learned.setdefault(region, {})
            if _get(bucket, full) != limit:
                _set(bucket, full, limit)
                learned.append(f"{region} {full} = {limit}")
        if learned and self.learned_path:
            try:
                self.learned_path.write_text(json.dumps(self.learned, indent=2), encoding="utf-8")
            except OSError:
                pass
        return learned


@dataclass
class Check:
    name: str          # BRAIN-style id, e.g. LOW_SHARPE
    label: str
    metric: str
    value: float | None
    limit: str         # human-readable threshold, e.g. ">= 1.25"
    result: str        # PASS | FAIL | WARN | PENDING
    why: str
    fixes: list[str]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def pct(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 100:.1f}%"


def evaluate(metrics: dict[str, Any], settings: SimSettings, rules: Rules, *,
             correlation: dict | None = None) -> list[Check]:
    d = int(settings.delay)
    region = settings.region

    def rv(path: str, delay: int | None = None):
        return rules.value(path, delay, region)

    checks: list[Check] = []
    sharpe = float(metrics.get("sharpe", 0.0))
    ret = float(metrics.get("returns", 0.0))
    to = float(metrics.get("turnover", 0.0))
    fit = float(metrics.get("fitness", 0.0))

    # --- Sharpe ---------------------------------------------------------
    s_min, s_src = rv("sharpe", d)
    ok = sharpe >= s_min
    fixes = []
    if sharpe < 0:
        why = (f"Sharpe is {sharpe:.2f}: the alpha is systematically wrong-way. Flipping its sign "
               f"would give about +{-sharpe:.2f}.")
        fixes.append("Flip the sign (multiply the expression by -1).")
    elif ok:
        why = f"Sharpe {sharpe:.2f} clears the {s_min} floor for delay-{d} alphas."
    else:
        why = (f"Sharpe {sharpe:.2f} is below the {s_min} floor for delay-{d} alphas"
               + (" (delay-0 is stricter than delay-1)." if d == 0 else "."))
        fixes += ["Try a finer neutralization (INDUSTRY or SUBINDUSTRY) to strip common-factor noise.",
                  "On BRAIN, try a risk-factor neutralization (FAST, SLOW, CROWDING, STATISTICAL): Learn2Quant "
                  "Lesson 9 shows it typically raises Sharpe and lowers drawdown for slightly more turnover.",
                  "Rank or winsorize the signal so a few extreme values don't dominate.",
                  "Combine with a complementary, low-correlation signal (below 0.5, per Lesson 8)."]
    checks.append(Check("LOW_SHARPE", "Sharpe", "sharpe", sharpe, f">= {s_min}", "PASS" if ok else "FAIL",
                        why, fixes, s_src))

    # --- Fitness ----------------------------------------------------------
    f_min, f_src = rv("fitness", d)
    ok = fit >= f_min
    eff_to = max(to, 0.125)
    why = (f"Fitness = Sharpe x sqrt(|Returns| / max(Turnover, 12.5%)) = {sharpe:.2f} x "
           f"sqrt({pct(abs(ret))} / {pct(eff_to)}) = {fit:.2f}; floor is {f_min} for delay-{d}. "
           "(Every IS summary shown in the Learn2Quant lessons matches this formula.)")
    fixes = []
    if not ok:
        if to > 0.125:
            at_floor = sharpe * math.sqrt(abs(ret) / 0.125)
            why += (f" Turnover ({pct(to)}) is the drag: at the same Sharpe and returns, turnover of 12.5% "
                    f"would give fitness about {at_floor:.2f}. Below 12.5% turnover the formula gives no "
                    "further credit (the decay floor).")
            fixes += ["Increase decay (e.g. +2 to +6 days) or smooth with ts_mean/ts_decay_linear.",
                      "Lengthen the lookback windows so the signal changes more slowly.",
                      "For reversion ideas, trade only when conditions favour reversion, e.g. higher-volatility "
                      "regimes (Learn2Quant Lesson 4's suggested improvement)."]
        else:
            why += " Turnover is already at or below 12.5%, so fitness can only rise through Sharpe or returns."
            fixes += ["Raise returns: loosen neutralization (e.g. INDUSTRY -> SECTOR) if Sharpe holds up.",
                      "Improve Sharpe (see the Sharpe fixes)."]
    checks.append(Check("LOW_FITNESS", "Fitness", "fitness", fit, f">= {f_min}", "PASS" if ok else "FAIL",
                        why, fixes, f_src))

    # --- Turnover band --------------------------------------------------------
    lo, lo_src = rv("turnover.min")
    hi, hi_src = rv("turnover.max")
    if to < lo:
        checks.append(Check("LOW_TURNOVER", "Turnover", "turnover", to, f"{pct(lo)} to {pct(hi)}", "FAIL",
                            f"Turnover {pct(to)} is below the {pct(lo)} minimum: the portfolio barely trades.",
                            ["Reduce decay.", "Shorten lookback windows or use a faster-updating field."], lo_src))
    elif to > hi:
        checks.append(Check("HIGH_TURNOVER", "Turnover", "turnover", to, f"{pct(lo)} to {pct(hi)}", "FAIL",
                            f"Turnover {pct(to)} is above the {pct(hi)} ceiling.",
                            ["Add decay (try 4-10; the lessons use 7-10).", "Wrap in ts_decay_linear(x, 5) or ts_mean(x, 5).",
                             "Use trade_when() or hump() so positions only change on meaningful moves.",
                             "For reversion ideas, update positions only in higher-volatility regimes (Lesson 4)."],
                            hi_src))
    else:
        bucket = "low (<5%)" if to < 0.05 else ("medium (5-15%)" if to <= 0.15 else "high (>15%)")
        checks.append(Check("TURNOVER", "Turnover", "turnover", to, f"{pct(lo)} to {pct(hi)}", "PASS",
                            f"Turnover {pct(to)} is inside the {pct(lo)}-{pct(hi)} band; Learn2Quant Lesson 8 "
                            f"would call it {bucket} turnover.", [], hi_src))

    # --- Weight concentration -------------------------------------------------------
    wmax, w_src = rv("weight.max")
    mw = float(metrics.get("max_weight", 0.0))
    ok = mw <= wmax + 1e-9
    fixes = [] if ok else [f"Lower truncation to {min(settings.truncation, wmax) / 2:.2f} or less.",
                           "Rank or winsorize the signal.", "Backfill sparse data (ts_backfill) so more stocks hold positions."]
    why = (f"The largest single-stock weight was {pct(mw)} of the book (limit {pct(wmax)}; truncation setting "
           f"{settings.truncation:.2f}).")
    cov = metrics.get("coverage")
    if cov is not None and cov < 0.5:
        why += f" Only {pct(cov)} of the universe holds a position on an average day, which concentrates weight."
    checks.append(Check("CONCENTRATED_WEIGHT", "Max stock weight", "max_weight", mw, f"<= {pct(wmax)}",
                        "PASS" if ok else "FAIL", why, fixes, w_src))

    # --- Sub-universe robustness ------------------------------------------------------
    sub = SUB_UNIVERSE.get(settings.universe)
    sub_sh = metrics.get("sub_universe_sharpe")
    factor, fac_src = rv("sub_universe.factor")
    if sub and sub_sh is not None:
        ratio = math.sqrt(UNIVERSE_SIZE[sub] / UNIVERSE_SIZE[settings.universe])
        need = factor * ratio * sharpe
        ok = sub_sh >= need
        why = (f"In {sub}, Sharpe is {sub_sh:.2f}; it needs >= {factor} x sqrt({UNIVERSE_SIZE[sub]}/"
               f"{UNIVERSE_SIZE[settings.universe]}) x {sharpe:.2f} = {need:.2f}. The bar scales with "
               "sub-universe size. (Community-reported formula, not confirmed by WorldQuant; BRAIN's own "
               "limit is used once you verify on BRAIN.)")
        fixes = [] if ok else ["The signal leans on small caps: neutralize by size bucket, e.g. "
                               "group_neutralize(x, bucket(rank(cap), range=\"0.1,1,0.1\")).",
                               f"Simulate directly in {sub} or a larger-cap universe."]
        checks.append(Check("LOW_SUB_UNIVERSE_SHARPE", "Sub-universe Sharpe", "sub_universe_sharpe", sub_sh,
                            f">= {need:.2f}", "PASS" if ok else "FAIL", why, fixes, fac_src))

    # --- Self-correlation ---------------------------------------------------------------
    cmax, c_src = rv("self_correlation.max")
    improve, _ = rv("self_correlation.sharpe_improvement")
    if correlation and correlation.get("max") is not None:
        c = float(correlation["max"])
        other = correlation.get("with_sharpe")
        label = correlation.get("with_name") or correlation.get("with_id") or "an existing alpha"
        src_note = "verified on BRAIN" if correlation.get("source") == "brain" else "simulated"
        if c < cmax:
            combine, _ = rv("diversity.combine_max_corr")
            note = (" Below 0.5, so it also diversifies your pool (Learn2Quant Lesson 8)." if c < combine
                    else " It passes, but above 0.5 it adds little diversity to your pool (Lesson 8).")
            res, why, fixes = "PASS", f"Highest correlation of daily PnL is {c:.2f} with {label} ({src_note}).{note}", []
        elif other is not None and sharpe >= float(other) * (1 + improve):
            res = "PASS"
            why = (f"Correlation {c:.2f} with {label} is above {cmax}, but Sharpe {sharpe:.2f} beats its "
                   f"{float(other):.2f} by at least {improve:.0%}, which BRAIN accepts ({src_note}).")
            fixes = []
        else:
            res = "FAIL"
            need = f" or Sharpe >= {float(other) * (1 + improve):.2f}" if other is not None else ""
            why = (f"Self-correlation {c:.2f} vs {label} exceeds {cmax}{need} ({src_note}; measured on daily "
                   "PnL changes, not cumulative PnL).")
            fixes = ["Change the horizon (lookback x3 or /3) to decorrelate.",
                     "Swap to a different field in the same idea, or a different dataset.",
                     "Change neutralization or universe.",
                     "Orthogonalize: regression_neut(x, <the correlated signal>)."]
        checks.append(Check("SELF_CORRELATION", "Self-correlation", "self_correlation", c, f"< {cmax}",
                            res, why, fixes, c_src))
    else:
        checks.append(Check("SELF_CORRELATION", "Self-correlation", "self_correlation", None, f"< {cmax}",
                            "PENDING", "Load your active alphas (Correlation tab) or paste some to test this.",
                            [], c_src))

    # --- Returns / drawdown (Learn2Quant Lesson 3) and optional drawdown cap ---------------------------
    dd = float(metrics.get("drawdown", 0.0))
    rmin, r_src = rv("returns_drawdown.min")
    r_enf, _ = rv("returns_drawdown.enforced")
    ratio = (ret / dd) if dd > 1e-9 else None
    if ratio is not None:
        ok = ratio >= rmin
        why = (f"Returns {pct(ret)} / drawdown {pct(dd)} = {ratio:.2f}. Learn2Quant: target a returns-to-drawdown "
               f"ratio above {rmin:g}; higher is better. Drawdown here is the largest loss within any single year.")
        fixes = [] if ok else ["Use a finer or risk-factor neutralization (Lesson 9: lowers drawdown).",
                               "Scale by volatility: x / ts_std_dev(returns, 20).", "Lower truncation."]
        checks.append(Check("RETURNS_DRAWDOWN", "Returns / drawdown", "returns_drawdown", ratio, f">= {rmin:g}",
                            "PASS" if ok else ("FAIL" if r_enf else "WARN"), why, fixes, r_src))
    dmax, d_src = rv("drawdown.max")
    if dmax is not None:
        enforced, _ = rv("drawdown.enforced")
        ok = dd <= float(dmax)
        res = "PASS" if ok else ("FAIL" if enforced else "WARN")
        why = f"Drawdown {pct(dd)} of half the book (your cap {pct(float(dmax))}{'' if enforced else ', advisory'})."
        fixes = [] if ok else ["Use a finer neutralization.", "Scale by volatility: x / ts_std_dev(returns, 20).",
                               "Lower truncation."]
        checks.append(Check("DRAWDOWN", "Drawdown", "drawdown", dd, f"<= {pct(float(dmax))}", res, why, fixes, d_src))

    # --- Recent (2-year) Sharpe ----------------------------------------------------------
    s2 = metrics.get("sharpe_2y")
    if s2 is not None:
        need2, src2 = rv("two_year_sharpe", d)
        enforced2, _ = rv("two_year_sharpe.enforced")
        ok = s2 >= need2
        res = "PASS" if ok else ("FAIL" if enforced2 else "WARN")
        checks.append(Check("LOW_2Y_SHARPE", "Last-2-years Sharpe", "sharpe_2y", float(s2), f">= {need2}", res,
                            f"Sharpe over the most recent two years is {float(s2):.2f}. BRAIN runs this as its "
                            "LOW_2Y_SHARPE check: a whole-period pass with a faded recent signal still comes back "
                            "'Needs Improvement'." + ("" if ok else " The signal may be fading."), [] if ok else
                            ["Check the yearly table for when it stopped working.",
                             "Prefer fixes that improve the recent years, not just the average."], src2))

    # --- Held-out test period (BRAIN's TRAIN/TEST split) ------------------------------------------
    ts = metrics.get("test_sharpe")
    if ts is not None:
        need_t, src_t = rv("sharpe", d)
        ok = float(ts) >= need_t
        checks.append(Check("LOW_TEST_SHARPE", "Held-out test Sharpe", "test_sharpe", float(ts), f">= {need_t}",
                            "PASS" if ok else "WARN",
                            f"Sharpe over the held-out test months is {float(ts):.2f} (train "
                            f"{metrics.get('train_sharpe', 0.0):.2f}). BRAIN's verdict covers the whole period "
                            "including the test year, so a strong TRAIN tab with a weak TEST tab still fails.",
                            [] if ok else ["Distrust fixes that only improve the early years.",
                                           "Verify on BRAIN and check its TEST tab before spending more time."],
                            src_t))

    # --- Your risk controls (optional, stricter than BRAIN) -----------------------------------------
    risk = rules.risk()
    if risk.get("max_turnover"):
        cap = float(risk["max_turnover"])
        ok = to <= cap
        checks.append(Check("RISK_TURNOVER", "Your turnover ceiling", "turnover", to, f"<= {pct(cap)}",
                            "PASS" if ok else "FAIL", f"Turnover {pct(to)} vs your ceiling of {pct(cap)}.",
                            [] if ok else ["Add decay or smoothing (see the turnover fixes)."], "your risk controls"))
    if risk.get("max_weight"):
        cap = float(risk["max_weight"])
        mw = float(metrics.get("max_weight", 0.0))
        ok = mw <= cap + 1e-9
        checks.append(Check("RISK_WEIGHT", "Your position cap", "max_weight", mw, f"<= {pct(cap)}",
                            "PASS" if ok else "FAIL", f"Largest position {pct(mw)} vs your cap of {pct(cap)}.",
                            [] if ok else [f"Set truncation to {cap} or lower."], "your risk controls"))
    if risk.get("max_sector_net") and metrics.get("max_sector_net") is not None:
        cap = float(risk["max_sector_net"])
        sn = float(metrics["max_sector_net"])
        ok = sn <= cap
        checks.append(Check("RISK_SECTOR", "Your sector exposure limit", "max_sector_net", sn, f"<= {pct(cap)}",
                            "PASS" if ok else "FAIL", f"Largest average net sector exposure {pct(sn)} vs your limit "
                            f"of {pct(cap)}.", [] if ok else ["Use SECTOR, INDUSTRY or SUBINDUSTRY neutralization."],
                            "your risk controls"))

    checks.append(Check("PROD_CORRELATION", "Production correlation", "prod_correlation", None, "< 0.7",
                        "PENDING", "Correlation with all alphas on the platform can only be measured by BRAIN. "
                        "Run a live simulation, then use 'Check on BRAIN'.", [], "BRAIN only"))
    return checks


def brain_outlook(metrics: dict, checks: list[dict], rules: "Rules", settings: SimSettings,
                  calibrated: dict | None = None) -> dict[str, Any]:
    """How an offline pass is likely to land on BRAIN.

    Levels: fail (offline checks already fail) / likely-fail (calibration puts the BRAIN equivalent
    below a floor) / borderline (clears floors but inside the optimism margin, or only thanks to
    metrics BRAIN judges differently) / comfortable. `checks` are check dicts (Check.to_dict()).
    """
    d, region = int(settings.delay), settings.region
    fails = [c.get("label") or c.get("name") for c in checks if c.get("result") == "FAIL"]
    pending = [c.get("label") or c.get("name") for c in checks if c.get("result") == "PENDING"]
    cal_n = max([int(v.get("n", 0)) for v in (calibrated or {}).values()], default=0)
    out: dict[str, Any] = {"pending": pending, "calibration_n": cal_n}
    if fails:
        return {**out, "level": "fail", "reasons": [f"Fails offline: {', '.join(fails)}."]}

    margin, _ = rules.value("outlook.margin", region=region)
    level, reasons = "comfortable", []
    for metric, label in (("sharpe", "Sharpe"), ("fitness", "Fitness")):
        floor, _ = rules.value(metric, d, region)
        raw = float(metrics.get(metric) or 0.0)
        calv = (calibrated or {}).get(metric, {}).get("value")
        if calv is not None:
            if calv < floor:
                level = "likely-fail"
                reasons.append(f"{label} is {raw:.2f} offline, but your past reconciliations put the BRAIN "
                               f"equivalent near {calv:.2f}, below the {floor:g} floor.")
            elif calv < floor * margin:
                level = "borderline" if level != "likely-fail" else level
                reasons.append(f"{label}: calibrated BRAIN estimate {calv:.2f} clears the {floor:g} floor with "
                               "little room to spare.")
        elif raw < floor * margin:
            level = "borderline" if level == "comfortable" else level
            reasons.append(f"{label} {raw:.2f} is within {int(round((margin - 1) * 100))}% of the {floor:g} floor. "
                           "Offline numbers run optimistic, so on BRAIN this often lands as 'Needs Improvement'.")
    warns = [c for c in checks if c.get("result") == "WARN" and c.get("name") in
             ("LOW_TEST_SHARPE", "RETURNS_DRAWDOWN")]
    for c in warns:
        level = "borderline" if level == "comfortable" else level
        reasons.append(f"{c.get('label')}: {c.get('value'):.2f} vs {c.get('limit')} (advisory, but it drags "
                       "BRAIN's whole-period verdict).") if isinstance(c.get("value"), (int, float)) else None
    to = float(metrics.get("turnover") or 0.0)
    tmax, _ = rules.value("turnover.max", region=region)
    if 0 < tmax <= to / 0.85:
        reasons.append(f"Turnover {to:.0%} is close to the {tmax:.0%} ceiling; BRAIN's measurement can land above it.")
    if pending:
        reasons.append("Checks only BRAIN can run (" + ", ".join(pending) + ") can still fail this alpha: they are "
                       "a common reason a clean offline pass comes back 'Needs Improvement'.")
    return {**out, "level": level, "reasons": reasons, "margin": margin}


def summarize(checks: list[Check]) -> dict[str, Any]:
    hard = [c for c in checks if c.result == "FAIL"]
    pending = [c for c in checks if c.result == "PENDING"]
    return {"passed": not hard, "failures": [c.name for c in hard], "pending": [c.name for c in pending],
            "headline": ("All offline checks pass" + (f"; {len(pending)} need BRAIN" if pending else ""))
            if not hard else f"Fails {len(hard)} check{'s' if len(hard) > 1 else ''}: " +
            ", ".join(c.label for c in hard)}


def brain_checks_view(alpha: dict) -> list[dict]:
    """Normalise the checks block of a BRAIN alpha for side-by-side display."""
    out = []
    for c in ((alpha.get("is") or {}).get("checks") or []):
        out.append({"name": c.get("name"), "result": c.get("result"), "limit": c.get("limit"),
                    "value": c.get("value"), "source": "verified on BRAIN"})
    return out
