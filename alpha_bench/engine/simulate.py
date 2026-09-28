"""Offline BRAIN-style simulation: alpha matrix -> positions -> PnL -> metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..expr.ast import Node, to_str
from ..expr.parser import parse
from ..settings_model import SUB_UNIVERSE, UNIVERSE_SIZE, SimSettings
from .evaluator import LRU, EvalError, Evaluator, FieldCatalog, GroupIndex, _cs_mean, decay_linear
from .market import Market

BOOKSIZE = 20_000_000.0
TRADING_DAYS = 252

FIDELITY_NOTES = [
    "Data: offline results run on a synthetic market (stylised anomalies, realistic factor structure). "
    "Turnover, weight concentration, coverage and settings effects transfer well to BRAIN; absolute "
    "Sharpe and returns do not. Use offline numbers to rank and screen, then verify on BRAIN.",
    "Order of operations: expression -> pasteurize to universe -> linear decay -> neutralization -> "
    "scale to book -> truncation (iteratively capped and re-scaled). BRAIN's internal order may differ slightly.",
    "Delay: delay-1 alphas see data through the previous close and hold positions close-to-close; delay-0 "
    "alphas see the same day's close (slightly optimistic versus BRAIN's intraday delay-0 snapshot).",
    "PnL: book size $20M, no transaction costs (as in BRAIN's IS stats). Returns = annualised PnL / "
    "($20M / 2). Drawdown is the largest peak-to-trough fall of cumulative PnL as a % of $10M. Margin = "
    "PnL per dollar traded, in basis points.",
    "Neutralization: MARKET/SECTOR/INDUSTRY/SUBINDUSTRY subtract the group mean. BRAIN's risk-model "
    "neutralizations (STATISTICAL, CROWDING, FAST, SLOW...) are approximated by SUBINDUSTRY.",
    "Sub-universe test: the alpha is re-weighted inside the next smaller universe using the same signal, "
    "rather than fully re-simulated.",
    "Universe scale: synthetic universes hold fewer names than BRAIN's (e.g. 600 vs 3000), so the truncation "
    "cap is scaled by the size ratio and max weight / long and short counts are reported in BRAIN-equivalent "
    "terms. A 1% cap on TOP3000 therefore behaves like BRAIN's 1% cap.",
    "Drawdown follows Learn2Quant's definition: the largest loss incurred during any single year of the "
    "backtest (as a % of half the book). The whole-period peak-to-trough is shown alongside.",
    "Test period: with a test period set (Learn2Quant uses 1 year), the last months are also reported "
    "separately as train vs test; the headline numbers cover the whole in-sample window.",
    "Small universes: synthetic TOP500/TOP200 hold only about 100/40 names, so INDUSTRY or SUBINDUSTRY "
    "neutralization there leaves very few stocks per group and results are much noisier than on BRAIN "
    "(e.g. Learn2Quant's TOP200 buzz alpha). Prefer TOP3000/TOP1000 for offline screening.",
    "Fields not in the synthetic market are replaced by deterministic proxies shaped by their dataset "
    "category; results that use proxies are flagged as lower-fidelity.",
]


@dataclass
class SimResult:
    expression: str
    settings: SimSettings
    metrics: dict[str, float]
    yearly: list[dict[str, Any]]
    series: dict[str, list]
    exposures: dict[str, dict[str, float]]
    daily_pnl: np.ndarray
    dates: list[str]
    proxied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, include_series: bool = True) -> dict[str, Any]:
        out = {"expression": self.expression, "settings": self.settings.to_dict(), "metrics": self.metrics,
               "yearly": self.yearly, "exposures": self.exposures, "proxied": self.proxied,
               "warnings": self.warnings, "source": "offline"}
        if include_series:
            out["series"] = self.series
        return out


def truncate(w: np.ndarray, cap: float, iters: int = 50) -> np.ndarray:
    """Cap |w| at `cap` (fraction of book) and re-scale the rest so gross stays 1.

    Rows with too few names to satisfy the cap (n * cap < 1) end up equal-weighted
    at the cap, with gross below 1, which is what a hard per-stock limit implies.
    """
    if not (0 < cap < 1):
        return w
    rows = np.where((np.abs(w) > cap + 1e-9).any(axis=1))[0]
    if len(rows) == 0:
        return w
    w = w.copy()
    sub = w[rows]
    live = np.abs(sub) > 0
    infeasible = live.sum(axis=1) * cap <= 1 + 1e-9
    sub[infeasible] = np.sign(sub[infeasible]) * cap * live[infeasible]
    todo = ~infeasible
    for _ in range(iters):
        if not todo.any():
            break
        cur = sub[todo]
        absw = np.abs(cur)
        at_cap = absw >= cap - 1e-12
        fixed = at_cap.sum(axis=1) * cap
        free = np.where(at_cap, 0.0, absw).sum(axis=1)
        with np.errstate(all="ignore"):
            scale = np.where(free > 0, np.clip((1 - fixed) / free, 0, None), 0.0)
        cur = np.where(at_cap, np.sign(cur) * cap, cur * scale[:, None])
        sub[todo] = cur
        still = (np.abs(cur) > cap + 1e-9).any(axis=1)
        idx = np.where(todo)[0]
        todo[idx[~still]] = False
    w[rows] = sub
    return w


def decayed(alpha: np.ndarray, settings: SimSettings, umask: np.ndarray) -> np.ndarray:
    a = np.where(umask, alpha, np.nan)
    if settings.decay > 1:
        a = np.where(umask, decay_linear(a, settings.decay), np.nan)
    return a


def construct_weights(alpha: np.ndarray, settings: SimSettings, umask: np.ndarray,
                      groups: dict[str, np.ndarray], group_cache: dict | None = None,
                      pre_decayed: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    """Alpha values -> daily weights as a fraction of book size (gross = 1).

    `scale` = nominal universe size / synthetic names held; the truncation cap is widened by it so
    a cap means the same thing relative to the average position as it does on BRAIN.
    """
    a = decayed(alpha, settings, umask) if pre_decayed is None else np.where(umask, pre_decayed, np.nan)
    neut = settings.neutralization.upper()
    if neut == "MARKET":
        a = a - _cs_mean(a)
    elif neut != "NONE":
        gname = neut.lower() if neut.lower() in ("sector", "industry", "subindustry") else "subindustry"
        gi = None if group_cache is None else group_cache.get(gname)
        if gi is None:
            gi = GroupIndex(np.asarray(groups[gname], dtype=float))
            if group_cache is not None:
                group_cache[gname] = gi
        a = a - gi.mean(a)
    a = np.where(np.isnan(a), 0.0, a)
    gross = np.abs(a).sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        w = np.where(gross > 0, a / np.where(gross > 0, gross, 1), 0.0)
    return truncate(w, min(0.99, settings.truncation * max(1.0, scale)))


def turnover_bucket(turnover: float) -> str:
    """Learn2Quant Lesson 8's holding-period buckets."""
    return "low (<5%)" if turnover < 0.05 else ("medium (5-15%)" if turnover <= 0.15 else "high (>15%)")


def _sharpe(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    sd = p.std(ddof=1)
    return float(p.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def fitness(sharpe: float, returns: float, turnover: float) -> float:
    return float(sharpe * math.sqrt(abs(returns) / max(turnover, 0.125)))


def yearly_max_drawdown(pnl: np.ndarray, years: np.ndarray | None) -> float:
    """Largest loss incurred during any single year (Learn2Quant Lesson 3), as a fraction of half the book."""
    if years is None or not len(pnl):
        return 0.0
    worst = 0.0
    for y in np.unique(years):
        cum = np.cumsum(pnl[years == y])
        dd = (np.maximum.accumulate(np.maximum(cum, 0)) - cum).max() if len(cum) else 0.0
        worst = max(worst, float(dd))
    return worst / (BOOKSIZE / 2)


def compute_metrics(w: np.ndarray, rets_next: np.ndarray, start: int, end: int,
                    years: np.ndarray | None = None) -> dict[str, Any]:
    """Metrics for positions w[t] earning rets_next[t] (the t -> t+1 return), t in [start, end)."""
    pnl = (w[start:end] * rets_next[start:end]).sum(axis=1) * BOOKSIZE
    prev = w[start - 1:end - 1] if start > 0 else np.vstack([np.zeros((1, w.shape[1])), w[start:end - 1]])
    traded = np.abs(w[start:end] - prev).sum(axis=1)
    half = BOOKSIZE / 2
    cum = np.cumsum(pnl)
    dd_series = (np.maximum.accumulate(np.maximum(cum, 0)) - cum) / half
    sharpe = _sharpe(pnl)
    ret = float(pnl.mean() * TRADING_DAYS / half) if len(pnl) else 0.0
    turnover = float(traded.mean()) if len(traded) else 0.0
    total_traded = traded.sum() * BOOKSIZE
    full_dd = float(dd_series.max()) if len(dd_series) else 0.0
    return {
        "pnl": pnl, "traded": traded, "cum": cum, "dd_series": dd_series,
        "sharpe": sharpe, "returns": ret, "turnover": turnover,
        "fitness": fitness(sharpe, ret, turnover),
        "drawdown": yearly_max_drawdown(pnl, years) if years is not None else full_dd,
        "drawdown_full": full_dd,
        "margin": float(pnl.sum() / total_traded * 1e4) if total_traded > 0 else 0.0,
        "pnl_total": float(pnl.sum()),
        "long_count": float((w[start:end] > 0).sum(axis=1).mean()),
        "short_count": float((w[start:end] < 0).sum(axis=1).mean()),
        "max_weight": float(np.abs(w[start:end]).max()) if end > start else 0.0,
    }


class OfflineEngine:
    def __init__(self, market: Market, catalog: FieldCatalog | None = None, cache_items: int = 32):
        self.market = market
        self.catalog = catalog or FieldCatalog()
        self.cache = LRU(cache_items)
        self._group_cache: dict = {}
        r = np.nan_to_num(np.asarray(market.fields["returns"], dtype=np.float64))
        self.rets_next = np.vstack([r[1:], np.zeros((1, r.shape[1]))])
        self.dates = [str(d) for d in market.dates]

    def alpha_matrix(self, node: Node, settings: SimSettings) -> tuple[np.ndarray, set[str]]:
        ev = Evaluator(self.market, delay=settings.delay, universe=settings.universe,
                       pasteurize=settings.pasteurization.upper() == "ON", catalog=self.catalog, cache=self.cache)
        alpha = ev.evaluate(node)
        return alpha, ev.proxied

    def simulate(self, expression: str | Node, settings: SimSettings, *, sub_universe: bool = True,
                 alpha: np.ndarray | None = None, proxied: set[str] | None = None,
                 quick: bool = False) -> SimResult:
        """Full offline simulation. quick=True skips the yearly table, chart series and exposures
        (used when screening many candidates)."""
        node = parse(expression) if isinstance(expression, str) else expression
        text = expression if isinstance(expression, str) else to_str(node)
        if alpha is None:
            alpha, proxied = self.alpha_matrix(node, settings)
        m = self.market
        umask = m.universe.get(settings.universe, m.universe["TOP3000"])
        T = alpha.shape[0]
        start, end = m.warmup, T - 1
        scale = self._scale(settings.universe, umask, start, end)
        dec = decayed(alpha, settings, umask)
        w = construct_weights(alpha, settings, umask, m.groups, self._group_cache, pre_decayed=dec, scale=scale)
        years = np.array([d[:4] for d in self.dates[start + 1:end + 1]])
        stats = compute_metrics(w, self.rets_next, start, end, years=years)

        warn: list[str] = []
        live = (w[start:end] != 0).sum(axis=1)
        members = umask[start:end].sum(axis=1)
        coverage = float(np.mean(live / np.maximum(members, 1)))
        if coverage < 0.5:
            warn.append(f"Only {coverage:.0%} of the universe gets a position on an average day; sparse "
                        "signals concentrate weight. Consider ts_backfill() or a denser field.")
        if (live == 0).mean() > 0.05:
            warn.append("The alpha has no positions on more than 5% of days.")

        metrics = {k: stats[k] for k in ("sharpe", "returns", "turnover", "fitness", "drawdown", "drawdown_full",
                                         "margin", "pnl_total", "long_count", "short_count", "max_weight")}
        # BRAIN-equivalent position stats (synthetic universes are smaller than BRAIN's).
        metrics["max_weight"] = stats["max_weight"] / scale
        metrics["long_count"] = stats["long_count"] * scale
        metrics["short_count"] = stats["short_count"] * scale
        metrics["universe_scale"] = scale
        metrics["returns_drawdown"] = (stats["returns"] / stats["drawdown"]) if stats["drawdown"] > 1e-9 else None
        metrics["turnover_bucket"] = turnover_bucket(stats["turnover"])
        metrics["coverage"] = coverage
        months = settings.test_period_months()
        test_days = int(round(months * TRADING_DAYS / 12))
        if 0 < test_days < len(stats["pnl"]) - TRADING_DAYS:
            train, test = stats["pnl"][:-test_days], stats["pnl"][-test_days:]
            metrics["train_sharpe"] = _sharpe(train)
            metrics["test_sharpe"] = _sharpe(test)
            metrics["test_returns"] = float(test.mean() * TRADING_DAYS / (BOOKSIZE / 2))
            metrics["test_months"] = months
        sec = self._sector_index()
        net = np.bincount(sec, weights=w[start:end].sum(axis=0), minlength=sec.max() + 1) / max(1, end - start)
        metrics["max_sector_net"] = float(np.abs(net).max())
        tail = stats["pnl"][-2 * TRADING_DAYS:]
        metrics["sharpe_2y"] = _sharpe(tail)

        if sub_universe and settings.universe in SUB_UNIVERSE:
            sub = SUB_UNIVERSE[settings.universe]
            sub_mask = m.universe.get(sub)
            if sub_mask is not None:
                ws = construct_weights(alpha, settings, sub_mask, m.groups, self._group_cache, pre_decayed=dec,
                                       scale=self._scale(sub, sub_mask, start, end))
                metrics["sub_universe"] = sub
                metrics["sub_universe_sharpe"] = _sharpe(compute_metrics(ws, self.rets_next, start, end)["pnl"])

        pnl_dates = self.dates[start + 1:end + 1]
        if quick:
            yearly, series, exposures = [], {}, {}
        else:
            yearly = self._yearly(stats, w, pnl_dates, start, end, scale)
            series = self._series(stats, pnl_dates)
            exposures = self._exposures(w[start:end])
        return SimResult(expression=text, settings=settings, metrics=metrics, yearly=yearly, series=series,
                         exposures=exposures, daily_pnl=stats["pnl"], dates=pnl_dates,
                         proxied=sorted(proxied or []), warnings=warn)

    def _scale(self, universe: str, umask: np.ndarray, start: int, end: int) -> float:
        key = ("scale", universe)
        if key not in self._group_cache:
            held = float(umask[start:end].sum(axis=1).mean()) or 1.0
            self._group_cache[key] = max(1.0, UNIVERSE_SIZE.get(universe, held) / held)
        return self._group_cache[key]

    def _yearly(self, stats, w, pnl_dates, start, end, scale: float = 1.0) -> list[dict]:
        years = np.array([d[:4] for d in pnl_dates])
        out = []
        for y in sorted(set(years)):
            idx = np.where(years == y)[0]
            p, tr = stats["pnl"][idx], stats["traded"][idx]
            cum = np.cumsum(p)
            dd = (np.maximum.accumulate(np.maximum(cum, 0)) - cum).max() / (BOOKSIZE / 2)
            sh, ret, to = _sharpe(p), float(p.mean() * TRADING_DAYS / (BOOKSIZE / 2)), float(tr.mean())
            wy = w[start + idx]
            out.append({"year": y, "sharpe": sh, "returns": ret, "turnover": to, "fitness": fitness(sh, ret, to),
                        "drawdown": float(dd), "margin": float(p.sum() / max(tr.sum() * BOOKSIZE, 1) * 1e4),
                        "long_count": float((wy > 0).sum(1).mean()) * scale,
                        "short_count": float((wy < 0).sum(1).mean()) * scale})
        return out

    def _series(self, stats, pnl_dates) -> dict[str, list]:
        pnl = stats["pnl"]
        win = 126
        roll = np.full(len(pnl), np.nan)
        if len(pnl) > win:
            c1 = np.cumsum(np.r_[0, pnl])
            c2 = np.cumsum(np.r_[0, pnl * pnl])
            s1 = c1[win:] - c1[:-win]
            s2 = c2[win:] - c2[:-win]
            mean = s1 / win
            sd = np.sqrt(np.maximum(s2 / win - mean ** 2, 0))
            with np.errstate(all="ignore"):
                roll[win - 1:] = np.where(sd > 0, mean / sd * math.sqrt(TRADING_DAYS), np.nan)
        step = max(1, len(pnl) // 630)
        sl = slice(None, None, step)

        def clean(a):
            return [None if not np.isfinite(v) else round(float(v), 6) for v in a[sl]]
        return {"dates": pnl_dates[sl], "cum_pnl": clean(stats["cum"]), "drawdown": clean(-stats["dd_series"]),
                "turnover": clean(stats["traded"]), "rolling_sharpe": clean(roll)}

    def _sector_index(self) -> np.ndarray:
        if not hasattr(self, "_sec"):
            self._sec = np.asarray(self.market.groups["sector"][0]).astype(int)
        return self._sec

    def _exposures(self, w: np.ndarray) -> dict[str, dict[str, float]]:
        sector = np.asarray(self.market.groups["sector"][0]).astype(int)
        out = {}
        for s in np.unique(sector):
            cols = sector == s
            out[f"Sector {s + 1}"] = {"net": float(w[:, cols].sum(1).mean()),
                                      "gross": float(np.abs(w[:, cols]).sum(1).mean())}
        return out


__all__ = ["OfflineEngine", "SimResult", "EvalError", "FIDELITY_NOTES", "BOOKSIZE", "fitness",
           "construct_weights", "decayed", "truncate", "compute_metrics"]
