"""Evaluate Fast Expression trees on a Market panel (days x stocks).

Conventions that mirror BRAIN:
* Delay d: every data field is read shifted by d days, so the alpha on day t
  only sees information available at the close of day t-d.
* Pasteurization ON: input values of stocks outside the universe are NaN, so
  cross-sectional operators only see universe members.
* Group fields (market, sector, industry, subindustry, exchange, country) are
  never delayed or pasteurized.
Approximations are listed in `APPROXIMATIONS` and surfaced in the UI.
"""
from __future__ import annotations

import math
import os
import re
import threading
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..expr.ast import BinOp, Call, Ident, Node, Num, Program, Str, Ternary, Unary, inline
from ..expr.operators import GROUP_FIELDS
from .market import FIELD_INFO, Market, proxy_field

NAN = np.nan
# Windows or rows that are entirely NaN are expected (listing gaps, sparse data) and yield NaN.
# warnings.catch_warnings() isn't thread-safe, so these are silenced once at import instead.
for _msg in ("All-NaN slice encountered", "Mean of empty slice", "Degrees of freedom <= 0",
             "invalid value encountered"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

APPROXIMATIONS = {
    "hump": "hump() threshold is scaled by cross-sectional mean |x| x sqrt(N); BRAIN's exact rule may differ.",
    "ts_rank": "ts_rank ties use average rank.",
    "kth_element": "kth_element counts only valid (non-NaN) values.",
    "last_diff_value": "last_diff_value looks back to the most recent change only.",
    "group_backfill": "group_backfill fills NaN with the group mean of the d-day average (no winsorization).",
    "vec": "vector fields are treated as a single daily value offline, so vec_* operators are identity-like.",
}


class EvalError(ValueError):
    pass


# Guess a dataset category from a BRAIN field id when the account catalog isn't loaded.
_CATEGORY_HINTS = [
    (re.compile(r"^(fnd|fn_|fundamental|mdf_|fscore|cf_|bs_|is_)"), "fundamental"),
    (re.compile(r"^(anl|est_|analyst|eps_est|rec_)"), "analyst"),
    (re.compile(r"^(snt|scl|news|nws|sentiment|rp_|ravenpack)"), "sentiment"),
    (re.compile(r"^(soc|social|twitter|stocktwits)"), "socialmedia"),
    (re.compile(r"^(opt|implied|iv_|pcr_|call_|put_)"), "option"),
    (re.compile(r"^(mdl|model|rsk|risk|beta_)"), "model"),
    (re.compile(r"^(shrt|short)"), "shortinterest"),
    (re.compile(r"^(ins|insider)"), "insiders"),
    (re.compile(r"^(pv\d|vol_|volume_)"), "pv"),
]


def guess_category(field_id: str) -> str:
    low = field_id.lower()
    for pattern, cat in _CATEGORY_HINTS:
        if pattern.search(low):
            return cat
    return "other"


class FieldCatalog:
    """Fields known to the account (from /data-fields) plus the offline set."""

    def __init__(self) -> None:
        self.platform: dict[str, dict] = {}

    def load_platform(self, rows: list[dict]) -> None:
        for r in rows:
            fid = r.get("id")
            if not fid:
                continue
            cat = r.get("category")
            cat_name = (cat.get("id") or cat.get("name")) if isinstance(cat, dict) else cat
            ds = r.get("dataset") or {}
            self.platform[fid] = {"id": fid, "category": str(cat_name or guess_category(fid)).lower(),
                                  "type": r.get("type", "MATRIX"), "description": r.get("description", ""),
                                  "dataset": ds.get("id") if isinstance(ds, dict) else ds,
                                  "coverage": r.get("coverage"), "userCount": r.get("userCount"),
                                  "alphaCount": r.get("alphaCount")}

    def category(self, name: str) -> str:
        if name in FIELD_INFO:
            return FIELD_INFO[name][0]
        if name in self.platform:
            return self.platform[name]["category"]
        return guess_category(name)

    def is_group(self, name: str) -> bool:
        return name in GROUP_FIELDS or self.platform.get(name, {}).get("type") == "GROUP"


class LRU:
    """Thread-safe LRU for evaluated sub-expressions (shared by parallel screening workers)."""

    def __init__(self, max_items: int = 32):
        self.max_items = max_items
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


_POOL: ThreadPoolExecutor | None = None


def parallel_map(fn, items, workers: int | None = None) -> list:
    """Map over items with a small shared thread pool (numpy releases the GIL for heavy work)."""
    global _POOL
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=workers or max(1, min(3, (os.cpu_count() or 2) - 1)),
                                   thread_name_prefix="offline-sim")
    return list(_POOL.map(fn, items))


# ======================================================================
# numeric helpers
# ======================================================================

def _clean(x):
    if isinstance(x, np.ndarray) and not np.isfinite(x).all():
        return np.where(np.isfinite(x), x, NAN)
    return x


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return x
    out = np.full_like(x, NAN)
    if d < len(x):
        out[d:] = x[:-d]
    return out


def _rolling(x: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling NaN-aware sum and count over the last d rows (partial windows allowed)."""
    valid = ~np.isnan(x)
    c = np.cumsum(np.where(valid, x, 0.0), axis=0)
    n = np.cumsum(valid, axis=0, dtype=np.float64)
    s, cnt = c.copy(), n.copy()
    if d < len(x):
        s[d:] = c[d:] - c[:-d]
        cnt[d:] = n[d:] - n[:-d]
    return s, cnt


def _rmean(x, d):
    s, n = _rolling(x, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, NAN)


def _rstd(x, d):
    s, n = _rolling(x, d)
    s2, _ = _rolling(x * x, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = np.maximum(s2 / n - mean * mean, 0.0)
        return np.where(n > 1, np.sqrt(var), NAN)


def _sliding(x: np.ndarray, d: int, fn: Callable[[np.ndarray], np.ndarray], chunk: int = 96) -> np.ndarray:
    T, N = x.shape
    padded = np.vstack([np.full((d - 1, N), NAN), x]) if d > 1 else x
    win = sliding_window_view(padded, d, axis=0)  # (T, N, d), oldest first
    out = np.empty((T, N))
    for s in range(0, T, chunk):
        out[s:s + chunk] = fn(win[s:s + chunk])
    return out


def _ffill(x: np.ndarray, limit: int | None = None) -> np.ndarray:
    T, N = x.shape
    valid = ~np.isnan(x)
    idx = np.where(valid, np.arange(T)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = x[np.clip(idx, 0, None), np.arange(N)[None, :]]
    bad = idx < 0
    if limit is not None:
        bad |= (np.arange(T)[:, None] - idx) > limit
    return np.where(bad, NAN, out)


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def _tail_ppf(q):
    return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /         ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)


def norm_ppf(p) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    x = np.full_like(p, NAN)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi) & ~np.isnan(p)
    x[lo] = _tail_ppf(np.sqrt(-2 * np.log(p[lo])))
    x[hi] = -_tail_ppf(np.sqrt(-2 * np.log(1 - p[hi])))
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q /         (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    return x


def cs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank per row in [0, 1], ties averaged, NaN preserved."""
    T, N = x.shape
    valid = ~np.isnan(x)
    filled = np.where(valid, x, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    sv = np.take_along_axis(filled, order, axis=1)
    start = np.ones((T, N), dtype=bool)
    start[:, 1:] = sv[:, 1:] != sv[:, :-1]
    run = np.cumsum(start.ravel()) - 1
    pos = np.broadcast_to(np.arange(N, dtype=float), (T, N)).ravel()
    avg = (np.bincount(run, weights=pos) / np.bincount(run))[run].reshape(T, N)
    ranks = np.empty((T, N))
    np.put_along_axis(ranks, order, avg, axis=1)
    n = valid.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 1, ranks / (n - 1), 0.5)
    return np.where(valid, out, NAN)


def _cs_mean(x):
    valid = ~np.isnan(x)
    n = valid.sum(axis=1, keepdims=True)
    s = np.where(valid, x, 0.0).sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), NAN)


def _cs_std(x):
    m = _cs_mean(x)
    valid = ~np.isnan(x)
    n = valid.sum(axis=1, keepdims=True)
    ss = np.where(valid, (x - m) ** 2, 0.0).sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.where(n > 0, np.sqrt(ss / np.maximum(n, 1)), NAN)


def cs_zscore(x):
    s = _cs_std(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, (x - _cs_mean(x)) / s, NAN)


class GroupIndex:
    """Flattened (day, group) index so group reductions are single bincounts."""

    def __init__(self, g: np.ndarray):
        T, N = g.shape
        invalid = np.isnan(g)
        codes = np.where(invalid, -1, g).astype(np.int64)
        uniq, inv = np.unique(codes, return_inverse=True)
        self.G = len(uniq)
        self.T, self.N = T, N
        self.idx = (np.arange(T)[:, None] * self.G + inv.reshape(T, N)).ravel()
        self.invalid = invalid.ravel()
        self.size = T * self.G

    def reduce(self, x: np.ndarray, weights: np.ndarray | None = None):
        xf = x.ravel()
        ok = ~np.isnan(xf) & ~self.invalid
        w = np.ones_like(xf) if weights is None else np.nan_to_num(weights.ravel())
        if weights is not None:
            ok &= ~np.isnan(weights.ravel())
        s = np.bincount(self.idx[ok], weights=(xf * w)[ok], minlength=self.size)
        c = np.bincount(self.idx[ok], weights=w[ok], minlength=self.size)
        return s, c, ok

    def broadcast(self, per_group: np.ndarray) -> np.ndarray:
        out = per_group[self.idx]
        out[self.invalid] = NAN
        return out.reshape(self.T, self.N)

    def mean(self, x, weights=None):
        s, c, _ = self.reduce(x, weights)
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.broadcast(np.where(c > 0, s / c, NAN))

    def std(self, x):
        s, c, _ = self.reduce(x)
        s2, _, _ = self.reduce(x * x)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
            var = np.maximum(s2 / c - m * m, 0)
            return self.broadcast(np.where(c > 1, np.sqrt(var), NAN))

    def extreme(self, x, fn):
        xf = x.ravel()
        ok = ~np.isnan(xf) & ~self.invalid
        init = -np.inf if fn is np.maximum else np.inf
        acc = np.full(self.size, init)
        fn.at(acc, self.idx[ok], xf[ok])
        acc[~np.isfinite(acc)] = NAN
        return self.broadcast(acc)

    def ordered(self, x):
        """(sorted flat positions, rank within group, count per element) for rank/median.

        One argsort on a composite key (group index + within-day rank scaled into [0, 0.5])
        instead of a two-key lexsort.
        """
        xf = x.ravel()
        valid = ~np.isnan(xf) & ~self.invalid
        r = cs_rank(np.where(valid.reshape(x.shape), x, NAN)).ravel()
        key = self.idx.astype(np.float64) + np.where(valid, np.nan_to_num(r) * 0.5, 0.9)
        order = np.argsort(key, kind="stable")
        sidx = self.idx[order]
        valid_sorted = valid[order]
        first = np.r_[True, sidx[1:] != sidx[:-1]]
        start_pos = np.maximum.accumulate(np.where(first, np.arange(len(sidx)), 0))
        within = np.arange(len(sidx)) - start_pos
        cnt = np.bincount(self.idx[valid], minlength=self.size)
        return order, within, cnt, valid_sorted


# ======================================================================
# evaluator
# ======================================================================

class Evaluator:
    def __init__(self, market: Market, *, delay: int = 1, universe: str = "TOP3000",
                 pasteurize: bool = True, catalog: FieldCatalog | None = None, cache: LRU | None = None):
        self.m = market
        self.delay = int(delay)
        self.universe_name = universe
        self.umask = market.universe.get(universe, market.universe["TOP3000"])
        self.pasteurize = pasteurize
        self.catalog = catalog or FieldCatalog()
        self.cache = cache if cache is not None else LRU()
        self.proxied: set[str] = set()
        self.T, self.N = market.shape
        self._ctx = (self.delay, universe, pasteurize)

    # ------------------------------------------------------------ public
    def evaluate(self, node: Node) -> np.ndarray:
        node = inline(node) if isinstance(node, Program) else node
        out = self.eval(node)
        if not isinstance(out, np.ndarray):
            raise EvalError("The alpha evaluates to a constant; it needs at least one data field.")
        return out

    def eval(self, node: Node):
        if isinstance(node, Num):
            return float(node.value)
        if isinstance(node, Str):
            raise EvalError(f"Unexpected text argument \"{node.value}\".")
        if isinstance(node, Ident):
            return self.ident(node.name)
        key = (node, self._ctx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if isinstance(node, Call):
            out = self.call(node)
        elif isinstance(node, BinOp):
            out = self.binop(node.op, self.eval(node.left), self.eval(node.right))
        elif isinstance(node, Unary):
            v = self.eval(node.operand)
            out = -v if node.op == "-" else _not(v)
        elif isinstance(node, Ternary):
            out = _if_else(self.eval(node.cond), self.eval(node.then), self.eval(node.other))
        else:
            raise EvalError(f"Unsupported syntax: {type(node).__name__}")
        out = _clean(out)
        if isinstance(out, np.ndarray):
            out.setflags(write=False)
            self.cache.put(key, out)
        return out

    def arr(self, v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        return np.full((self.T, self.N), float(v))

    # ------------------------------------------------------------ fields
    def ident(self, name: str):
        if name in self.m.groups:
            return self.m.groups[name]
        key = ("field", name, self._ctx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if name in self.m.fields:
            raw = self.m.fields[name]
        elif name.lower() in ("nan",):
            return NAN
        else:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", name):
                raise EvalError(f"Unknown identifier {name!r}.")
            raw = proxy_field(self.m, name, self.catalog.category(name))
            self.proxied.add(name)
        x = _shift(np.asarray(raw, dtype=np.float64), self.delay)
        if self.pasteurize:
            x = np.where(self.umask, x, NAN)
        x.setflags(write=False)
        self.cache.put(key, x)
        return x

    def group(self, node: Node) -> GroupIndex:
        key = ("gidx", node, self._ctx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        g = self.eval(node)
        gi = GroupIndex(self.arr(g))
        self.cache.put(key, gi)
        return gi

    # ------------------------------------------------------------- calls
    def const(self, node: Node | None, default=None, what: str = "parameter"):
        if node is None:
            return default
        if isinstance(node, Str):
            return node.value
        v = self.eval(node)
        if isinstance(v, np.ndarray):
            raise EvalError(f"The {what} must be a constant number.")
        return v

    def window(self, call: Call, i: int) -> int:
        if i >= len(call.args):
            raise EvalError(f"{call.name}() needs a lookback-days argument.")
        d = self.const(call.args[i], what=f"lookback days of {call.name}()")
        if d is None or (isinstance(d, float) and math.isnan(d)):
            raise EvalError(f"{call.name}() lookback must be a number.")
        d = int(round(d))
        if d < 1:
            raise EvalError(f"{call.name}() lookback must be at least 1 day.")
        return min(d, self.T)

    def call(self, c: Call):
        fn = OPS.get(c.name)
        if fn is None:
            raise EvalError(f"Operator {c.name}() isn't available in the offline engine yet. "
                            "It can still be simulated on BRAIN.")
        try:
            return fn(self, c)
        except EvalError:
            raise
        except IndexError:
            raise EvalError(f"{c.name}() is missing an argument.") from None

    def a(self, c: Call, i: int) -> np.ndarray:
        if i >= len(c.args):
            raise EvalError(f"{c.name}() is missing argument {i + 1}.")
        return self.arr(self.eval(c.args[i]))

    def v(self, c: Call, i: int):
        if i >= len(c.args):
            raise EvalError(f"{c.name}() is missing argument {i + 1}.")
        return self.eval(c.args[i])

    def binop(self, op: str, x, y):
        with np.errstate(all="ignore"):
            if op == "+":
                return x + y
            if op == "-":
                return x - y
            if op == "*":
                return x * y
            if op == "/":
                return _div(x, y)
            if op == "^":
                return _pow(x, y)
            if op == "&&":
                return _logic(x, y, np.logical_and)
            if op == "||":
                return _logic(x, y, np.logical_or)
            return _cmp(x, y, op)


def _div(x, y):
    with np.errstate(all="ignore"):
        if isinstance(y, np.ndarray):
            return np.where(y != 0, x / np.where(y != 0, y, 1), NAN)
        return x / y if y != 0 else (x * NAN)


def _pow(x, y):
    with np.errstate(all="ignore"):
        return np.power(x, y)


def _nanmask(*vals):
    m = None
    for v in vals:
        if isinstance(v, np.ndarray):
            mv = np.isnan(v)
            m = mv if m is None else (m | mv)
        elif isinstance(v, float) and math.isnan(v):
            return True
    return m


def _cmp(x, y, op):
    fn = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal,
          "==": np.equal, "!=": np.not_equal}[op]
    with np.errstate(invalid="ignore"):
        r = fn(x, y).astype(float)
    m = _nanmask(x, y)
    if m is True:
        return r * NAN
    return np.where(m, NAN, r) if m is not None else r


def _logic(x, y, fn):
    r = fn(np.asarray(x) > 0, np.asarray(y) > 0).astype(float)
    m = _nanmask(x, y)
    if m is True:
        return r * NAN
    return np.where(m, NAN, r) if m is not None else r


def _not(x):
    r = (np.asarray(x) <= 0).astype(float)
    return np.where(np.isnan(x), NAN, r) if isinstance(x, np.ndarray) else r


def _if_else(cond, a, b):
    c = np.asarray(cond)
    r = np.where(c > 0, a, b)
    return np.where(np.isnan(c), NAN, r) if isinstance(cond, np.ndarray) else r


# ======================================================================
# operator implementations: fn(evaluator, call) -> array | float
# ======================================================================
OPS: dict[str, Callable[[Evaluator, Call], object]] = {}


def op(*names):
    def deco(fn):
        for n in names:
            OPS[n] = fn
        return fn
    return deco


def _variadic(ev: Evaluator, c: Call, fn):
    vals = [ev.v(c, i) for i in range(len(c.args))]
    if ev.const(c.kw("filter"), False):
        vals = [np.nan_to_num(v) if isinstance(v, np.ndarray) else v for v in vals]
    out = vals[0]
    for v in vals[1:]:
        out = fn(out, v)
    return out


@op("add")
def _add(ev, c):
    return _variadic(ev, c, lambda a, b: a + b)


@op("subtract")
def _sub(ev, c):
    return _variadic(ev, c, lambda a, b: a - b)


@op("multiply")
def _mul(ev, c):
    return _variadic(ev, c, lambda a, b: a * b)


@op("divide")
def _divide(ev, c):
    return _div(ev.v(c, 0), ev.v(c, 1))


@op("power")
def _power(ev, c):
    return _pow(ev.v(c, 0), ev.v(c, 1))


@op("signed_power")
def _spow(ev, c):
    x = ev.v(c, 0)
    with np.errstate(all="ignore"):
        return np.sign(x) * np.power(np.abs(x), ev.v(c, 1))


@op("abs")
def _abs(ev, c):
    return np.abs(ev.v(c, 0))


@op("sqrt")
def _sqrt(ev, c):
    x = ev.v(c, 0)
    with np.errstate(all="ignore"):
        return np.where(np.asarray(x) >= 0, np.sqrt(np.abs(x)), NAN)


@op("log")
def _log(ev, c):
    x = ev.v(c, 0)
    with np.errstate(all="ignore"):
        return np.where(np.asarray(x) > 0, np.log(np.where(np.asarray(x) > 0, x, 1)), NAN)


@op("exp")
def _exp(ev, c):
    with np.errstate(all="ignore"):
        return np.exp(np.clip(ev.v(c, 0), -50, 50))


@op("sign")
def _sign(ev, c):
    return np.sign(ev.v(c, 0))


@op("inverse")
def _inverse(ev, c):
    return _div(1.0, ev.v(c, 0))


@op("reverse")
def _reverse(ev, c):
    return -ev.v(c, 0)


@op("s_log_1p")
def _slog(ev, c):
    x = ev.v(c, 0)
    return np.sign(x) * np.log1p(np.abs(x))


@op("max")
def _max(ev, c):
    return _variadic(ev, c, np.maximum)


@op("min")
def _min(ev, c):
    return _variadic(ev, c, np.minimum)


@op("densify")
def _densify(ev, c):
    x = ev.a(c, 0)
    valid = ~np.isnan(x)
    out = np.full_like(x, NAN)
    _, inv = np.unique(x[valid], return_inverse=True)
    out[valid] = inv
    return out


@op("to_nan")
def _to_nan(ev, c):
    x = ev.a(c, 0)
    value = ev.const(c.kw("value"), 0.0)
    if ev.const(c.kw("reverse"), False):
        return np.where(x == value, x, NAN)
    return np.where(x == value, NAN, x)


@op("if_else")
def _ife(ev, c):
    return _if_else(ev.v(c, 0), ev.v(c, 1), ev.v(c, 2))


@op("is_nan")
def _isnan(ev, c):
    return np.isnan(ev.a(c, 0)).astype(float)


@op("not")
def _notop(ev, c):
    return _not(ev.v(c, 0))


@op("and")
def _and(ev, c):
    return _logic(ev.v(c, 0), ev.v(c, 1), np.logical_and)


@op("or")
def _or(ev, c):
    return _logic(ev.v(c, 0), ev.v(c, 1), np.logical_or)


for _name, _sym in (("less", "<"), ("greater", ">"), ("less_equal", "<="), ("greater_equal", ">="),
                    ("equal", "=="), ("not_equal", "!=")):
    OPS[_name] = (lambda s: (lambda ev, c: _cmp(ev.v(c, 0), ev.v(c, 1), s)))(_sym)


# --- time series ------------------------------------------------------------
@op("ts_delay")
def _ts_delay(ev, c):
    return _shift(ev.a(c, 0), ev.window(c, 1))


@op("ts_delta")
def _ts_delta(ev, c):
    x = ev.a(c, 0)
    return x - _shift(x, ev.window(c, 1))


@op("ts_mean")
def _ts_mean(ev, c):
    return _rmean(ev.a(c, 0), ev.window(c, 1))


@op("ts_sum")
def _ts_sum(ev, c):
    s, n = _rolling(ev.a(c, 0), ev.window(c, 1))
    return np.where(n > 0, s, NAN)


@op("ts_std_dev")
def _ts_std(ev, c):
    return _rstd(ev.a(c, 0), ev.window(c, 1))


@op("ts_zscore")
def _ts_z(ev, c):
    x, d = ev.a(c, 0), ev.window(c, 1)
    sd = _rstd(x, d)
    with np.errstate(all="ignore"):
        return np.where(sd > 0, (x - _rmean(x, d)) / sd, NAN)


@op("ts_ir")
def _ts_ir(ev, c):
    x, d = ev.a(c, 0), ev.window(c, 1)
    sd = _rstd(x, d)
    with np.errstate(all="ignore"):
        return np.where(sd > 0, _rmean(x, d) / sd, NAN)


def _ts_rank_arr(x, d):
    def fn(w):
        last = w[..., -1]
        valid = ~np.isnan(w)
        less = (w < last[..., None]).sum(-1)
        eq = (w == last[..., None]).sum(-1)
        n = valid.sum(-1)
        with np.errstate(all="ignore"):
            r = (less + 0.5 * (eq - 1)) / (n - 1)
        return np.where(np.isnan(last), NAN, np.where(n > 1, r, 0.5))
    return _sliding(x, d, fn)


@op("ts_rank")
def _ts_rank(ev, c):
    return _ts_rank_arr(ev.a(c, 0), ev.window(c, 1)) + float(ev.const(c.kw("constant"), 0.0))


@op("ts_quantile")
def _ts_quantile(ev, c):
    r = _ts_rank_arr(ev.a(c, 0), ev.window(c, 1))
    driver = str(ev.const(c.kw("driver"), "gaussian")).lower()
    u = np.clip(r, 0.005, 0.995)
    if driver == "uniform":
        return 2 * u - 1
    if driver == "cauchy":
        return np.tan(np.pi * (u - 0.5))
    return norm_ppf(u)


@op("ts_min")
def _ts_min(ev, c):
    return _sliding(ev.a(c, 0), ev.window(c, 1), lambda w: np.nanmin(w, axis=-1))


@op("ts_max")
def _ts_max(ev, c):
    return _sliding(ev.a(c, 0), ev.window(c, 1), lambda w: np.nanmax(w, axis=-1))


@op("ts_max_diff")
def _ts_max_diff(ev, c):
    x = ev.a(c, 0)
    return x - _sliding(x, ev.window(c, 1), lambda w: np.nanmax(w, axis=-1))


@op("ts_median")
def _ts_median(ev, c):
    return _sliding(ev.a(c, 0), ev.window(c, 1), lambda w: np.nanmedian(w, axis=-1))


def _argext(ev, c, sign):
    d = ev.window(c, 1)

    def fn(w):
        allnan = np.isnan(w).all(-1)
        filled = np.where(np.isnan(w), -np.inf, sign * w)
        idx = np.argmax(filled[..., ::-1], axis=-1)  # 0 = today; ties -> most recent
        return np.where(allnan, NAN, idx.astype(float))
    return _sliding(ev.a(c, 0), d, fn)


@op("ts_arg_max")
def _ts_argmax(ev, c):
    return _argext(ev, c, 1.0)


@op("ts_arg_min")
def _ts_argmin(ev, c):
    return _argext(ev, c, -1.0)


@op("ts_product")
def _ts_prod(ev, c):
    def fn(w):
        allnan = np.isnan(w).all(-1)
        return np.where(allnan, NAN, np.nanprod(w, axis=-1))
    return _sliding(ev.a(c, 0), ev.window(c, 1), fn)


def _pair_stats(x, y, d):
    m = np.isnan(x) | np.isnan(y)
    xm, ym = np.where(m, NAN, x), np.where(m, NAN, y)
    sx, n = _rolling(xm, d)
    sy, _ = _rolling(ym, d)
    sxy, _ = _rolling(xm * ym, d)
    sxx, _ = _rolling(xm * xm, d)
    syy, _ = _rolling(ym * ym, d)
    with np.errstate(all="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = np.maximum(sxx / n - (sx / n) ** 2, 0)
        vy = np.maximum(syy / n - (sy / n) ** 2, 0)
    return n, sx, sy, cov, vx, vy


@op("ts_corr")
def _ts_corr(ev, c):
    n, _, _, cov, vx, vy = _pair_stats(ev.a(c, 0), ev.a(c, 1), ev.window(c, 2))
    with np.errstate(all="ignore"):
        den = np.sqrt(vx * vy)
        return np.where((n > 2) & (den > 1e-12), cov / np.where(den > 0, den, 1), NAN)


@op("ts_covariance")
def _ts_cov(ev, c):
    n, _, _, cov, _, _ = _pair_stats(ev.a(c, 0), ev.a(c, 1), ev.window(c, 2))
    return np.where(n > 1, cov, NAN)


@op("ts_regression")
def _ts_reg(ev, c):
    y, x, d = ev.a(c, 0), ev.a(c, 1), ev.window(c, 2)
    lag = int(ev.const(c.kw("lag"), 0) or 0)
    if lag:
        x = _shift(x, lag)
    rettype = int(ev.const(c.kw("rettype"), 0) or 0)
    n, sx, sy, cov, vx, vy = _pair_stats(y, x, d)
    with np.errstate(all="ignore"):
        beta = np.where(vx > 1e-12, cov / np.where(vx > 0, vx, 1), NAN)
        alpha = sy / n - beta * sx / n
        fit = alpha + beta * x
        if rettype == 1:
            return alpha
        if rettype == 2:
            return beta
        if rettype == 3:
            return fit
        if rettype == 6:
            return np.where(vx * vy > 0, cov * cov / (vx * vy), NAN)
        return y - fit


def _weighted(x, weights):
    d = len(weights)

    def fn(w):
        valid = ~np.isnan(w)
        num = np.nansum(w * weights, axis=-1)
        den = (valid * weights).sum(-1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), NAN)
    return _sliding(x, d, fn)


def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Linearly decayed average, newest weight d ... oldest weight 1, NaN-aware.

    Uses cumulative sums: sum_{s=t-d+1..t} (s - (t - d)) * x_s, so cost is O(T*N)
    regardless of d.
    """
    if d <= 1:
        return x
    valid = ~np.isnan(x)
    if d <= 16:
        # Short decays: a handful of shifted adds beats cumulative sums.
        xv = np.where(valid, x, 0.0)
        vf = valid.astype(float)
        num = d * xv
        den = d * vf
        for k in range(1, min(d, len(x))):
            wk = d - k
            num[k:] += wk * xv[:-k]
            den[k:] += wk * vf[:-k]
        with np.errstate(all="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1), NAN)
    t = np.arange(len(x), dtype=float)[:, None]

    def wsum(v):
        c1 = np.cumsum(v, axis=0)
        c2 = np.cumsum(v * t, axis=0)
        s1, s2 = c1.copy(), c2.copy()
        s1[d:] -= c1[:-d]
        s2[d:] -= c2[:-d]
        return s2 - (t - d) * s1

    num = wsum(np.where(valid, x, 0.0))
    den = wsum(valid.astype(float))
    with np.errstate(all="ignore"):
        return np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1), NAN)


@op("ts_decay_linear")
def _ts_decay_linear(ev, c):
    return decay_linear(ev.a(c, 0), ev.window(c, 1))


@op("ts_decay_exp_window")
def _ts_decay_exp(ev, c):
    d = ev.window(c, 1)
    f = ev.const(c.kw("factor", c.args[2] if len(c.args) > 2 else None), 0.5)
    if not 0 < f <= 1:
        raise EvalError("ts_decay_exp_window factor must be in (0, 1].")
    return _weighted(ev.a(c, 0), f ** np.arange(d - 1, -1, -1, dtype=float))


@op("ts_av_diff")
def _ts_av_diff(ev, c):
    x = ev.a(c, 0)
    return x - _rmean(x, ev.window(c, 1))


@op("ts_scale")
def _ts_scale(ev, c):
    x, d = ev.a(c, 0), ev.window(c, 1)
    lo = _sliding(x, d, lambda w: np.nanmin(w, axis=-1))
    hi = _sliding(x, d, lambda w: np.nanmax(w, axis=-1))
    with np.errstate(all="ignore"):
        r = np.where(hi > lo, (x - lo) / (hi - lo), NAN)
    return r + float(ev.const(c.kw("constant"), 0.0))


@op("ts_backfill")
def _ts_backfill(ev, c):
    d = ev.window(c, 1) if len(c.args) > 1 else int(ev.const(c.kw("lookback"), 252))
    return _ffill(ev.a(c, 0), d)


@op("ts_count_nans")
def _ts_count_nans(ev, c):
    s, _ = _rolling(np.isnan(ev.a(c, 0)).astype(float), ev.window(c, 1))
    return s


def _moment(ev, c, k):
    x, d = ev.a(c, 0), ev.window(c, 1)

    def fn(w):
        m = np.nanmean(w, -1, keepdims=True)
        dev = w - m
        var = np.nanmean(dev ** 2, -1)
        with np.errstate(all="ignore"):
            if k == 3:
                return np.where(var > 0, np.nanmean(dev ** 3, -1) / var ** 1.5, NAN)
            return np.where(var > 0, np.nanmean(dev ** 4, -1) / var ** 2 - 3, NAN)
    return _sliding(x, d, fn)


@op("ts_skewness")
def _ts_skew(ev, c):
    return _moment(ev, c, 3)


@op("ts_kurtosis")
def _ts_kurt(ev, c):
    return _moment(ev, c, 4)


@op("ts_step")
def _ts_step(ev, c):
    return np.broadcast_to(np.arange(ev.T, dtype=float)[:, None], (ev.T, ev.N)).copy()


def _last_change_idx(x):
    T = len(x)
    prev = _shift(x, 1)
    changed = ~((x == prev) | (np.isnan(x) & np.isnan(prev)))
    changed[0] = True
    idx = np.where(changed, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return idx


@op("days_from_last_change")
def _dflc(ev, c):
    x = ev.a(c, 0)
    return (np.arange(ev.T)[:, None] - _last_change_idx(x)).astype(float)


@op("last_diff_value")
def _ldv(ev, c):
    x, d = ev.a(c, 0), ev.window(c, 1)
    idx = _last_change_idx(x) - 1
    ok = (idx >= 0) & ((np.arange(ev.T)[:, None] - idx) <= d)
    vals = x[np.clip(idx, 0, None), np.arange(ev.N)[None, :]]
    return np.where(ok, vals, NAN)


@op("kth_element")
def _kth(ev, c):
    d = ev.window(c, 1)
    k = int(ev.const(c.kw("k", c.args[2] if len(c.args) > 2 else None), 1))
    if k <= 1:
        return _ffill(ev.a(c, 0), d - 1)

    def fn(w):
        rev = w[..., ::-1]
        cnt = np.cumsum(~np.isnan(rev), axis=-1)
        hit = cnt == k
        pos = np.argmax(hit, axis=-1)
        val = np.take_along_axis(rev, pos[..., None], axis=-1)[..., 0]
        return np.where(hit.any(-1), val, NAN)
    return _sliding(ev.a(c, 0), d, fn)


@op("hump")
def _hump(ev, c):
    x = ev.a(c, 0)
    h = float(ev.const(c.kw("hump", c.args[1] if len(c.args) > 1 else None), 0.01))
    out = x.copy()
    held = x[0].copy()
    for t in range(1, ev.T):
        cur = x[t]
        valid = ~np.isnan(cur)
        n = max(1, int(valid.sum()))
        thr = h * np.nanmean(np.abs(cur)) * math.sqrt(n) if valid.any() else 0.0
        move = np.isnan(held) | (np.abs(cur - held) > thr)
        held = np.where(move, cur, held)
        held = np.where(valid, held, NAN)
        out[t] = held
    return out


# --- cross sectional ----------------------------------------------------------
@op("rank")
def _rank(ev, c):
    return cs_rank(ev.a(c, 0))


@op("zscore")
def _zscore(ev, c):
    return cs_zscore(ev.a(c, 0))


@op("scale")
def _scale(ev, c):
    x = ev.a(c, 0)
    target = float(ev.const(c.kw("scale"), 1.0))
    s = np.nansum(np.abs(x), axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.where(s > 0, x / s * target, NAN)


@op("normalize")
def _normalize(ev, c):
    x = ev.a(c, 0)
    out = x - _cs_mean(x)
    if ev.const(c.kw("useStd"), False):
        s = _cs_std(x)
        with np.errstate(all="ignore"):
            out = np.where(s > 0, out / s, NAN)
    limit = float(ev.const(c.kw("limit"), 0.0))
    return np.clip(out, -limit, limit) if limit > 0 else out


@op("quantile")
def _quantile(ev, c):
    r = cs_rank(ev.a(c, 0))
    driver = str(ev.const(c.kw("driver"), "gaussian")).lower()
    sigma = float(ev.const(c.kw("sigma"), 1.0))
    u = np.clip(r, 0.005, 0.995)
    if driver == "uniform":
        return (2 * u - 1) * sigma
    if driver == "cauchy":
        return np.tan(np.pi * (u - 0.5)) * sigma
    return norm_ppf(u) * sigma


@op("winsorize")
def _winsorize(ev, c):
    x = ev.a(c, 0)
    k = float(ev.const(c.kw("std"), 4.0))
    m, s = _cs_mean(x), _cs_std(x)
    return np.clip(x, m - k * s, m + k * s)


@op("regression_neut")
def _reg_neut(ev, c):
    y, x = ev.a(c, 0), ev.a(c, 1)
    mask = np.isnan(x) | np.isnan(y)
    xm, ym = np.where(mask, NAN, x), np.where(mask, NAN, y)
    dx, dy = xm - _cs_mean(xm), ym - _cs_mean(ym)
    with np.errstate(all="ignore"):
        beta = np.nansum(dx * dy, 1, keepdims=True) / np.nansum(dx * dx, 1, keepdims=True)
    return np.where(mask, NAN, dy - np.nan_to_num(beta) * dx)


@op("vector_neut")
def _vec_neut(ev, c):
    x, y = ev.a(c, 0), ev.a(c, 1)
    with np.errstate(all="ignore"):
        proj = np.nansum(x * y, 1, keepdims=True) / np.nansum(y * y, 1, keepdims=True)
    return x - np.nan_to_num(proj) * y


# --- group ----------------------------------------------------------------------
def _gi(ev, c, i=1):
    return ev.group(c.args[i])


@op("group_neutralize")
def _gneut(ev, c):
    x = ev.a(c, 0)
    return x - _gi(ev, c).mean(x)


@op("group_mean")
def _gmean(ev, c):
    x = ev.a(c, 0)
    if len(c.args) >= 3:
        return ev.group(c.args[2]).mean(x, weights=ev.a(c, 1))
    return _gi(ev, c).mean(x)


@op("group_zscore")
def _gz(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    sd = g.std(x)
    with np.errstate(all="ignore"):
        return np.where(sd > 0, (x - g.mean(x)) / sd, NAN)


@op("group_std_dev")
def _gstd(ev, c):
    return _gi(ev, c).std(ev.a(c, 0))


@op("group_sum")
def _gsum(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    s, _, _ = g.reduce(x)
    return g.broadcast(s)


@op("group_count")
def _gcount(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    _, cnt, _ = g.reduce(x)
    return g.broadcast(cnt.astype(float))


@op("group_max")
def _gmax(ev, c):
    return _gi(ev, c).extreme(ev.a(c, 0), np.maximum)


@op("group_min")
def _gmin(ev, c):
    return _gi(ev, c).extreme(ev.a(c, 0), np.minimum)


@op("group_scale")
def _gscale(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    lo, hi = g.extreme(x, np.minimum), g.extreme(x, np.maximum)
    with np.errstate(all="ignore"):
        return np.where(hi > lo, (x - lo) / (hi - lo), NAN)


@op("group_normalize")
def _gnorm(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    s, _, _ = g.reduce(np.abs(x))
    tot = g.broadcast(s)
    with np.errstate(all="ignore"):
        return np.where(tot > 0, x / tot, NAN)


@op("group_rank")
def _grank(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    order, within, cnt, valid_sorted = g.ordered(x)
    n = cnt[g.idx[order]].astype(float)
    with np.errstate(all="ignore"):
        r = np.where(n > 1, within / (n - 1), 0.5)
    out = np.full(g.T * g.N, NAN)
    out[order] = np.where(valid_sorted, r, NAN)
    return out.reshape(g.T, g.N)


@op("group_median")
def _gmedian(ev, c):
    x = ev.a(c, 0)
    g = _gi(ev, c)
    order, within, cnt, valid_sorted = g.ordered(x)
    sidx = g.idx[order]
    target = (cnt[sidx] - 1) // 2
    pick = valid_sorted & (within == target)
    med = np.full(g.size, NAN)
    med[sidx[pick]] = x.ravel()[order][pick]
    return g.broadcast(med)


@op("group_backfill")
def _gbackfill(ev, c):
    x = ev.a(c, 0)
    d = ev.window(c, 2)
    fill = _gi(ev, c).mean(_rmean(x, d))
    return np.where(np.isnan(x), fill, x)


# --- transformational -------------------------------------------------------------
@op("trade_when")
def _trade_when(ev, c):
    x, y, z = ev.v(c, 0), ev.a(c, 1), ev.v(c, 2)
    enter = np.asarray(ev.arr(x)) > 0
    exit_ = np.asarray(ev.arr(z)) > 0
    trig = enter | exit_
    val = np.where(exit_, NAN, y)
    idx = np.where(trig, np.arange(ev.T)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = val[np.clip(idx, 0, None), np.arange(ev.N)[None, :]]
    return np.where(idx < 0, NAN, out)


@op("bucket")
def _bucket(ev, c):
    x = ev.a(c, 0)
    spec = c.kw("buckets")
    if spec is not None:
        edges = np.array([float(v) for v in str(ev.const(spec)).split(",") if v.strip()])
    else:
        lo, hi, step = (float(v) for v in str(ev.const(c.kw("range"), "0,1,0.1")).split(","))
        if step <= 0:
            raise EvalError("bucket() range step must be positive.")
        edges = np.arange(lo, hi + step / 2, step)
    out = np.digitize(np.nan_to_num(x, nan=0.0), edges).astype(float)
    return np.where(np.isnan(x), NAN, out)


@op("pasteurize")
def _paste(ev, c):
    return np.where(ev.umask, ev.a(c, 0), NAN)


@op("clamp")
def _clamp(ev, c):
    x = ev.a(c, 0)
    lo = float(ev.const(c.kw("lower"), -np.inf))
    hi = float(ev.const(c.kw("upper"), np.inf))
    return np.clip(x, lo, hi)


@op("tail")
def _tail(ev, c):
    x = ev.a(c, 0)
    lo, hi = float(ev.const(c.kw("lower"), 0)), float(ev.const(c.kw("upper"), 0))
    newval = float(ev.const(c.kw("newval"), 0))
    return np.where((x >= lo) & (x <= hi), newval, x)


@op("left_tail")
def _ltail(ev, c):
    x = ev.a(c, 0)
    return np.where(x > float(ev.const(c.kw("maximum"), 0)), NAN, x)


@op("right_tail")
def _rtail(ev, c):
    x = ev.a(c, 0)
    return np.where(x < float(ev.const(c.kw("minimum"), 0)), NAN, x)


@op("vec_avg", "vec_sum", "vec_max", "vec_min")
def _vec_identity(ev, c):
    return ev.a(c, 0)


@op("vec_count")
def _vec_count(ev, c):
    return (~np.isnan(ev.a(c, 0))).astype(float)


@op("vec_stddev")
def _vec_std(ev, c):
    x = ev.a(c, 0)
    return np.where(np.isnan(x), NAN, 0.0)


def implemented(name: str) -> bool:
    return name in OPS
