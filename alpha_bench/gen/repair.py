"""Explainable, minimal-change repair.

Given the checks an alpha fails, propose small concrete edits (settings tweaks
first, then light expression wraps, then structural changes), simulate each,
and rank them by how many failures they clear with the smallest change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..engine.evaluator import EvalError, FieldCatalog, parallel_map
from ..engine.simulate import OfflineEngine
from ..expr.ast import BinOp, Call, Ident, Node, Num, Unary, inline, to_str, transform, walk
from ..expr.fingerprint import logic_fingerprint
from ..expr.operators import GROUP_FIELDS, OPERATORS
from ..rules.engine import Check, Rules, evaluate
from ..settings_model import NEUTRALIZATIONS, SUB_UNIVERSE, SimSettings
from .ideas import C, F
from .scoring import score

_NICE_WINDOWS = (2, 3, 5, 10, 15, 20, 40, 60, 120, 126, 250, 252, 500)
_SPARSE_CATS = {"sentiment", "news", "socialmedia", "analyst", "option", "model", "earnings"}
_NEUT_ORDER = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]


@dataclass
class Fix:
    title: str
    detail: str
    targets: list[str]
    node: Node
    settings: SimSettings
    change: str
    size: int                       # 1 settings tweak, 2 light wrap, 3 structural
    result: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (logic_fingerprint(self.node), tuple(sorted(self.settings.to_dict().items())))


# ------------------------------------------------------------ tree edits
def flip(node: Node) -> Node:
    node = inline(node)
    if isinstance(node, Unary) and node.op == "-":
        return node.operand
    return Unary("-", node)


def scale_windows(node: Node, factor: float) -> Node:
    def fn(n: Node):
        if not isinstance(n, Call):
            return None
        spec = OPERATORS.get(n.name)
        if not spec or spec.window is None or spec.window >= len(n.args):
            return None
        w = n.args[spec.window]
        if not isinstance(w, Num) or w.value < 2:
            return None
        target = max(2, min(504, w.value * factor))
        nice = min(_NICE_WINDOWS, key=lambda v: abs(v - target))
        if nice == w.value:
            nice = int(round(target))
        args = list(n.args)
        args[spec.window] = Num(float(nice))
        return Call(n.name, tuple(args), n.kwargs)
    return transform(inline(node), fn)


def has_windows(node: Node) -> bool:
    return any(isinstance(n, Call) and OPERATORS.get(n.name) and OPERATORS[n.name].window is not None
               for n in walk(inline(node)))


def backfill_sparse(node: Node, catalog: FieldCatalog, days: int = 20) -> Node | None:
    tree = inline(node)
    protected = {id(a) for n in walk(tree) if isinstance(n, Call) and n.name in ("ts_backfill", "group_backfill")
                 for a in n.args[:1]}
    changed = False

    def fn(n: Node):
        nonlocal changed
        if isinstance(n, Ident) and n.name not in GROUP_FIELDS and id(n) not in protected \
                and catalog.category(n.name) in _SPARSE_CATS:
            changed = True
            return C("ts_backfill", n, days)
        return None
    out = transform(tree, fn)
    return out if changed else None


def _looks_like_reversion(node: Node) -> bool:
    """Short-horizon price/return signal with a negative outer sign (e.g. -ts_delta(close, 5))."""
    from ..expr.classify import _outer_sign  # noqa: PLC0415
    tree = inline(node)
    names = {n.name for n in walk(tree) if isinstance(n, Ident)}
    wins = [n.args[1].value for n in walk(tree) if isinstance(n, Call) and n.name in ("ts_delta", "ts_sum", "ts_mean",
            "ts_zscore") and len(n.args) > 1 and isinstance(n.args[1], Num)]
    short = not wins or max(wins) <= 20
    return bool(names & {"close", "returns", "open", "vwap"}) and short and _outer_sign(tree) < 0 \
        or (isinstance(tree, Call) and tree.name == "group_rank" and isinstance(tree.args[0], BinOp))


_PRICES = {"close", "open", "high", "low", "vwap"}


def _raw_price_difference(node: Node) -> Node | None:
    """Rewrite `a - b` between two price fields as `(a - b) / close` (unless already divided)."""
    tree = inline(node)
    divided = {id(n.left) for n in walk(tree) if isinstance(n, BinOp) and n.op == "/"}
    changed = False

    def fn(n: Node):
        nonlocal changed
        if (isinstance(n, BinOp) and n.op == "-" and isinstance(n.left, Ident) and isinstance(n.right, Ident)
                and n.left.name in _PRICES and n.right.name in _PRICES and id(n) not in divided):
            changed = True
            return BinOp("/", n, F("close"))
        return None
    out = transform(tree, fn)
    return out if changed else None


def select_for_round(fixes: list[Fix], per_round: int) -> list[Fix]:
    """Smallest edits first, but always leave room for a couple of structural ideas."""
    by = {1: [], 2: [], 3: []}
    for f in fixes:
        by.setdefault(f.size, []).append(f)
    structural = by[3][:2]
    light = by[2][:max(2, per_round // 3)]
    rest = by[1][:max(0, per_round - len(structural) - len(light))]
    chosen = rest + light + structural
    extra = [f for f in fixes if f not in chosen]
    return (chosen + extra)[:per_round]


def outer_is(node: Node, *names: str) -> bool:
    node = inline(node)
    return isinstance(node, Call) and node.name in names


# ------------------------------------------------------------ proposals
def propose(node: Node, s: SimSettings, checks: list[Check], metrics: dict,
            catalog: FieldCatalog, correlation: dict | None = None) -> list[Fix]:
    failing = {c.name for c in checks if c.result in ("FAIL", "WARN")}
    out: list[Fix] = []
    sharpe = metrics.get("sharpe", 0.0)
    to = metrics.get("turnover", 0.0)

    def add(title, detail, targets, n, st, change, size):
        out.append(Fix(title, detail, targets, n, st, change, size))

    if sharpe < 0:
        add("Flip the sign", f"Sharpe is {sharpe:.2f}: the idea is right but the direction is wrong.",
            ["LOW_SHARPE", "LOW_FITNESS"], flip(node), s, "expression x -1", 1)

    if "HIGH_TURNOVER" in failing or "RISK_TURNOVER" in failing or ("LOW_FITNESS" in failing and to > 0.125):
        for d in sorted({s.decay + 2, s.decay + 4, max(6, s.decay * 2), s.decay + 8}):
            if d <= 30 and d != s.decay:
                add(f"Decay {s.decay} -> {d}", "Linear decay averages the signal over recent days, so positions "
                    "change more slowly. Fitness gains stop once turnover reaches 12.5%.",
                    ["HIGH_TURNOVER", "LOW_FITNESS"], node, s.with_(decay=d), f"decay {s.decay} -> {d}", 1)
        if has_windows(node):
            add("Lengthen lookbacks x2", "Longer windows make the signal itself slower-moving.",
                ["HIGH_TURNOVER", "LOW_FITNESS"], scale_windows(node, 2.0), s, "all lookback windows x2", 2)
        if not outer_is(node, "ts_decay_linear", "ts_mean"):
            add("Smooth with ts_decay_linear(x, 5)", "Smoothing inside the expression cuts turnover while keeping "
                "the decay setting free.", ["HIGH_TURNOVER", "LOW_FITNESS"],
                C("ts_decay_linear", inline(node), 5), s, "wrap in ts_decay_linear(., 5)", 2)
        add("Only trade on meaningful changes (hump)", "hump() holds positions until the signal moves enough.",
            ["HIGH_TURNOVER"], C("hump", inline(node), hump=0.01), s, "wrap in hump(., hump=0.01)", 2)

    if "LOW_TURNOVER" in failing:
        if s.decay > 0:
            nd = max(0, s.decay - 4)
            add(f"Decay {s.decay} -> {nd}", "Less decay lets positions update faster.", ["LOW_TURNOVER"],
                node, s.with_(decay=nd), f"decay {s.decay} -> {nd}", 1)
        if has_windows(node):
            add("Shorten lookbacks /2", "Shorter windows make the signal respond faster.", ["LOW_TURNOVER"],
                scale_windows(node, 0.5), s, "all lookback windows /2", 2)

    if ("HIGH_TURNOVER" in failing or "LOW_FITNESS" in failing) and _looks_like_reversion(node):
        gate = BinOp(">", C("ts_std_dev", F("returns"), 20), C("ts_mean", C("ts_std_dev", F("returns"), 20), 120))
        add("Trade only in high-volatility regimes", "Learn2Quant Lesson 4: control turnover by trading reversion "
            "only in higher-volatility conditions and holding during low volatility.",
            ["HIGH_TURNOVER", "LOW_FITNESS"], C("trade_when", gate, inline(node), -1), s,
            "trade_when(volatility above its 6-month average, ., -1)", 3)

    raw = _raw_price_difference(node)
    if raw is not None and failing & {"LOW_SHARPE", "LOW_FITNESS", "RETURNS_DRAWDOWN"}:
        add("Scale the price difference by price", "A raw price difference ranks expensive stocks as 'bigger' "
            "moves; dividing by close makes it a percentage move, comparable across stocks.",
            ["LOW_SHARPE", "LOW_FITNESS"], raw, s, "(price difference) / close", 2)

    if failing & {"LOW_SHARPE", "LOW_FITNESS", "DRAWDOWN", "RETURNS_DRAWDOWN", "LOW_2Y_SHARPE"}:
        for n in _NEUT_ORDER:
            if n != s.neutralization.upper():
                add(f"Neutralization {s.neutralization} -> {n}",
                    "Finer groups remove more common-factor noise (often higher Sharpe); coarser groups keep more "
                    "return (sometimes higher fitness).", ["LOW_SHARPE", "LOW_FITNESS", "DRAWDOWN", "RETURNS_DRAWDOWN"],
                    node, s.with_(neutralization=n), f"neutralization {s.neutralization} -> {n}", 1)
        if not outer_is(node, "rank", "group_rank", "quantile", "zscore"):
            add("Rank the signal", "Ranking caps the influence of outliers and usually steadies Sharpe.",
                ["LOW_SHARPE", "CONCENTRATED_WEIGHT"], C("rank", inline(node)), s, "wrap in rank()", 2)
        add("Industry-relative ranking", "Compare each stock only with its industry peers.",
            ["LOW_SHARPE"], C("group_rank", inline(node), F("subindustry" if s.neutralization != "SUBINDUSTRY"
                                                              else "industry")), s, "wrap in group_rank(., subindustry)", 2)
        if failing & {"DRAWDOWN", "RETURNS_DRAWDOWN"}:
            add("Volatility-scale", "Dividing by recent volatility shrinks bets on the riskiest names.",
                ["DRAWDOWN", "LOW_SHARPE"], BinOp("/", inline(node), C("ts_std_dev", F("returns"), 20)), s,
                "divide by ts_std_dev(returns, 20)", 2)

    if "RISK_SECTOR" in failing:
        for n in ("SECTOR", "INDUSTRY", "SUBINDUSTRY"):
            if n != s.neutralization.upper():
                add(f"Neutralization {s.neutralization} -> {n}", "Group neutralization removes net sector bets.",
                    ["RISK_SECTOR"], node, s.with_(neutralization=n), f"neutralization {s.neutralization} -> {n}", 1)

    if "CONCENTRATED_WEIGHT" in failing or "RISK_WEIGHT" in failing:
        for t in (0.05, 0.02, 0.01):
            if t < s.truncation:
                add(f"Truncation {s.truncation} -> {t}", "Hard-caps each stock's weight.", ["CONCENTRATED_WEIGHT"],
                    node, s.with_(truncation=t), f"truncation {s.truncation} -> {t}", 1)
        if not outer_is(node, "rank", "group_rank", "quantile"):
            add("Rank the signal", "Ranks are bounded, so no single stock dominates.", ["CONCENTRATED_WEIGHT"],
                C("rank", inline(node)), s, "wrap in rank()", 2)
        bf = backfill_sparse(node, catalog)
        if bf is not None:
            add("Backfill sparse fields", "Sparse data leaves few names with positions; forward-filling spreads "
                "the book.", ["CONCENTRATED_WEIGHT"], bf, s, "ts_backfill(field, 20) on sparse fields", 2)

    if "LOW_SUB_UNIVERSE_SHARPE" in failing:
        add("Neutralize by size bucket", "Stops the alpha leaning on small caps, which is what fails in the "
            "larger-cap sub-universe.", ["LOW_SUB_UNIVERSE_SHARPE"],
            C("group_neutralize", inline(node), C("bucket", C("rank", F("cap")), range="0.1,1,0.1")), s,
            "group_neutralize(., bucket(rank(cap)))", 2)
        add("Tilt toward liquid names", "Weights the signal by liquidity rank.", ["LOW_SUB_UNIVERSE_SHARPE"],
            BinOp("*", C("rank", inline(node)), C("rank", F("adv20"))), s, "rank(.) * rank(adv20)", 2)
        if s.universe in SUB_UNIVERSE:
            su = SUB_UNIVERSE[s.universe]
            add(f"Universe {s.universe} -> {su}", "Simulate directly where the test is applied.",
                ["LOW_SUB_UNIVERSE_SHARPE"], node, s.with_(universe=su), f"universe {s.universe} -> {su}", 1)

    if "SELF_CORRELATION" in failing:
        if has_windows(node):
            for f in (3.0, 1 / 3):
                add(f"Change horizon (windows x{f:.2g})", "A different horizon trades a different part of the "
                    "idea and decorrelates PnL.", ["SELF_CORRELATION"], scale_windows(node, f), s,
                    f"all lookback windows x{f:.2g}", 2)
        for n in _NEUT_ORDER:
            if n != s.neutralization.upper() and n != "NONE":
                add(f"Neutralization -> {n}", "Different neutralization changes which risks the PnL carries.",
                    ["SELF_CORRELATION"], node, s.with_(neutralization=n), f"neutralization -> {n}", 1)

    uniq: dict[tuple, Fix] = {}
    for fx in out:
        uniq.setdefault(fx.key(), fx)
    return list(uniq.values())


# ------------------------------------------------------------ evaluation
def assess(engine: OfflineEngine, rules: Rules, node: Node, settings: SimSettings,
           corr_fn: Callable | None = None, *, full: bool = False) -> dict:
    res = engine.simulate(node, settings, sub_universe=True, quick=not full)
    corr = corr_fn(res) if corr_fn else None
    checks = evaluate(res.metrics, settings, rules, correlation=corr)
    return {"sim": res, "checks": checks, "correlation": corr,
            "score": score(res.metrics, checks, rules, settings)}


def rank_fixes(engine: OfflineEngine, rules: Rules, base: dict, fixes: list[Fix],
               corr_fn: Callable | None = None, limit: int = 12) -> list[Fix]:
    base_fail = {c.name for c in base["checks"] if c.result == "FAIL"}
    ranked = []

    def run(fx: Fix):
        try:
            return assess(engine, rules, fx.node, fx.settings, corr_fn)
        except EvalError as exc:
            return exc

    fixes = fixes[:40]
    for fx, a in zip(fixes, parallel_map(run, fixes)):
        if isinstance(a, Exception):
            fx.result = {"error": str(a)}
            continue
        now_fail = {c.name for c in a["checks"] if c.result == "FAIL"}
        m, bm = a["sim"].metrics, base["sim"].metrics
        fx.result = {
            "expression": to_str(fx.node), "settings": fx.settings.to_dict(), "score": a["score"],
            "metrics": {k: m.get(k) for k in ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin",
                                             "max_weight", "sub_universe_sharpe")},
            "delta": {k: (m.get(k) or 0) - (bm.get(k) or 0) for k in ("sharpe", "fitness", "turnover", "returns")},
            "cleared": sorted(base_fail - now_fail), "introduced": sorted(now_fail - base_fail),
            "still_failing": sorted(now_fail), "passes_all": not now_fail,
        }
        fx._assessment = a  # type: ignore[attr-defined]
        ranked.append(fx)
    ranked.sort(key=lambda f: (-len(f.result["cleared"]) + len(f.result["introduced"]), f.size,
                               -f.result["score"]))
    return ranked[:limit]


def auto_repair(engine: OfflineEngine, rules: Rules, node: Node, settings: SimSettings, catalog: FieldCatalog,
                corr_fn: Callable | None = None, rounds: int = 3, per_round: int = 10,
                current: dict | None = None) -> dict:
    """Greedy: apply the best-scoring fix each round until all checks pass or nothing helps."""
    current = current or assess(engine, rules, node, settings, corr_fn)
    history = []
    for _ in range(rounds):
        if not any(c.result == "FAIL" for c in current["checks"]):
            break
        fixes = propose(node, settings, current["checks"], current["sim"].metrics, catalog, current["correlation"])
        ranked = rank_fixes(engine, rules, current, select_for_round(fixes, per_round), corr_fn, limit=per_round)
        best = max(ranked, key=lambda f: f.result["score"], default=None)
        if best is None or best.result["score"] <= current["score"] + 1e-6:
            break
        node, settings = best.node, best.settings
        current = best._assessment  # type: ignore[attr-defined]
        history.append({"title": best.title, "change": best.change, "detail": best.detail,
                        "cleared": best.result["cleared"]})
    return {"node": node, "settings": settings, "assessment": current, "history": history}


def neutralization_note(neut: str) -> str:
    return {
        "NONE": "No neutralization: the alpha keeps its market and sector tilts (more return, more risk).",
        "MARKET": "MARKET neutralization makes the book dollar-neutral, so only relative stock picks matter.",
        "SECTOR": "SECTOR neutralization removes sector-wide moves while keeping industry tilts.",
        "INDUSTRY": "INDUSTRY neutralization removes industry-wide moves, isolating stock-specific effects.",
        "SUBINDUSTRY": "SUBINDUSTRY neutralization compares each stock only with its closest peers.",
    }.get(neut.upper(), f"{neut} neutralization.")


__all__ = ["Fix", "propose", "rank_fixes", "auto_repair", "assess", "flip", "scale_windows", "NEUTRALIZATIONS"]
