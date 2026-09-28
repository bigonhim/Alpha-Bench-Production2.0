"""Classify a pasted alpha's structure: idea family, horizon, and building blocks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .ast import BinOp, Call, Ident, Node, Num, Unary, idents, inline, walk
from .operators import GROUP_FIELDS, OPERATORS


@dataclass
class Structure:
    family: str
    thesis: str
    horizon: str
    fields: list[str]
    field_categories: dict[str, str]
    operators: list[str]
    windows: list[int]
    has_rank: bool
    has_group_op: bool
    has_smoothing: bool
    has_event_gate: bool
    sign: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = [f"A {self.horizon}-horizon {self.family.replace('_', ' ')} alpha ({self.thesis})"]
        if self.fields:
            parts.append(f"built on {', '.join(self.fields[:5])}")
        if self.has_group_op:
            parts.append("with group-relative comparison")
        if self.has_event_gate:
            parts.append("gated by an event/condition")
        if self.has_smoothing:
            parts.append("and smoothing that lowers turnover")
        return " ".join(parts) + "."


def _windows(node: Node) -> list[int]:
    out = []
    for n in walk(node):
        if isinstance(n, Call):
            spec = OPERATORS.get(n.name)
            if spec and spec.window is not None and spec.window < len(n.args):
                w = n.args[spec.window]
                if isinstance(w, Num):
                    out.append(int(w.value))
    return out


def _outer_sign(node: Node) -> int:
    if isinstance(node, Unary) and node.op == "-":
        return -_outer_sign(node.operand)
    if isinstance(node, Num):
        return 1 if node.value >= 0 else -1
    if isinstance(node, BinOp) and node.op == "*" and isinstance(node.left, Num):
        return (1 if node.left.value >= 0 else -1) * _outer_sign(node.right)
    if isinstance(node, BinOp) and node.op == "*" and isinstance(node.right, Num):
        return (1 if node.right.value >= 0 else -1) * _outer_sign(node.left)
    if isinstance(node, Call) and node.name == "reverse":
        return -1
    if isinstance(node, Call) and node.name in ("rank", "zscore", "scale", "group_neutralize", "ts_decay_linear",
                                                  "group_rank", "group_zscore", "winsorize", "ts_mean"):
        return _outer_sign(node.args[0]) if node.args else 1
    return 1


def classify(node: Node, category_of) -> Structure:
    tree = inline(node)
    names = [n for n in sorted(idents(tree)) if n not in GROUP_FIELDS]
    cats = {n: category_of(n) for n in names}
    ops = sorted({n.name for n in walk(tree) if isinstance(n, Call)})
    wins = _windows(tree)
    sign = _outer_sign(tree)
    has_group = any(o.startswith("group_") for o in ops)
    smoothing = any(OPERATORS.get(o) is not None and OPERATORS[o].smooth for o in ops)
    event = any(o in ("trade_when", "if_else", "days_from_last_change") for o in ops) or \
        any(isinstance(n, BinOp) and n.op in ("<", ">", "<=", ">=", "==", "!=") for n in walk(tree))
    catset = set(cats.values())
    fset = set(names)
    notes: list[str] = []

    if event:
        family = "event_driven"
    elif catset <= {"pv"}:
        family = "price_volume"
    elif "fundamental" in catset and catset <= {"fundamental", "pv"}:
        family = "fundamental"
    elif catset & {"sentiment", "socialmedia", "news", "option", "shortinterest", "insiders", "other", "model"}:
        family = "alternative"
    elif "analyst" in catset:
        family = "analyst"
    else:
        family = "cross_sectional"

    short = [w for w in wins if w <= 10]
    horizon = "short" if (short and max(wins) <= 20) else "long" if wins and min(wins) >= 120 else "medium"
    if not wins:
        horizon = "long" if family == "fundamental" else "short"

    if family == "price_volume":
        if "ts_corr" in ops and {"volume"} & fset:
            thesis = "price-volume divergence"
        elif fset & {"returns", "close", "vwap", "open"} and horizon == "short":
            thesis = "short-term reversal" if sign < 0 else "short-term continuation"
        elif horizon == "long" and fset & {"returns", "close"}:
            thesis = "momentum" if sign > 0 else "long-term reversal"
        elif "ts_std_dev" in ops:
            thesis = "volatility anomaly"
        elif fset & {"volume", "adv20"}:
            thesis = "volume/liquidity effect"
        else:
            thesis = "price pattern"
    elif family == "fundamental":
        if fset & {"income", "cashflow_op"} and "-" in {n.op for n in walk(tree) if isinstance(n, BinOp)}:
            thesis = "accruals / earnings quality"
        elif "cap" in fset or "close" in fset or "enterprise_value" in fset:
            thesis = "valuation (value)"
        elif fset & {"operating_income", "income", "ebitda"} and fset & {"assets", "equity"}:
            thesis = "profitability (quality)"
        elif "ts_delta" in ops:
            thesis = "fundamental momentum / growth"
        else:
            thesis = "fundamental ratio"
    elif family == "analyst":
        thesis = "estimate revisions" if "ts_delta" in ops or "ts_rank" in ops else "analyst consensus"
    elif family == "alternative":
        thesis = "sentiment" if catset & {"sentiment", "socialmedia", "news"} else "positioning / options"
    elif family == "event_driven":
        thesis = "conditional / event-gated signal"
    else:
        thesis = "multi-factor composite"

    if len(catset) >= 2 and family != "event_driven":
        notes.append("Combines data from several categories.")
    if not any(o in ops for o in ("rank", "zscore", "group_rank", "quantile", "ts_rank", "scale", "winsorize")):
        notes.append("No normalization operator: extreme values can concentrate weight.")
    return Structure(family=family, thesis=thesis, horizon=horizon, fields=names, field_categories=cats,
                     operators=ops, windows=wins, has_rank="rank" in ops or "group_rank" in ops,
                     has_group_op=has_group, has_smoothing=smoothing, has_event_gate=event, sign=sign,
                     notes=notes)
