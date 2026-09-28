"""Structural fingerprints used to avoid re-deriving the same alpha logic.

* logic fingerprint: the full tree with numeric constants bucketed, so
  ts_mean(x, 20) and ts_mean(x, 21) are the same idea but ts_mean(x, 5) is not.
* shape fingerprint: additionally replaces each field with its category, so
  rank(sales/assets) and rank(income/assets) share a shape.
"""
from __future__ import annotations

import hashlib
import math

from .ast import BinOp, Call, Ident, Node, Num, Str, Ternary, Unary, inline, walk
from .operators import GROUP_FIELDS

_BUCKETS = (1, 2, 3, 5, 8, 13, 22, 40, 66, 130, 260, 520)
_COMMUTATIVE = {"+", "*", "add", "multiply", "max", "min", "&&", "||", "and", "or"}


def bucket(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if v == int(v) and 1 <= abs(v) <= 1000:
        mag = abs(v)
        idx = min(range(len(_BUCKETS)), key=lambda i: abs(math.log(mag) - math.log(_BUCKETS[i])))
        return ("-" if v < 0 else "") + f"w{_BUCKETS[idx]}"
    return ("-" if v < 0 else "") + f"c{round(math.log10(abs(v)), 0) if v else 0}"


def _canon(node: Node, field_mapper) -> str:
    if isinstance(node, Num):
        return bucket(node.value)
    if isinstance(node, Str):
        return f'"{node.value}"'
    if isinstance(node, Ident):
        return node.name if node.name in GROUP_FIELDS else field_mapper(node.name)
    if isinstance(node, Call):
        parts = [_canon(a, field_mapper) for a in node.args]
        if node.name in _COMMUTATIVE:
            parts.sort()
        parts += [f"{k}={_canon(v, field_mapper)}" for k, v in sorted(node.kwargs)]
        return f"{node.name}({','.join(parts)})"
    if isinstance(node, BinOp):
        a, b = _canon(node.left, field_mapper), _canon(node.right, field_mapper)
        if node.op in _COMMUTATIVE and b < a:
            a, b = b, a
        return f"({a}{node.op}{b})"
    if isinstance(node, Unary):
        return f"{node.op}{_canon(node.operand, field_mapper)}"
    if isinstance(node, Ternary):
        return f"({_canon(node.cond, field_mapper)}?{_canon(node.then, field_mapper)}:{_canon(node.other, field_mapper)})"
    return type(node).__name__


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def logic_fingerprint(node: Node) -> str:
    return _h(_canon(inline(node), lambda f: f))


def shape_fingerprint(node: Node, category_of) -> str:
    return _h(_canon(inline(node), lambda f: f"<{category_of(f)}>"))


def operator_set(node: Node) -> frozenset[str]:
    return frozenset(n.name for n in walk(inline(node)) if isinstance(n, Call))
