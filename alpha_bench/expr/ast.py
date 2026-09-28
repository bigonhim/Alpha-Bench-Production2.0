"""Immutable AST for BRAIN Fast Expressions, plus printing and tree utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator


class Node:
    __slots__ = ()


@dataclass(frozen=True)
class Num(Node):
    value: float


@dataclass(frozen=True)
class Str(Node):
    value: str


@dataclass(frozen=True)
class Ident(Node):
    name: str


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: tuple[Node, ...] = ()
    kwargs: tuple[tuple[str, Node], ...] = ()

    def kw(self, key: str, default: Node | None = None) -> Node | None:
        for k, v in self.kwargs:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Unary(Node):
    op: str
    operand: Node


@dataclass(frozen=True)
class Ternary(Node):
    cond: Node
    then: Node
    other: Node


@dataclass(frozen=True)
class Program(Node):
    assigns: tuple[tuple[str, Node], ...]
    result: Node


# Infix operator -> named BRAIN operator, and binding strength for printing.
INFIX_TO_OP = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "^": "power",
               "<": "less", ">": "greater", "<=": "less_equal", ">=": "greater_equal",
               "==": "equal", "!=": "not_equal", "&&": "and", "||": "or"}
PRECEDENCE = {"?": 1, "||": 2, "&&": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4, "!=": 4,
              "+": 5, "-": 5, "*": 6, "/": 6, "unary": 7, "^": 8}


def fmt_num(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def to_str(node: Node) -> str:
    """Canonical Fast Expression text for a tree."""
    if isinstance(node, Program):
        lines = [f"{name} = {to_str(expr)};" for name, expr in node.assigns]
        lines.append(to_str(node.result))
        return "\n".join(lines)
    return _fmt(node, 0)


def _fmt(node: Node, parent_prec: int) -> str:
    if isinstance(node, Num):
        text = fmt_num(node.value)
        return f"({text})" if node.value < 0 and parent_prec > 0 else text
    if isinstance(node, Str):
        return '"' + node.value.replace('"', '\\"') + '"'
    if isinstance(node, Ident):
        return node.name
    if isinstance(node, Call):
        parts = [_fmt(a, 0) for a in node.args]
        parts += [f"{k}={_fmt(v, 0)}" for k, v in node.kwargs]
        return f"{node.name}({', '.join(parts)})"
    if isinstance(node, Unary):
        prec = PRECEDENCE["unary"]
        text = node.op + _fmt(node.operand, prec)
        return f"({text})" if prec < parent_prec else text
    if isinstance(node, BinOp):
        prec = PRECEDENCE[node.op]
        right_assoc = node.op == "^"
        left = _fmt(node.left, prec + (1 if right_assoc else 0))
        right = _fmt(node.right, prec + (0 if right_assoc else 1))
        text = f"{left} {node.op} {right}"
        return f"({text})" if prec < parent_prec else text
    if isinstance(node, Ternary):
        prec = PRECEDENCE["?"]
        text = f"{_fmt(node.cond, prec + 1)} ? {_fmt(node.then, prec)} : {_fmt(node.other, prec)}"
        return f"({text})" if prec < parent_prec else text
    raise TypeError(f"unknown node {node!r}")


def children(node: Node) -> tuple[Node, ...]:
    if isinstance(node, Call):
        return node.args + tuple(v for _, v in node.kwargs)
    if isinstance(node, BinOp):
        return (node.left, node.right)
    if isinstance(node, Unary):
        return (node.operand,)
    if isinstance(node, Ternary):
        return (node.cond, node.then, node.other)
    if isinstance(node, Program):
        return tuple(v for _, v in node.assigns) + (node.result,)
    return ()


def walk(node: Node) -> Iterator[Node]:
    yield node
    for child in children(node):
        yield from walk(child)


def transform(node: Node, fn: Callable[[Node], Node | None]) -> Node:
    """Bottom-up rewrite. `fn` returns a replacement node, or None to keep it."""
    if isinstance(node, Call):
        node = Call(node.name, tuple(transform(a, fn) for a in node.args),
                    tuple((k, transform(v, fn)) for k, v in node.kwargs))
    elif isinstance(node, BinOp):
        node = BinOp(node.op, transform(node.left, fn), transform(node.right, fn))
    elif isinstance(node, Unary):
        node = Unary(node.op, transform(node.operand, fn))
    elif isinstance(node, Ternary):
        node = Ternary(transform(node.cond, fn), transform(node.then, fn), transform(node.other, fn))
    elif isinstance(node, Program):
        node = Program(tuple((k, transform(v, fn)) for k, v in node.assigns), transform(node.result, fn))
    out = fn(node)
    return node if out is None else out


def inline(node: Node) -> Node:
    """Substitute assignments so the whole alpha is a single expression tree."""
    if not isinstance(node, Program):
        return node
    env: dict[str, Node] = {}

    def sub(n: Node) -> Node | None:
        if isinstance(n, Ident) and n.name in env:
            return env[n.name]
        return None

    for name, expr in node.assigns:
        env[name] = transform(expr, sub)
    return transform(node.result, sub)


def idents(node: Node) -> set[str]:
    return {n.name for n in walk(inline(node)) if isinstance(n, Ident)}


def calls(node: Node) -> list[str]:
    return [n.name for n in walk(inline(node)) if isinstance(n, Call)]


def depth(node: Node) -> int:
    kids = children(node)
    return 1 + (max(depth(k) for k in kids) if kids else 0)


def size(node: Node) -> int:
    return sum(1 for _ in walk(node))
