"""Tokenizer and recursive-descent parser for BRAIN Fast Expressions.

Supports: numbers, identifiers (data fields, group names, variables), strings,
function calls with positional and keyword arguments, infix arithmetic and
comparison, `&&` / `||`, unary `-` `+` `!`, `^` power, `cond ? a : b`, and
multi-statement alphas (`a = ...; b = ...; final_expr`). `#` and `//`
comments are stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ast import BinOp, Call, Ident, Node, Num, Program, Str, Ternary, Unary


class ParseError(ValueError):
    def __init__(self, message: str, pos: int, source: str):
        self.pos = pos
        self.source = source
        line = source.count("\n", 0, pos) + 1
        col = pos - (source.rfind("\n", 0, pos) + 1) + 1
        super().__init__(f"{message} (line {line}, column {col})")


@dataclass
class Tok:
    kind: str   # num | str | ident | op | eof
    text: str
    pos: int


_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<op>&&|\|\||<=|>=|==|!=|[-+*/^<>=!?:,;()])
""", re.VERBOSE)


def _strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines():
        in_str = None
        cut = len(line)
        for i, ch in enumerate(line):
            if in_str:
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch == "#" or line.startswith("//", i):
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ParseError(f"Unexpected character {src[pos]!r}", pos, src)
        kind = m.lastgroup
        if kind != "ws":
            toks.append(Tok(kind, m.group(), pos))
        pos = m.end()
    toks.append(Tok("eof", "", len(src)))
    return toks


class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    @property
    def cur(self) -> Tok:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def eat(self, text: str | None = None, kind: str | None = None) -> Tok:
        tok = self.cur
        if (text is not None and tok.text != text) or (kind is not None and tok.kind != kind):
            want = repr(text) if text else kind
            got = repr(tok.text) if tok.text else "end of expression"
            raise ParseError(f"Expected {want} but found {got}", tok.pos, self.src)
        self.i += 1
        return tok

    def accept(self, text: str) -> bool:
        if self.cur.kind == "op" and self.cur.text == text:
            self.i += 1
            return True
        return False

    # program := stmt (';' stmt)* ';'?
    def program(self) -> Node:
        assigns: list[tuple[str, Node]] = []
        exprs: list[Node] = []
        while self.cur.kind != "eof":
            if self.cur.kind == "ident" and self.peek().kind == "op" and self.peek().text == "=":
                name = self.eat(kind="ident").text
                self.eat("=")
                assigns.append((name, self.expr()))
            else:
                exprs.append(self.expr())
            if not self.accept(";"):
                break
        if self.cur.kind != "eof":
            raise ParseError(f"Unexpected {self.cur.text!r}", self.cur.pos, self.src)
        if not exprs:
            if assigns:
                # BRAIN uses the last statement's value; allow a trailing assignment.
                return Program(tuple(assigns[:-1]), assigns[-1][1]) if len(assigns) > 1 else assigns[0][1]
            raise ParseError("Empty expression", 0, self.src)
        if len(exprs) > 1:
            raise ParseError("Only the last statement may be a bare expression", self.cur.pos, self.src)
        return Program(tuple(assigns), exprs[0]) if assigns else exprs[0]

    def expr(self) -> Node:
        cond = self.binary(2)
        if self.accept("?"):
            then = self.expr()
            self.eat(":")
            return Ternary(cond, then, self.expr())
        return cond

    _LEVELS = {2: ("||",), 3: ("&&",), 4: ("<", ">", "<=", ">=", "==", "!="), 5: ("+", "-"), 6: ("*", "/")}

    def binary(self, level: int) -> Node:
        if level > 6:
            return self.unary()
        left = self.binary(level + 1)
        while self.cur.kind == "op" and self.cur.text in self._LEVELS[level]:
            op = self.eat().text
            left = BinOp(op, left, self.binary(level + 1))
        return left

    def unary(self) -> Node:
        if self.cur.kind == "op" and self.cur.text in ("-", "+", "!"):
            op = self.eat().text
            operand = self.unary()
            if op == "+":
                return operand
            if op == "-" and isinstance(operand, Num):
                return Num(-operand.value)
            return Unary(op, operand)
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        if self.accept("^"):
            return BinOp("^", base, self.unary())
        return base

    def primary(self) -> Node:
        tok = self.cur
        if tok.kind == "num":
            self.i += 1
            return Num(float(tok.text))
        if tok.kind == "str":
            self.i += 1
            return Str(bytes(tok.text[1:-1], "utf-8").decode("unicode_escape"))
        if tok.kind == "ident":
            self.i += 1
            low = tok.text.lower()
            if low == "nan":
                return Num(float("nan"))
            if low in ("true", "false"):
                return Num(1.0 if low == "true" else 0.0)
            if self.accept("("):
                return self.call(tok.text)
            return Ident(tok.text)
        if self.accept("("):
            inner = self.expr()
            self.eat(")")
            return inner
        got = repr(tok.text) if tok.text else "end of expression"
        raise ParseError(f"Expected a value but found {got}", tok.pos, self.src)

    def call(self, name: str) -> Node:
        args: list[Node] = []
        kwargs: list[tuple[str, Node]] = []
        if not self.accept(")"):
            while True:
                if self.cur.kind == "ident" and self.peek().kind == "op" and self.peek().text == "=":
                    key = self.eat(kind="ident").text
                    self.eat("=")
                    kwargs.append((key, self.expr()))
                else:
                    if kwargs:
                        raise ParseError("Positional argument after keyword argument", self.cur.pos, self.src)
                    args.append(self.expr())
                if self.accept(")"):
                    break
                self.eat(",")
        return Call(name, tuple(args), tuple(kwargs))


def parse(src: str) -> Node:
    return _Parser(_strip_comments(src)).program()
