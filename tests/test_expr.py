import pytest

from alpha_bench.expr.ast import Call, Program, idents, inline, to_str
from alpha_bench.expr.classify import classify
from alpha_bench.expr.fingerprint import logic_fingerprint, shape_fingerprint
from alpha_bench.expr.parser import ParseError, parse
from alpha_bench.engine.evaluator import FieldCatalog


@pytest.mark.parametrize("src", [
    "rank(close)",
    "-ts_delta(close, 5)",
    "group_neutralize(rank(sales / assets), industry)",
    "ts_decay_exp_window(returns, 10, factor=0.5)",
    'group_rank(x, bucket(rank(cap), range="0.1,1,0.1"))',
    "close > open ? rank(volume) : -rank(volume)",
    "trade_when(volume > 2 * adv20, -returns, -1)",
    "a - b - c",
    "a / (b * c)",
    "2 ^ 3 ^ 2",
])
def test_roundtrip(src):
    tree = parse(src)
    assert parse(to_str(tree)) == tree


def test_precedence_and_unary():
    assert to_str(parse("-a * b + c")) == "-a * b + c"
    assert to_str(parse("(a + b) * c")) == "(a + b) * c"
    assert to_str(parse("a - (b - c)")) == "a - (b - c)"
    assert parse("-5").value == -5


def test_multi_statement_and_comments():
    tree = parse("""
        # momentum with a skip
        m = ts_sum(returns, 252) - ts_sum(returns, 21);   // long minus short
        rank(m)
    """)
    assert isinstance(tree, Program)
    assert to_str(inline(tree)) == "rank(ts_sum(returns, 252) - ts_sum(returns, 21))"
    assert idents(tree) == {"returns"}


def test_keyword_args_and_strings():
    tree = parse('bucket(rank(cap), range="0,1,0.1")')
    assert isinstance(tree, Call) and tree.kw("range").value == "0,1,0.1"


@pytest.mark.parametrize("src,msg", [("rank(close", "Expected"), ("rank(close))", "Unexpected"),
                                     ("", "Empty"), ("close $ open", "Unexpected character"),
                                     ("f(a=1, 2)", "Positional argument after keyword")])
def test_parse_errors_have_positions(src, msg):
    with pytest.raises(ParseError) as err:
        parse(src)
    assert msg in str(err.value)
    assert "column" in str(err.value)


def test_fingerprint_buckets_windows_and_commutes():
    assert logic_fingerprint(parse("ts_mean(close, 20)")) == logic_fingerprint(parse("ts_mean(close, 21)"))
    assert logic_fingerprint(parse("ts_mean(close, 20)")) != logic_fingerprint(parse("ts_mean(close, 5)"))
    assert logic_fingerprint(parse("rank(a) + rank(b)")) == logic_fingerprint(parse("rank(b) + rank(a)"))
    cat = FieldCatalog().category
    assert shape_fingerprint(parse("rank(sales/assets)"), cat) == shape_fingerprint(parse("rank(income/equity)"), cat)


def test_classify():
    cat = FieldCatalog().category
    s = classify(parse("-ts_sum(returns, 5)"), cat)
    assert s.family == "price_volume" and s.thesis == "short-term reversal" and s.horizon == "short"
    s = classify(parse("rank(equity / cap)"), cat)
    assert s.family == "fundamental" and "value" in s.thesis
    s = classify(parse("trade_when(volume > adv20, -returns, -1)"), cat)
    assert s.family == "event_driven"
