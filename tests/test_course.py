"""The Learn2Quant lessons are ground truth: the rules engine must agree with what BRAIN showed."""
import math

import pytest

from alpha_bench.course import EXAMPLES, PRINCIPLES, by_id
from alpha_bench.engine.simulate import fitness
from alpha_bench.gen.reconcile import parse_pasted_metrics
from alpha_bench.rules.engine import Rules, evaluate
from alpha_bench.settings_model import SimSettings

SHOWN = [e for e in EXAMPLES if e.brain.get("fitness") is not None and e.brain.get("turnover") is not None]


@pytest.mark.parametrize("ex", SHOWN, ids=lambda e: e.id)
def test_fitness_formula_reproduces_brain(ex):
    b = ex.brain
    assert math.isclose(fitness(b["sharpe"], b["returns"], b["turnover"]), b["fitness"], abs_tol=0.04)


@pytest.mark.parametrize("ex", SHOWN, ids=lambda e: e.id)
def test_rules_agree_with_lesson_verdicts(ex, tmp_path):
    settings = SimSettings.from_dict(ex.settings)
    checks = {c.name: c for c in evaluate(dict(ex.brain), settings, Rules(tmp_path))}
    fails = {n for n, c in checks.items() if c.result == "FAIL"}
    if ex.verdict.startswith("Passes"):
        assert not fails & {"LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER"}
    else:
        assert fails & {"LOW_FITNESS", "HIGH_TURNOVER"}
    if ex.id == "l9-two-day-reversal":
        assert "HIGH_TURNOVER" in fails


def test_examples_round_trip_through_reconcile_parser():
    for ex in SHOWN:
        parsed = parse_pasted_metrics(ex.pasted_text())["metrics"]
        assert math.isclose(parsed["sharpe"], ex.brain["sharpe"], abs_tol=1e-9)
        assert math.isclose(parsed["turnover"], ex.brain["turnover"], abs_tol=1e-4)


def test_course_content_is_complete():
    assert by_id("l4-open-close").expression == "-group_rank(close - open, subindustry)"
    assert {p[0] for p in PRINCIPLES} >= {"Returns / drawdown", "Diversity", "Turnover buckets", "Drawdown"}
    for ex in EXAMPLES:
        assert ex.video.startswith("https://www.youtube.com/watch?v=")


def test_lesson_alphas_evaluate_offline_with_real_field_names(engine):
    for ex in EXAMPLES:
        r = engine.simulate(ex.expression, SimSettings.from_dict(ex.settings), quick=True)
        assert r.proxied == []  # every lesson field exists in the offline market
