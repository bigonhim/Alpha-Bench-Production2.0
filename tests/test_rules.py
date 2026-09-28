import json

from alpha_bench.gen.reconcile import parse_pasted_metrics
from alpha_bench.rules.engine import Rules, evaluate, summarize
from alpha_bench.settings_model import SimSettings

GOOD = {"sharpe": 1.8, "fitness": 1.3, "returns": 0.10, "turnover": 0.2, "drawdown": 0.04, "margin": 8.0,
        "max_weight": 0.03, "sub_universe_sharpe": 1.2, "sharpe_2y": 1.6, "coverage": 0.9}


def by_name(checks):
    return {c.name: c for c in checks}


def test_passing_alpha(rules):
    checks = evaluate(GOOD, SimSettings(), rules)
    assert summarize(checks)["passed"]
    assert by_name(checks)["LOW_SHARPE"].limit == ">= 1.25"


def test_delay0_is_stricter(rules):
    checks = by_name(evaluate({**GOOD, "sharpe": 1.8}, SimSettings(delay=0), rules))
    assert checks["LOW_SHARPE"].result == "FAIL" and "2.0" in checks["LOW_SHARPE"].limit


def test_turnover_band_and_fitness_explanation(rules):
    c = by_name(evaluate({**GOOD, "turnover": 0.85, "fitness": 0.6}, SimSettings(), rules))
    assert c["HIGH_TURNOVER"].result == "FAIL"
    assert "12.5%" in c["LOW_FITNESS"].why and c["LOW_FITNESS"].fixes
    c = by_name(evaluate({**GOOD, "turnover": 0.005}, SimSettings(), rules))
    assert c["LOW_TURNOVER"].result == "FAIL"


def test_negative_sharpe_suggests_flip(rules):
    c = by_name(evaluate({**GOOD, "sharpe": -1.5}, SimSettings(), rules))
    assert any("Flip" in f for f in c["LOW_SHARPE"].fixes)


def test_self_correlation_exception(rules):
    corr = {"max": 0.82, "with_name": "A1", "with_sharpe": 1.5, "source": "brain"}
    assert by_name(evaluate({**GOOD, "sharpe": 1.8}, SimSettings(), rules, correlation=corr))["SELF_CORRELATION"].result == "PASS"
    assert by_name(evaluate({**GOOD, "sharpe": 1.6}, SimSettings(), rules, correlation=corr))["SELF_CORRELATION"].result == "FAIL"


def test_sub_universe_scales_with_size(rules):
    c = by_name(evaluate({**GOOD, "sharpe": 2.0, "sub_universe_sharpe": 0.5}, SimSettings(universe="TOP3000"), rules))
    assert c["LOW_SUB_UNIVERSE_SHARPE"].result == "FAIL"
    assert "0.87" in c["LOW_SUB_UNIVERSE_SHARPE"].limit  # 0.75 * sqrt(1000/3000) * 2.0


def test_returns_drawdown_is_the_course_standard(tmp_path):
    rules = Rules(tmp_path)
    c = by_name(evaluate({**GOOD, "returns": 0.05, "drawdown": 0.08}, SimSettings(), rules))
    assert c["RETURNS_DRAWDOWN"].result == "WARN" and c["RETURNS_DRAWDOWN"].source == "Learn2Quant Lesson 3"
    assert "DRAWDOWN" not in c  # no absolute cap unless you set one
    data = json.loads((tmp_path / "rules.json").read_text())
    data["returns_drawdown"]["enforced"] = True
    data["drawdown"]["max"] = 0.05
    (tmp_path / "rules.json").write_text(json.dumps(data))
    rules.reload()
    c = by_name(evaluate({**GOOD, "returns": 0.05, "drawdown": 0.08}, SimSettings(), rules))
    assert c["RETURNS_DRAWDOWN"].result == "FAIL"
    assert c["DRAWDOWN"].result == "WARN" and c["DRAWDOWN"].source == "your rules file"


def test_threshold_precedence_and_regions(tmp_path):
    rules = Rules(tmp_path)
    assert rules.value("sharpe", 1) == (1.25, "default (public BRAIN criteria)")
    rules.learn_from_brain([{"name": "LOW_SHARPE", "limit": 2.07, "result": "FAIL"}], delay=1, region="CHN")
    assert rules.value("sharpe", 1, "CHN") == (2.07, "BRAIN check limit (CHN)")
    assert rules.value("sharpe", 1, "USA")[0] == 1.25  # learned limits are per region
    data = json.loads((tmp_path / "rules.json").read_text())
    data["sharpe"]["delay1"] = 1.4
    (tmp_path / "rules.json").write_text(json.dumps(data))
    rules.reload()
    assert rules.value("sharpe", 1, "CHN") == (1.4, "your rules file")
    assert rules.conflicts("CHN")


def test_risk_controls_add_checks_and_fixes(tmp_path, engine, catalog):
    from alpha_bench.expr.parser import parse
    from alpha_bench.gen.repair import assess, propose

    rules = Rules(tmp_path)
    assert "RISK_TURNOVER" not in by_name(evaluate(GOOD, SimSettings(), rules))  # off by default
    rules.set_risk({"max_turnover": 0.15, "max_weight": 0.02, "max_sector_net": 0.001})
    assert Rules(tmp_path).risk()["max_turnover"] == 0.15  # persisted in rules.json
    c = by_name(evaluate({**GOOD, "turnover": 0.3, "max_sector_net": 0.01}, SimSettings(), rules))
    assert c["RISK_TURNOVER"].result == "FAIL" and c["RISK_WEIGHT"].result == "FAIL"
    assert c["RISK_SECTOR"].result == "FAIL" and c["RISK_SECTOR"].source == "your risk controls"
    node, s = parse("rank(-returns)"), SimSettings(neutralization="NONE")
    base = assess(engine, rules, node, s)
    fixes = {f.change for f in propose(node, s, base["checks"], base["sim"].metrics, catalog)}
    assert any(ch.startswith("decay") for ch in fixes) and any("neutralization NONE -> SECTOR" == ch for ch in fixes)


def test_parse_pasted_metrics():
    text = """IS Summary
    Sharpe  Turnover  Fitness  Returns  Drawdown  Margin
    1.45    23.45%    1.12     8.61%    3.21%     7.35‱
    LOW_SUB_UNIVERSE_SHARPE  FAIL
    Self-correlation 0.81"""
    out = parse_pasted_metrics(text)
    m = out["metrics"]
    assert m["sharpe"] == 1.45 and abs(m["turnover"] - 0.2345) < 1e-9 and m["margin"] == 7.35
    assert abs(m["returns"] - 0.0861) < 1e-9 and m["self_correlation"] == 0.81
    assert {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"} in out["checks"]
    assert parse_pasted_metrics("Sharpe: 2.1, Fitness 1.3, Turnover 45%")["metrics"]["turnover"] == 0.45


def test_two_year_sharpe_is_enforced_and_test_period_warns(rules):
    c = by_name(evaluate({**GOOD, "sharpe_2y": 0.6}, SimSettings(), rules))
    assert c["LOW_2Y_SHARPE"].result == "FAIL"  # BRAIN's real LOW_2Y_SHARPE check
    c = by_name(evaluate({**GOOD, "test_sharpe": 0.4, "train_sharpe": 2.2, "test_months": 12},
                         SimSettings(), rules))
    assert c["LOW_TEST_SHARPE"].result == "WARN"
    assert "TRAIN" in c["LOW_TEST_SHARPE"].why


def test_brain_outlook_levels(rules):
    from alpha_bench.rules.engine import brain_outlook

    def outlook(m, calibrated=None):
        checks = [c.to_dict() for c in evaluate(m, SimSettings(), rules)]
        return brain_outlook(m, checks, rules, SimSettings(), calibrated=calibrated)

    assert outlook({**GOOD, "sharpe": 2.2, "fitness": 1.6})["level"] == "comfortable"
    o = outlook({**GOOD, "sharpe": 1.30, "fitness": 1.05})
    assert o["level"] == "borderline" and any("optimistic" in r for r in o["reasons"])
    assert any("only BRAIN can run" in r for r in o["reasons"])  # pending checks called out
    o = outlook({**GOOD, "sharpe": 2.2, "fitness": 1.6}, calibrated={"sharpe": {"value": 1.0, "n": 5}})
    assert o["level"] == "likely-fail" and o["calibration_n"] == 5
    assert outlook({**GOOD, "turnover": 0.85})["level"] == "fail"
