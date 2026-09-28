import numpy as np

from alpha_bench.expr.parser import parse
from alpha_bench.gen.generator import Generator
from alpha_bench.gen.optimize import sweep
from alpha_bench.gen.reconcile import Calibration, reconcile
from alpha_bench.gen.recreate import recreate
from alpha_bench.gen.repair import assess, auto_repair, propose
from alpha_bench.gen.scoring import OfflinePnlPool
from alpha_bench.settings_model import SimSettings
from alpha_bench.store.library import Library


def test_create_never_repeats_logic(engine, rules, catalog):
    gen = Generator(engine, rules, catalog, OfflinePnlPool(), seed=5)
    seen, theses = set(), []
    for _ in range(4):
        r = gen.create(budget_s=8, n=6)
        assert "error" not in r, r
        assert r["fingerprint"] not in seen
        seen.add(r["fingerprint"])
        theses.append(r["thesis_id"])
        assert r["rationale"]["why_it_works"] and r["settings"]["universe"]
        assert {c["name"] for c in r["checks"]} >= {"LOW_SHARPE", "LOW_FITNESS", "CONCENTRATED_WEIGHT"}
    assert len(set(theses)) >= 3  # varies the economic thesis, not just parameters


def test_create_respects_families_and_operators(engine, rules, catalog):
    gen = Generator(engine, rules, catalog, OfflinePnlPool(), seed=9)
    r = gen.create({"fundamental"}, budget_s=8, n=5)
    assert r["family"] == "fundamental"
    cands = gen.candidates({"price_volume"}, SimSettings(), n=6,
                           allowed_ops={"rank", "ts_sum", "ts_delta", "ts_mean", "ts_zscore", "ts_corr"})
    for c in cands:
        from alpha_bench.expr.fingerprint import operator_set
        assert operator_set(c.node) <= {"rank", "ts_sum", "ts_delta", "ts_mean", "ts_zscore", "ts_corr"}


def test_repair_flips_inverted_alpha(engine, rules, catalog):
    node = parse("rank(ts_sum(returns, 5))")  # reversal market: this is the wrong way round
    base = assess(engine, rules, node, SimSettings(neutralization="MARKET"))
    assert base["sim"].metrics["sharpe"] < 0
    fixes = propose(node, SimSettings(neutralization="MARKET"), base["checks"], base["sim"].metrics, catalog)
    assert fixes[0].title == "Flip the sign"
    fixed = auto_repair(engine, rules, node, SimSettings(neutralization="MARKET"), catalog)
    assert fixed["assessment"]["sim"].metrics["sharpe"] > 0
    assert fixed["history"][0]["change"] == "expression x -1"


def test_high_turnover_fix_is_minimal(engine, rules, catalog):
    node = parse("rank(-returns)")
    s = SimSettings(neutralization="MARKET", decay=0)
    base = assess(engine, rules, node, s)
    fixes = propose(node, s, base["checks"], base["sim"].metrics, catalog)
    assert any(f.change.startswith("decay 0 ->") and f.size == 1 for f in fixes)


def test_recreate_modify_and_rebuild(engine, rules, catalog):
    gen = Generator(engine, rules, catalog, OfflinePnlPool(), seed=2)
    m = recreate(engine, rules, catalog, gen, "rank(ts_sum(returns, 5))", SimSettings(), "modify")
    assert m["structure"]["family"] == "price_volume" and m["result"]["expression"]
    assert m["options"]
    r = recreate(engine, rules, catalog, gen, "rank(equity / cap)", SimSettings(neutralization="SUBINDUSTRY"),
                 "rebuild")
    assert "error" not in r
    assert r["result"]["expression"] != "rank(equity / cap)"
    assert -1 <= r["correlation_with_original"] <= 1


def test_sweep(engine, rules):
    out = sweep(engine, rules, parse("rank(-ts_sum(returns, 5))"), SimSettings(),
                {"decay": [0, 4], "truncation": [0.05, 0.08], "neutralization": ["MARKET", "INDUSTRY"]})
    assert out["runs"] == 8 and out["best"]["score"] >= out["rows"][-1]["score"]
    assert any(r["pareto"] for r in out["rows"])


def test_reconcile_localises_and_calibrates(engine, rules, catalog):
    brain_m = {"sharpe": 1.1, "fitness": 0.7, "turnover": 0.78, "returns": 0.07}
    brain_checks = [{"name": "HIGH_TURNOVER", "result": "FAIL", "limit": 0.7, "value": 0.78},
                    {"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.25, "value": 1.1},
                    {"name": "CONCENTRATED_WEIGHT", "result": "PASS", "limit": 0.1, "value": 0.02}]
    out = reconcile(engine, rules, catalog, expression="rank(-returns)", settings=SimSettings(),
                    brain_metrics=brain_m, brain_checks=brain_checks)
    assert set(out["failing"]) == {"HIGH_TURNOVER", "LOW_SHARPE"}
    assert any("HIGH_TURNOVER: BRAIN measured 78.0%" in d for d in out["diagnosis"])
    assert out["fixes"] and any("HIGH_TURNOVER" in f["predicted_clears"] for f in out["fixes"])
    assert rules.value("turnover.max")[1].startswith("BRAIN")  # learned BRAIN's limit
    rows = [{"metric": "sharpe", "offline": 2.0, "brain": 1.5, "family": "x"} for _ in range(4)]
    cal = Calibration(rows).adjust({"sharpe": 2.4})
    assert abs(cal["sharpe"]["value"] - 1.9) < 1e-9


def test_library_roundtrip(tmp_path):
    lib = Library(tmp_path / "lib.db")
    a = lib.add_alpha(expression="rank(close)", settings={"delay": 1}, mode="create", fingerprint="f1",
                      offline={"sharpe": 1.2})
    b = lib.add_alpha(expression="rank(-close)", settings={"delay": 1}, mode="fix", parent_id=a)
    lib.update_alpha(b, brain_alpha_id="XYZ1234", brain={"sharpe": 1.4}, status="verified")
    assert lib.get(b)["brain"]["sharpe"] == 1.4
    assert lib.lineage(b)["ancestors"][0]["id"] == a
    assert lib.lineage(a)["descendants"][0]["id"] == b
    exp = lib.export(b)
    assert "password" not in str(exp).lower() and exp["expression"] == "rank(-close)"
    lib.save_job({"id": "j1", "library_id": b, "expression": "x", "settings": {}, "state": "running",
                  "progress_url": "https://api.worldquantbrain.com/simulations/1", "created_at": 1, "updated_at": 1})
    assert lib.unfinished_jobs()[0]["progress_url"].endswith("/1")


def test_offline_pool_correlation():
    pool = OfflinePnlPool()
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    pool.add("a", "A", x, 1.5, "active")
    pool.add("b", "B", rng.normal(size=600), 1.2, "active")
    out = pool.max_corr(x * 2 + rng.normal(scale=0.01, size=600))
    assert out["with_id"] == "a" and out["max"] > 0.99


def test_lesson_theses_build_and_run(engine, catalog):
    from collections import Counter

    from alpha_bench.gen.fields import build_pool
    from alpha_bench.gen.generator import _dress
    from alpha_bench.gen.ideas import THESES, Ctx

    lesson = [t for t in THESES if "Learn2Quant" in t.title]
    assert len(lesson) >= 8
    rng = np.random.default_rng(1)
    for th in lesson:
        ctx = Ctx(rng, build_pool(catalog), Counter())
        core, note = th.build(ctx)
        node, _ = _dress(ctx, core, th)
        r = engine.simulate(node, SimSettings(), quick=True)
        assert np.isfinite(r.metrics["sharpe"]), th.id
