import math

import numpy as np
import pytest

from alpha_bench.engine.evaluator import EvalError, Evaluator, cs_rank, decay_linear, _weighted
from alpha_bench.engine.simulate import compute_metrics, construct_weights, fitness, truncate
from alpha_bench.expr.parser import parse
from alpha_bench.settings_model import SimSettings


def ev(market, src, **kw):
    return Evaluator(market, **kw).evaluate(parse(src))


def test_delay_shifts_all_fields(market):
    close = np.asarray(market.fields["close"], dtype=float)
    d1 = ev(market, "close", delay=1, pasteurize=False)
    d0 = ev(market, "close", delay=0, pasteurize=False)
    assert np.allclose(d0, close, equal_nan=True)
    assert np.allclose(d1[1:], close[:-1], equal_nan=True) and np.isnan(d1[0]).all()


def test_pasteurization_masks_outside_universe(market):
    x = ev(market, "close", delay=0, universe="TOP500", pasteurize=True)
    assert np.isnan(x[~market.universe["TOP500"]]).all()


def test_rank_bounds_and_ties():
    x = np.array([[3.0, 1.0, 2.0, np.nan], [5.0, 5.0, 1.0, 2.0]])
    r = cs_rank(x)
    assert np.allclose(r[0, :3], [1.0, 0.0, 0.5]) and np.isnan(r[0, 3])
    assert r[1, 0] == r[1, 1]  # ties averaged


def test_group_neutralize_zero_mean(market):
    x = ev(market, "group_neutralize(rank(returns), industry)", delay=0, pasteurize=False)
    ind = np.asarray(market.groups["industry"][0]).astype(int)
    t = 300
    for g in np.unique(ind):
        vals = x[t, ind == g]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            assert abs(vals.mean()) < 1e-9


def test_ts_ops_match_reference(market):
    x = ev(market, "ts_mean(close, 5)", delay=0, pasteurize=False)
    close = np.asarray(market.fields["close"], dtype=float)
    assert np.allclose(x[100], np.nanmean(close[96:101], axis=0), equal_nan=True)
    r = ev(market, "ts_rank(close, 10)", delay=0, pasteurize=False)
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 1


def test_decay_linear_fast_and_slow_paths_agree():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 20))
    x[rng.random(x.shape) < 0.15] = np.nan
    for d in (3, 12, 30):
        a, b = decay_linear(x, d), _weighted(x, np.arange(1, d + 1, dtype=float))
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_truncation_caps_weight_and_keeps_gross():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(50, 80))
    w[:, 0] *= 40
    w /= np.abs(w).sum(1, keepdims=True)
    t = truncate(w, 0.05)
    assert np.abs(t).max() <= 0.05 + 1e-9
    assert np.allclose(np.abs(t).sum(1), 1.0)


def test_market_neutral_weights(market):
    alpha = ev(market, "rank(returns)", delay=1)
    w = construct_weights(alpha, SimSettings(neutralization="MARKET", truncation=0.1),
                          market.universe["TOP3000"], market.groups)
    live = np.abs(w).sum(1) > 0
    assert np.allclose(w[live].sum(1), 0, atol=1e-6)
    assert np.allclose(np.abs(w[live]).sum(1), 1, atol=1e-6)


def test_decay_lowers_turnover(engine):
    fast = engine.simulate("rank(-returns)", SimSettings(decay=0)).metrics["turnover"]
    slow = engine.simulate("rank(-returns)", SimSettings(decay=8)).metrics["turnover"]
    assert slow < fast * 0.7


def test_fitness_formula_and_floor():
    assert math.isclose(fitness(2.0, 0.10, 0.5), 2.0 * math.sqrt(0.10 / 0.5))
    assert fitness(2.0, 0.10, 0.05) == fitness(2.0, 0.10, 0.125)  # no credit below 12.5% turnover


def test_metrics_definitions():
    w = np.zeros((5, 2))
    w[:, 0], w[:, 1] = 0.5, -0.5
    rets = np.zeros((5, 2))
    rets[:, 0] = 0.01
    m = compute_metrics(w, rets, 1, 4)
    assert math.isclose(m["returns"], 0.005 * 20e6 * 252 / 10e6)
    assert m["turnover"] == 0.0 and m["max_weight"] == 0.5


def test_simulate_outputs(engine):
    r = engine.simulate("rank(equity / cap)", SimSettings(neutralization="SUBINDUSTRY"))
    for k in ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin", "max_weight", "sub_universe_sharpe"):
        assert k in r.metrics and np.isfinite(r.metrics[k])
    assert r.yearly and r.series["cum_pnl"] and r.exposures


def test_unknown_field_uses_flagged_proxy(engine):
    r = engine.simulate("rank(fnd6_somefield / cap)", SimSettings())
    assert r.proxied == ["fnd6_somefield"]


def test_eval_errors_are_friendly(engine):
    with pytest.raises(EvalError, match="constant"):
        engine.simulate("1 + 2", SimSettings())
    with pytest.raises(EvalError, match="lookback"):
        engine.simulate("ts_mean(close, volume)", SimSettings())
    with pytest.raises(EvalError, match="offline engine"):
        engine.simulate("made_up_operator(close)", SimSettings())


@pytest.mark.parametrize("src", [
    "ts_corr(rank(close), rank(volume), 10)", "ts_regression(close, ts_step(1), 20)", "group_rank(close, sector)",
    "group_zscore(returns, industry)", "quantile(returns)", "winsorize(zscore(returns), std=3)",
    "trade_when(volume > adv20, -returns, -1)", "ts_backfill(news_sentiment, 10)", "hump(rank(returns), hump=0.01)",
    "ts_decay_exp_window(returns, 10, factor=0.7)", "regression_neut(rank(returns), rank(cap))",
    "days_from_last_change(eps)", "ts_arg_max(close, 20)", "kth_element(news_sentiment, 10, k=2)",
    "group_median(returns, sector)", "ts_quantile(close, 20)", "bucket(rank(cap), range=\"0,1,0.2\")",
    "signed_power(rank(returns) - 0.5, 2)", "if_else(close > open, 1, -1) * rank(volume)",
    "group_backfill(eps, industry, 20)", "last_diff_value(eps, 90)", "ts_skewness(returns, 20)",
])
def test_operators_evaluate(market, src):
    out = ev(market, src, delay=1)
    assert out.shape == market.shape
    assert np.isfinite(out[np.isfinite(out)]).all()
    assert np.isfinite(out).any()


def test_truncation_is_scaled_to_nominal_universe(engine):
    r = engine.simulate("rank(-returns)", SimSettings(truncation=0.01, neutralization="MARKET"))
    m = r.metrics
    assert m["universe_scale"] > 1                        # synthetic TOP3000 is smaller than 3000 names
    assert m["max_weight"] <= 0.01 + 1e-9                  # reported in BRAIN-equivalent terms
    assert m["long_count"] + m["short_count"] > 1000       # BRAIN-equivalent position counts


def test_drawdown_is_largest_loss_within_a_year_and_ratio_reported(engine):
    m = engine.simulate("rank(-ts_sum(returns, 5))", SimSettings()).metrics
    assert m["drawdown"] <= m["drawdown_full"] + 1e-12
    assert m["returns_drawdown"] == pytest.approx(m["returns"] / m["drawdown"])
    assert m["turnover_bucket"].startswith(("low", "medium", "high"))


def test_test_period_split(engine):
    # the test fixture's market is short, so hold out 6 months rather than the 1-year default
    m = engine.simulate("rank(-returns)", SimSettings(test_period="P0Y6M0D")).metrics
    assert m["test_months"] == 6 and "train_sharpe" in m and "test_sharpe" in m
    assert "test_sharpe" not in engine.simulate("rank(-returns)", SimSettings(test_period="P0Y0M0D")).metrics
