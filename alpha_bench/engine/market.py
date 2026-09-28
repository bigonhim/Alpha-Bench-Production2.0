"""Synthetic equity market used by the offline simulation engine.

BRAIN's data can't be downloaded, so offline screening runs on a synthetic
market that reproduces the *mechanics* that decide submittability: realistic
return factor structure, sector/industry/subindustry groups, cap-ranked
universes with churn, quarterly-updating fundamentals, patchy coverage, and
a set of weak, regime-varying anomalies (reversal, momentum, value, quality,
accruals, analyst revisions, sentiment, short interest, low volatility).

What transfers well to BRAIN: turnover, weight concentration, coverage and
NaN behaviour, neutralization/decay/truncation effects, relative ranking of
structurally similar ideas. What does NOT transfer: absolute Sharpe/returns,
because the anomalies here are stylised, not real. The reconciliation log
calibrates offline numbers against real BRAIN results over time.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GENERATOR_VERSION = 9
GROUP_NAMES = ("market", "sector", "industry", "subindustry", "exchange", "country")
UNIVERSE_SHARE = {"TOP3000": 1.0, "TOP1000": 1 / 3, "TOP500": 1 / 6, "TOPSP500": 1 / 6, "TOP200": 1 / 15}

# Field -> (category, role). Roles drive the idea generator.
FIELD_INFO: dict[str, tuple[str, str]] = {
    "open": ("pv", "price"), "high": ("pv", "price"), "low": ("pv", "price"), "close": ("pv", "price"),
    "vwap": ("pv", "price"), "returns": ("pv", "returns"), "volume": ("pv", "volume"),
    "adv20": ("pv", "volume"), "cap": ("pv", "size"), "sharesout": ("pv", "size"),
    "assets": ("fundamental", "balance"), "liabilities": ("fundamental", "balance"),
    "equity": ("fundamental", "balance"), "debt": ("fundamental", "balance"), "cash": ("fundamental", "balance"),
    "sales": ("fundamental", "flow"), "operating_income": ("fundamental", "flow"), "income": ("fundamental", "flow"),
    "cashflow_op": ("fundamental", "flow"), "ebitda": ("fundamental", "flow"), "capex": ("fundamental", "flow"),
    "eps": ("fundamental", "per_share"), "bookvalue_ps": ("fundamental", "per_share"),
    "enterprise_value": ("fundamental", "valuation"), "retained_earnings": ("fundamental", "balance"),
    "current_ratio": ("fundamental", "ratio"), "sales_growth": ("fundamental", "ratio"),
    "est_eps": ("analyst", "estimate"), "est_sales": ("analyst", "estimate"), "est_ptg": ("analyst", "estimate"),
    "num_analysts": ("analyst", "coverage"),
    "news_sentiment": ("sentiment", "sentiment"), "news_count": ("sentiment", "attention"),
    "social_buzz": ("sentiment", "attention"),
    "implied_volatility": ("option", "volatility"), "put_call_ratio": ("option", "positioning"),
    "short_interest": ("other", "positioning"),
    # Real BRAIN field ids used in the Learn2Quant lessons (sentiment buzz, option implied volatility
    # at 120/720-day horizons, Parkinson high-low volatility).
    "snt_buzz": ("sentiment", "attention"),
    "implied_volatility_call_120": ("option", "volatility"), "implied_volatility_put_120": ("option", "volatility"),
    "implied_volatility_call_720": ("option", "volatility"), "implied_volatility_put_720": ("option", "volatility"),
    "parkinson_volatility_120": ("option", "volatility"),
}


@dataclass
class Market:
    dates: np.ndarray                     # (T,) datetime64[D]
    fields: dict[str, np.ndarray]         # (T, N) float64
    groups: dict[str, np.ndarray]         # (T, N) float64 integer codes
    universe: dict[str, np.ndarray]       # (T, N) bool membership
    latents: dict[str, np.ndarray]        # hidden drivers, used only to synthesise proxy fields
    warmup: int                           # first in-sample day
    seed: int
    proxies: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.fields["close"].shape

    @property
    def returns(self) -> np.ndarray:
        return self.fields["returns"]


def _zs(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=-1, keepdims=True)
    s = np.nanstd(x, axis=-1, keepdims=True)
    return np.where(s > 0, (x - m) / np.where(s > 0, s, 1), 0.0)


def _ar1(rng, T, N, phi, sd=1.0, init=None):
    out = np.empty((T, N))
    innov = rng.standard_normal((T, N)) * sd * np.sqrt(1 - phi ** 2)
    out[0] = rng.standard_normal(N) * sd if init is None else init
    for t in range(1, T):
        out[t] = phi * out[t - 1] + innov[t]
    return out


def _quarterly_steps(rng, values: np.ndarray, lag: int = 45) -> np.ndarray:
    """Sample a latent at staggered quarterly report dates and hold it (point-in-time)."""
    T, N = values.shape
    offset = rng.integers(0, 63, N)
    out = np.full((T, N), np.nan)
    for i in range(N):
        reports = np.arange(offset[i], T, 63)
        for k, r in enumerate(reports):
            avail = r + lag
            if avail >= T:
                break
            end = reports[k + 1] + lag if k + 1 < len(reports) else T
            out[avail:min(end, T), i] = values[r, i]
    return out


def generate(n_stocks: int = 600, n_days: int = 1260, warmup: int = 260, seed: int = 7) -> Market:
    rng = np.random.default_rng(seed)
    T, N = n_days + warmup, n_stocks

    # --- groups ---------------------------------------------------------
    n_sectors = 11
    sector = rng.integers(0, n_sectors, N)
    industry = np.empty(N, dtype=int)
    subindustry = np.empty(N, dtype=int)
    ind_base = 0
    for s in range(n_sectors):
        members = np.where(sector == s)[0]
        k = rng.integers(3, 6)
        ind = rng.integers(0, k, len(members))
        industry[members] = ind_base + ind
        subindustry[members] = (ind_base + ind) * 2 + rng.integers(0, 2, len(members))
        ind_base += k
    n_ind = ind_base
    exchange = rng.choice(3, N, p=[0.45, 0.45, 0.10])

    # --- static characteristics ------------------------------------------
    log_cap0 = rng.normal(8.0, 1.3, N)                          # $M, lognormal cap
    size_z = (log_cap0 - log_cap0.mean()) / log_cap0.std()
    idio_vol = np.clip(0.021 - 0.005 * size_z + rng.normal(0, 0.004, N), 0.009, 0.045)
    beta = np.clip(rng.normal(1.0, 0.3, N), 0.3, 2.0)
    listing = np.where(rng.random(N) < 0.06, rng.integers(warmup // 2, T // 2, N), 0)

    # --- latent drivers (hidden) ------------------------------------------
    lat = {
        "value": _ar1(rng, T, N, 0.998),
        "quality": _ar1(rng, T, N, 0.999),
        "growth": _ar1(rng, T, N, 0.996),
        "accrual": _ar1(rng, T, N, 0.995),
        "revision": _ar1(rng, T, N, 0.96),
        "sentiment": _ar1(rng, T, N, 0.75),
        "informed": _ar1(rng, T, N, 0.985),
    }
    # Regimes: each anomaly's strength drifts slowly, so years differ and drawdowns happen.
    def regime(period, phase_amp=0.55):
        t = np.arange(T)
        return 1 + phase_amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.3)) + 0.15 * _ar1(rng, T, 1, 0.99)[:, 0]

    kappa = {  # daily expected-return loading per unit z-score (1bp = 1e-4)
        "value": 1.1e-4 * regime(900), "quality": 1.0e-4 * regime(1100), "accrual": 0.8e-4 * regime(800),
        "revision": 1.5e-4 * regime(600), "sentiment": 1.6e-4 * regime(400), "informed": 1.2e-4 * regime(700),
        "reversal": 3.2e-4 * regime(500), "momentum": 0.9e-4 * regime(650), "lowvol": 0.6e-4 * regime(1000),
        "volrev": 1.0e-4 * regime(450), "volmom": 1.4e-4 * regime(700),
    }
    # One sharp momentum crash, as in real markets.
    crash = int(T * 0.72)
    kappa["momentum"][crash:crash + 40] *= -4.0

    static_mu = (kappa["lowvol"][:, None] * -_zs(idio_vol[None, :]))
    slow_mu = (kappa["value"][:, None] * _zs(lat["value"]) + kappa["quality"][:, None] * _zs(lat["quality"])
               - kappa["accrual"][:, None] * _zs(lat["accrual"]) + kappa["revision"][:, None] * _zs(lat["revision"])
               + kappa["informed"][:, None] * _zs(lat["informed"]))
    # Sentiment is known the evening before; it predicts the next day's move.
    sent_mu = np.vstack([np.zeros((1, N)), kappa["sentiment"][1:, None] * _zs(lat["sentiment"][:-1])])

    # --- returns: factor structure + anomalies (sequential for path-dependent ones)
    mkt_vol = 0.010 * np.exp(0.35 * _ar1(rng, T, 1, 0.985)[:, 0])
    mkt = rng.normal(0.0003, 1, T) * mkt_vol
    sect_f = rng.normal(0, 0.0065, (T, n_sectors))
    ind_f = rng.normal(0, 0.0045, (T, n_ind))
    noise = rng.standard_normal((T, N)) * idio_vol
    vol_shock_lat = _ar1(rng, T, N, 0.55, sd=0.45)
    # Slow-moving participation level: rising interest (last week vs last year) precedes gains.
    vol_slow = _ar1(rng, T, N, 0.985, sd=0.45)
    vs_cum = np.cumsum(vol_slow, axis=0)

    rets = np.zeros((T, N))
    idio = np.zeros((T, N))
    sq_cum = np.zeros((T + 1, N))  # running sum of squared idiosyncratic returns
    vol_shock = np.zeros((T, N))
    for t in range(T):
        mu = slow_mu[t] + sent_mu[t] + static_mu[t]
        if t >= 6:
            short = idio[t - 5:t].sum(axis=0)
            rev = _zs(short[None, :])[0]
            if t >= 121:
                # Reversal pays mostly when a stock is in a high-volatility regime (Learn2Quant L4's
                # suggested improvement relies on this): scale by recent vs longer-run volatility.
                v20 = np.sqrt((sq_cum[t] - sq_cum[t - 20]) / 20)
                v120 = np.sqrt((sq_cum[t] - sq_cum[t - 120]) / 120)
                boost = np.clip(v20 / np.maximum(v120, 1e-6), 0.4, 3.0) ** 2
                rev = rev * boost / boost.mean()
            mu = mu - kappa["reversal"][t] * rev
            mu = mu - kappa["volrev"][t] * _zs((idio[t - 1] * np.clip(vol_shock[t - 1], 0, None))[None, :])[0]
        if t >= 241:
            recent_vol = (vs_cum[t - 1] - vs_cum[t - 6]) / 5
            year_vol = (vs_cum[t - 1] - vs_cum[t - 241]) / 240
            mu = mu + kappa["volmom"][t] * _zs((recent_vol - year_vol)[None, :])[0]
        if t >= 253:
            mom = idio[t - 252:t - 21].sum(axis=0)
            mu = mu + kappa["momentum"][t] * _zs(mom[None, :])[0]
        idio[t] = mu + noise[t]
        sq_cum[t + 1] = sq_cum[t] + idio[t] ** 2
        rets[t] = beta * mkt[t] + sect_f[t, sector] + ind_f[t, industry] + idio[t]
        vol_shock[t] = vol_shock_lat[t] + 1.2 * np.abs(noise[t]) / idio_vol
    rets = np.clip(rets, -0.6, 1.5)

    # --- prices and volume ------------------------------------------------
    p0 = np.exp(rng.normal(3.4, 0.8, N))
    close = p0 * np.cumprod(1 + rets, axis=0)
    prev_close = np.vstack([p0[None, :], close[:-1]])
    open_ = prev_close * (1 + 0.35 * rets + rng.normal(0, 0.003, (T, N)))
    # Intraday range scales with each stock's volatility (so Parkinson volatility is informative).
    hi_pad = np.abs(rng.normal(0, 1, (T, N))) * 0.35 * idio_vol
    lo_pad = np.abs(rng.normal(0, 1, (T, N))) * 0.35 * idio_vol
    high = np.maximum(open_, close) * (1 + hi_pad)
    low = np.minimum(open_, close) * (1 - lo_pad)
    vwap = (high + low + 2 * close) / 4 * (1 + rng.normal(0, 0.001, (T, N)))
    shares = np.exp(log_cap0) * 1e6 / p0
    shares_path = shares * np.cumprod(1 + np.where(rng.random((T, N)) < 0.004, rng.normal(0, 0.02, (T, N)), 0), axis=0)
    cap = close * shares_path / 1e6
    turnover_rate = np.exp(rng.normal(-5.2, 0.4, N))
    volume = shares_path * turnover_rate * np.exp(0.6 * vol_shock + vol_slow - 0.5)
    # adv20 = 20-day average daily volume (shares), so volume / adv20 is a volume-surge ratio.
    adv20 = np.full((T, N), np.nan)
    cs = np.cumsum(volume, axis=0)
    adv20[20:] = (cs[20:] - cs[:-20]) / 20

    # --- fundamentals (point-in-time, quarterly, ~45-day reporting lag) ----
    lev = np.clip(rng.normal(1.2, 0.5, N), 0.2, 4)
    bm = np.exp(0.45 * lat["value"] - 0.6 + rng.normal(0, 0.12, (T, N)))
    equity_q = _quarterly_steps(rng, cap * bm)
    assets_q = equity_q * (1 + lev)
    sales_q = assets_q * np.exp(rng.normal(-0.3, 0.3, N)) * np.exp(0.08 * lat["growth"])
    margin = 0.09 + 0.045 * lat["quality"] + rng.normal(0, 0.01, (T, N))
    op_inc_q = sales_q * _quarterly_steps(rng, margin)
    income_q = op_inc_q * 0.72
    acc_ratio = 0.035 * lat["accrual"] + rng.normal(0, 0.01, (T, N))
    cfo_q = income_q - assets_q * _quarterly_steps(rng, acc_ratio)
    debt_q = (assets_q - equity_q) * 0.6
    cash_q = assets_q * np.clip(rng.normal(0.1, 0.04, N), 0.01, 0.4)
    growth_q = _quarterly_steps(rng, 0.06 + 0.05 * lat["growth"] + rng.normal(0, 0.03, (T, N)))
    missing_fund = rng.random(N) < 0.04

    def fund(x):
        x = x.copy()
        x[:, missing_fund] = np.nan
        return x

    fields = {
        "open": open_, "high": high, "low": low, "close": close, "vwap": vwap, "returns": rets,
        "volume": volume, "adv20": adv20, "cap": cap, "sharesout": shares_path / 1e6,
        "equity": fund(equity_q), "assets": fund(assets_q), "liabilities": fund(assets_q - equity_q),
        "debt": fund(debt_q), "cash": fund(cash_q), "sales": fund(sales_q), "operating_income": fund(op_inc_q),
        "income": fund(income_q), "cashflow_op": fund(cfo_q), "ebitda": fund(op_inc_q * 1.25),
        "capex": fund(sales_q * 0.06), "retained_earnings": fund(equity_q * 0.4),
        "eps": fund(income_q / (shares_path / 1e6)), "bookvalue_ps": fund(equity_q / (shares_path / 1e6)),
        "enterprise_value": fund(cap + debt_q - cash_q),
        "current_ratio": fund(_quarterly_steps(rng, 1.5 + 0.3 * lat["quality"] + rng.normal(0, 0.2, (T, N)))),
        "sales_growth": fund(growth_q),
    }

    # --- analyst estimates: revisions follow the revision latent ---------------
    small = size_z < np.quantile(size_z, 0.3)
    rev_step = 0.0025 * lat["revision"] + rng.normal(0, 0.004, (T, N))
    base_eps = np.exp(rng.normal(0.5, 0.6, N)) * np.where(rng.random(N) < 0.12, -0.5, 1.0)
    est_eps = base_eps[None, :] * np.cumprod(1 + rev_step, axis=0)
    est_eps[:, small] = np.nan
    fields["est_eps"] = est_eps
    base_sales = np.exp(log_cap0) * np.exp(rng.normal(-0.2, 0.4, N))
    fields["est_sales"] = np.where(np.isnan(est_eps), np.nan, base_sales[None, :] *
                                   np.cumprod(1 + 0.6 * rev_step + rng.normal(0, 0.002, (T, N)), axis=0))
    fields["est_ptg"] = np.where(np.isnan(est_eps), np.nan,
                                 close * np.exp(0.15 + 0.08 * lat["revision"] + rng.normal(0, 0.05, (T, N))))
    fields["num_analysts"] = np.where(np.isnan(est_eps), np.nan,
                                      np.round(np.clip(3 + 2.5 * size_z + rng.normal(0, 1, N), 1, 40)))[..., :] \
        * np.ones((T, 1))

    # --- sentiment / news: sparse, noisy readings of the sentiment latent -------
    has_news = rng.random((T, N)) < np.clip(0.25 + 0.15 * size_z, 0.05, 0.8)
    fields["news_sentiment"] = np.where(has_news, lat["sentiment"] + rng.normal(0, 0.9, (T, N)), np.nan)
    fields["news_count"] = np.where(has_news, rng.poisson(2 + np.abs(lat["sentiment"]) * 2), 0).astype(float)
    fields["social_buzz"] = np.exp(0.3 * np.abs(lat["sentiment"]) + 0.2 * vol_shock + rng.normal(0, 0.4, (T, N)))

    # --- options / positioning ----------------------------------------------
    rv = np.sqrt(np.maximum(_rolling_mean(rets ** 2, 20), 1e-8)) * np.sqrt(252)
    fields["implied_volatility"] = rv * np.exp(0.08 + rng.normal(0, 0.08, (T, N))) + 0.02
    fields["put_call_ratio"] = np.exp(-0.25 * lat["sentiment"] - 0.15 * lat["informed"] + rng.normal(0, 0.3, (T, N)))
    fields["short_interest"] = np.clip(0.04 - 0.02 * lat["informed"] + rng.normal(0, 0.012, (T, N)), 0.001, None)

    # --- Learn2Quant lesson fields ---------------------------------------------
    # snt_buzz: investor attention. It spikes after recent outsized moves and on sentiment extremes, so a
    # buzz surge flags overreaction that tends to revert (Lesson 5: rising buzz -> short).
    recent = np.vstack([np.zeros((5, N)), np.cumsum(idio, axis=0)[5:] - np.cumsum(idio, axis=0)[:-5]])
    fields["snt_buzz"] = np.where(has_news | (rng.random((T, N)) < 0.5),
                                  np.exp(0.3 * np.abs(lat["sentiment"]) + 0.7 * _zs(recent) + 0.2 * vol_shock
                                         + 0.4 * size_z + rng.normal(0, 0.25, (T, N))), np.nan)
    # Parkinson volatility from the daily high-low range, annualised, 120-day window.
    park_var = np.log(high / low) ** 2 / (4 * np.log(2))
    fields["parkinson_volatility_120"] = np.sqrt(np.maximum(_rolling_mean(park_var, 120), 1e-10) * 252)
    rv120 = np.sqrt(np.maximum(_rolling_mean(rets ** 2, 120), 1e-8) * 252)
    rv252 = np.sqrt(np.maximum(_rolling_mean(rets ** 2, 252), 1e-8) * 252)
    # Option demand: informed call buying precedes gains, put buying precedes losses (Lessons 5 and 7).
    call_demand = 0.9 * lat["informed"] + 0.3 * lat["revision"] + _ar1(rng, T, N, 0.9, sd=0.6)
    put_demand = -0.9 * lat["informed"] - 0.2 * lat["sentiment"] + _ar1(rng, T, N, 0.9, sd=0.6)
    long_call = 0.5 * lat["informed"] + 0.3 * lat["quality"] + _ar1(rng, T, N, 0.97, sd=0.7)
    long_put = -0.5 * lat["informed"] - 0.2 * lat["value"] + _ar1(rng, T, N, 0.97, sd=0.7)
    has_opts = size_z > np.quantile(size_z, 0.2)  # small caps have no listed options
    for name, base, demand, bump in (("implied_volatility_call_120", rv120, call_demand, 0.06),
                                     ("implied_volatility_put_120", rv120, put_demand, 0.08),
                                     ("implied_volatility_call_720", rv252, long_call, 0.05),
                                     ("implied_volatility_put_720", rv252, long_put, 0.07)):
        quote_noise = rng.normal(0, 0.035, (T, N))
        iv = base * np.exp(bump + 0.10 * demand + quote_noise) + 0.02
        fields[name] = np.where(has_opts[None, :], iv, np.nan)

    # --- listing gaps -------------------------------------------------------
    for i in np.where(listing > 0)[0]:
        for arr in fields.values():
            arr[:listing[i], i] = np.nan

    # --- universes: rank by 20d-average cap, rebalanced monthly -------------------
    universe = {}
    cap_avg = _rolling_mean(np.nan_to_num(cap, nan=0.0), 20)
    rank_desc = np.full((T, N), N, dtype=float)
    for t in range(0, T, 21):
        order = np.argsort(-np.nan_to_num(cap_avg[t], nan=0.0))
        r = np.empty(N)
        r[order] = np.arange(N)
        rank_desc[t:t + 21] = r
    listed = ~np.isnan(close)
    for name, share in UNIVERSE_SHARE.items():
        universe[name] = (rank_desc < max(10, int(round(N * share)))) & listed

    group_rows = {"market": np.zeros(N), "sector": sector, "industry": industry, "subindustry": subindustry,
                  "exchange": exchange, "country": np.zeros(N)}
    dates = np.busday_offset(np.datetime64("2018-01-02"), np.arange(T), roll="forward")
    lat["size"] = np.broadcast_to(size_z[None, :], (T, N))
    lat["vol_shock"] = vol_shock
    # float32 storage keeps the whole market ~160MB; evaluation upcasts to float64.
    return Market(dates=dates, fields={k: v.astype(np.float32) for k, v in fields.items()},
                  groups=_broadcast_groups(group_rows, T), universe=universe,
                  latents={k: np.asarray(v, dtype=np.float32) for k, v in lat.items()},
                  warmup=warmup, seed=seed)


def _broadcast_groups(rows: dict[str, np.ndarray], T: int) -> dict[str, np.ndarray]:
    return {k: np.broadcast_to(np.asarray(v, dtype=np.float64)[None, :], (T, len(v))) for k, v in rows.items()}


def _rolling_mean(x: np.ndarray, d: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.nan_to_num(x), axis=0)
    out[d:] = (c[d:] - c[:-d]) / d
    return out


def load_or_generate(cache_dir: Path, n_stocks: int, n_days: int, seed: int) -> Market:
    key = f"v{GENERATOR_VERSION}_n{n_stocks}_d{n_days}_s{seed}"
    path = cache_dir / f"market_{key}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                return _unpack(z, seed)
        except Exception:  # noqa: BLE001 - stale or corrupt cache: rebuild
            log.info("Rebuilding synthetic market cache")
    market = generate(n_stocks, n_days, seed=seed)
    for old in cache_dir.glob("market_*.npz"):  # each cache is ~170MB; keep only the current one
        try:
            old.unlink()
        except OSError:
            pass
    try:
        packed = {"dates": market.dates.astype("datetime64[D]").astype("int64"), "warmup": np.array(market.warmup)}
        packed.update({f"f_{k}": v for k, v in market.fields.items()})
        packed.update({f"g_{k}": np.asarray(v[0]) for k, v in market.groups.items()})
        packed.update({f"u_{k}": v for k, v in market.universe.items()})
        packed.update({f"l_{k}": v for k, v in market.latents.items()})
        np.savez(path, **packed)
    except OSError:
        pass
    return market


def _unpack(z, seed: int) -> Market:
    def pick(prefix):
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}

    dates = z["dates"].astype("datetime64[D]")
    return Market(dates=dates, fields=pick("f_"), groups=_broadcast_groups(pick("g_"), len(dates)),
                  universe={k: v.astype(bool) for k, v in pick("u_").items()}, latents=pick("l_"),
                  warmup=int(z["warmup"]), seed=seed)


# ---------------------------------------------------------------- proxies
_PROXY_MIX = {
    # category -> latent loadings drawn per field (sign and weight vary by field hash)
    "fundamental": ("value", "quality", "growth", "accrual", "size"),
    "analyst": ("revision", "quality", "growth"),
    "sentiment": ("sentiment",), "news": ("sentiment",), "socialmedia": ("sentiment",),
    "option": ("informed", "sentiment"), "model": ("value", "quality", "revision", "informed"),
    "risk": ("size", "value"), "other": ("informed", "value"), "pv": ("vol_shock",),
    "earnings": ("revision", "quality"), "shortinterest": ("informed",), "insiders": ("informed",),
    "institutions": ("informed", "quality"), "macro": ("size",), "imbalance": ("sentiment", "vol_shock"),
}


def proxy_field(market: Market, name: str, category: str | None) -> np.ndarray:
    """Deterministic stand-in for a BRAIN field that has no synthetic counterpart.

    The field's dataset category decides which hidden drivers it loads on and how
    it updates (quarterly for fundamentals, daily and sparse for sentiment...).
    Results using proxies are flagged as lower-fidelity in the UI.
    """
    if name in market.proxies:
        return market.proxies[name]
    cat = (category or "other").lower().replace(" ", "")
    drivers = _PROXY_MIX.get(cat, _PROXY_MIX["other"])
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h ^ market.seed)
    T, N = market.shape
    x = np.zeros((T, N))
    for d in drivers:
        x += rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * np.nan_to_num(
            np.asarray(market.latents.get(d, np.zeros((T, N))), dtype=np.float64))
    x += rng.normal(0, rng.uniform(0.6, 1.6), (T, N))
    if cat in ("fundamental", "earnings", "institutions"):
        x = _quarterly_steps(rng, x)
        scale = np.exp(0.8 * np.nan_to_num(market.latents["size"]))
        x = x * scale if rng.random() < 0.6 else x
    elif cat in ("sentiment", "news", "socialmedia"):
        x = np.where(rng.random((T, N)) < rng.uniform(0.2, 0.6), x, np.nan)
    coverage = rng.uniform(0.75, 1.0)
    x[:, rng.random(N) > coverage] = np.nan
    x = x.astype(np.float32)
    market.proxies[name] = x
    return x
