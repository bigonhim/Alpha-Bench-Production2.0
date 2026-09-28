"""Idea space for "Create": economic theses that build alpha expressions.

Each thesis states a mechanism (why the anomaly should exist), requires field
*roles* rather than names, and builds a core signal whose sign means "long".
The generator then dresses the core (normalization, smoothing, conditioning)
and picks settings that suit the thesis' speed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..expr.ast import BinOp, Call, Ident, Node, Num, Str, Unary
from .fields import FieldPool, novelty_weight

FAMILIES = {
    "price_volume": "Price / volume",
    "fundamental": "Fundamental & analyst",
    "alternative": "Alternative data",
    "event_driven": "Event-driven",
    "cross_sectional": "Cross-sectional",
    "time_series": "Time-series",
}
SPARSE_ROLES = {"estimate", "coverage", "sentiment", "attention", "option", "positioning", "model"}


# ------------------------------------------------------------------ tiny DSL
def _n(x) -> Node:
    if isinstance(x, Node):
        return x
    if isinstance(x, str):
        return Str(x)
    return Num(float(x))


def C(name: str, *args, **kw) -> Call:
    return Call(name, tuple(_n(a) for a in args), tuple((k, _n(v)) for k, v in kw.items()))


def F(name: str) -> Ident:
    return Ident(name)


def neg(x: Node) -> Node:
    return Unary("-", x)


def div(a, b) -> Node:
    return BinOp("/", _n(a), _n(b))


def sub(a, b) -> Node:
    return BinOp("-", _n(a), _n(b))


def add(a, b) -> Node:
    return BinOp("+", _n(a), _n(b))


def mul(a, b) -> Node:
    return BinOp("*", _n(a), _n(b))


def gt(a, b) -> Node:
    return BinOp(">", _n(a), _n(b))


def lt(a, b) -> Node:
    return BinOp("<", _n(a), _n(b))


class Ctx:
    def __init__(self, rng: np.random.Generator, pool: FieldPool, usage: Counter):
        self.rng = rng
        self.pool = pool
        self.usage = usage
        self.used: list[str] = []
        self.notes: list[str] = []

    def coin(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def choice(self, seq):
        return seq[int(self.rng.integers(0, len(seq)))]

    def win(self, *choices: int) -> int:
        return int(self.choice(choices))

    def pick(self, role: str, exclude: tuple[str, ...] = (), prefer: tuple[str, ...] = ()) -> str | None:
        names = [n for n in self.pool.available(role) if n not in exclude and n not in self.used]
        if not names:
            names = [n for n in self.pool.available(role) if n not in exclude]
        if not names:
            return None
        preferred = [n for n in prefer if n in names]
        if preferred and self.coin(0.6):
            names = preferred
        w = np.array([novelty_weight(self.pool, n) / (1.0 + self.usage.get(n, 0)) for n in names])
        name = names[int(self.rng.choice(len(names), p=w / w.sum()))]
        self.used.append(name)
        return name

    def field(self, role: str, **kw) -> Node | None:
        """A field node; sparse data gets backfilled so coverage doesn't collapse."""
        name = self.pick(role, **kw)
        if name is None:
            return None
        cat = (self.pool.info.get(name) or {}).get("category", "")
        if role in SPARSE_ROLES or cat in ("sentiment", "news", "socialmedia", "analyst", "option", "model"):
            days = self.win(5, 10, 20, 60)
            self.notes.append(f"{name} is sparse, so it's forward-filled for up to {days} days (ts_backfill).")
            return C("ts_backfill", F(name), days)
        return F(name)


class Skip(Exception):
    """Raised by a builder when the fields it needs aren't available."""


def need(x):
    if x is None:
        raise Skip()
    return x


@dataclass
class Thesis:
    id: str
    family: str
    title: str
    mechanism: str
    speed: str                        # fast | medium | slow
    neutralizations: tuple[str, ...]
    roles: tuple[str, ...]
    build: Callable[[Ctx], tuple[Node, str]]
    normalized: bool = False          # builder already returns a bounded/ranked signal


THESES: list[Thesis] = []


def thesis(id, family, title, mechanism, speed, neuts, roles, normalized=False):
    def deco(fn):
        THESES.append(Thesis(id, family, title, mechanism, speed, tuple(neuts), tuple(roles), fn, normalized))
        return fn
    return deco


# ======================================================================
# Price / volume
# ======================================================================
@thesis("short_reversal", "price_volume", "Short-term reversal",
        "Liquidity providers are paid for absorbing temporary order imbalances, so stocks that overshoot "
        "over a few days tend to partially give the move back.",
        "fast", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("returns", "price"))
def _b(ctx: Ctx):
    k = ctx.win(3, 4, 5, 7, 10)
    style = ctx.choice(["sum", "zs", "delta"])
    if style == "sum":
        return neg(C("ts_sum", F("returns"), k)), f"{k}-day summed returns, sign flipped"
    p = need(ctx.pick("price", prefer=("close", "vwap")))
    if style == "zs":
        return neg(C("ts_zscore", F(p), k * 2)), f"{p} z-scored against its own {k * 2}-day history, sign flipped"
    return neg(div(C("ts_delta", F(p), k), F(p))), f"{k}-day percentage change in {p}, sign flipped"


@thesis("vwap_gap", "price_volume", "Close-vs-VWAP pressure",
        "A close far from the day's volume-weighted price shows late-session pressure that tends to unwind "
        "the next days.", "fast", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(1, 3, 5)
    core = div(sub(F("vwap"), F("close")), F("close"))
    if k > 1:
        return C("ts_mean", core, k), f"(vwap - close) / close averaged over {k} days"
    return core, "(vwap - close) / close"


@thesis("pv_divergence", "price_volume", "Price-volume divergence",
        "When price and volume rise together, trend-chasing flow is being exhausted; a strong positive "
        "price-volume correlation predicts weaker returns.", "fast", ("MARKET", "INDUSTRY", "SECTOR"),
        ("price", "volume"))
def _b(ctx: Ctx):
    k = ctx.win(5, 10, 15, 20)
    p = need(ctx.pick("price", prefer=("close", "vwap", "high")))
    return neg(C("ts_corr", C("rank", F(p)), C("rank", F("volume")), k)), \
        f"negative {k}-day correlation of ranked {p} and ranked volume"


@thesis("volume_shock_reversal", "price_volume", "High-volume moves revert",
        "Price moves on abnormally heavy volume mostly reflect liquidity demand rather than news, and "
        "revert more strongly than quiet moves.", "fast", ("MARKET", "INDUSTRY", "SUBINDUSTRY"),
        ("returns", "volume", "adv"))
def _b(ctx: Ctx):
    k = ctx.win(1, 2, 3)
    move = F("returns") if k == 1 else C("ts_sum", F("returns"), k)
    surge = div(F("volume"), F("adv20"))
    return neg(mul(C("rank", move), C("rank", surge))), \
        f"ranked {k}-day return times ranked volume surge (volume / adv20), sign flipped"


@thesis("range_position", "price_volume", "Intraday range position",
        "Closing near the top of the day's range after one-sided buying tends to reverse as that "
        "demand fades.", "fast", ("MARKET", "INDUSTRY"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(1, 3, 5, 10)
    pos = div(sub(sub(F("close"), F("low")), sub(F("high"), F("close"))), add(sub(F("high"), F("low")), 0.001))
    core = neg(pos if k == 1 else C("ts_mean", pos, k))
    return core, f"where the close sits in the high-low range{'' if k == 1 else f', averaged over {k} days'}, sign flipped"


@thesis("momentum", "price_volume", "Intermediate-term momentum",
        "Investors under-react to gradually arriving information, so 6-12 month winners keep outperforming; "
        "skipping the latest month avoids the short-term reversal.", "slow", ("MARKET", "INDUSTRY", "SECTOR"),
        ("returns",))
def _b(ctx: Ctx):
    L = ctx.win(126, 189, 231, 252)
    skip = ctx.win(10, 21, 22)
    if ctx.coin(0.5):
        return sub(C("ts_sum", F("returns"), L), C("ts_sum", F("returns"), skip)), \
            f"{L}-day return excluding the latest {skip} days"
    return sub(div(C("ts_delay", F("close"), skip), C("ts_delay", F("close"), L)), 1), \
        f"price change from {L} to {skip} days ago"


@thesis("low_volatility", "price_volume", "Low-volatility anomaly",
        "Leverage-constrained investors bid up volatile, lottery-like stocks, so low-risk stocks earn higher "
        "risk-adjusted returns.", "slow", ("MARKET", "SECTOR", "INDUSTRY"), ("returns",))
def _b(ctx: Ctx):
    k = ctx.win(20, 60, 120, 250)
    return neg(C("ts_std_dev", F("returns"), k)), f"negative {k}-day return volatility"


@thesis("overnight_gap", "price_volume", "Overnight gap fade",
        "Overnight gaps driven by retail orders and headline reactions partly reverse once full liquidity "
        "arrives.", "fast", ("MARKET", "INDUSTRY"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(1, 3, 5)
    gap = sub(div(F("open"), C("ts_delay", F("close"), 1)), 1)
    return neg(gap if k == 1 else C("ts_sum", gap, k)), f"open-vs-prior-close gap{'' if k == 1 else f' summed over {k} days'}, sign flipped"


@thesis("near_high", "price_volume", "Anchoring to the 52-week high",
        "Traders anchor on the 52-week high and are slow to bid through it, so stocks close to it tend to "
        "keep drifting up when news is good.", "slow", ("MARKET", "INDUSTRY", "SECTOR"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(126, 252)
    return div(F("close"), C("ts_max", F("high"), k)), f"close relative to its {k}-day high"


@thesis("illiquidity", "price_volume", "Illiquidity premium",
        "Investors demand a premium to hold stocks whose prices move a lot per dollar traded (Amihud "
        "illiquidity).", "slow", ("MARKET", "SECTOR", "INDUSTRY"), ("returns", "volume", "price"))
def _b(ctx: Ctx):
    k = ctx.win(20, 60, 120)
    return C("ts_mean", div(C("abs", F("returns")), mul(F("volume"), F("vwap"))), k), \
        f"{k}-day average of |return| per dollar traded"


# ======================================================================
# Fundamental & analyst
# ======================================================================
@thesis("value_book", "fundamental", "Book-to-market value",
        "Cheap stocks relative to accounting book value earn a premium: investors extrapolate past "
        "disappointments too far.", "slow", ("SUBINDUSTRY", "INDUSTRY", "SECTOR"), ("balance", "cap"))
def _b(ctx: Ctx):
    b = need(ctx.pick("balance", prefer=("equity",)))
    return div(F(b), F("cap")), f"{b} / market cap"


@thesis("earnings_yield", "fundamental", "Earnings / cash-flow yield",
        "Profits per dollar of market value are a cleaner value measure than book: high-yield firms are "
        "systematically under-priced.", "slow", ("SUBINDUSTRY", "INDUSTRY"), ("flow", "cap"))
def _b(ctx: Ctx):
    f = need(ctx.pick("flow", prefer=("income", "operating_income", "ebitda", "cashflow_op")))
    if ctx.pool.has("ev") and ctx.coin(0.35):
        return div(F(f), F("enterprise_value")), f"{f} / enterprise value"
    return div(F(f), F("cap")), f"{f} / market cap"


@thesis("profitability", "fundamental", "Profitability (quality)",
        "Highly profitable firms keep out-earning expectations; the market under-weights the persistence "
        "of profitability.", "slow", ("SUBINDUSTRY", "INDUSTRY", "SECTOR"), ("flow", "balance"))
def _b(ctx: Ctx):
    f = need(ctx.pick("flow", prefer=("operating_income", "income", "ebitda")))
    b = need(ctx.pick("balance", prefer=("assets", "equity")))
    return div(F(f), F(b)), f"{f} / {b}"


@thesis("accruals", "fundamental", "Accruals (earnings quality)",
        "Earnings made of accruals rather than cash are less persistent; investors fixate on headline "
        "earnings and are later disappointed.", "slow", ("SUBINDUSTRY", "INDUSTRY"), ("flow", "balance"))
def _b(ctx: Ctx):
    inc = need(ctx.pick("flow", prefer=("income", "operating_income")))
    cfo = need(ctx.pick("flow", prefer=("cashflow_op",), exclude=(inc,)))
    a = need(ctx.pick("balance", prefer=("assets",)))
    return neg(div(sub(F(inc), F(cfo)), F(a))), f"-({inc} - {cfo}) / {a}: favours cash-backed earnings"


@thesis("leverage", "fundamental", "Balance-sheet strength",
        "Over-levered firms carry distress risk the market under-prices; conservatively financed firms "
        "outperform.", "slow", ("SUBINDUSTRY", "INDUSTRY", "SECTOR"), ("balance",))
def _b(ctx: Ctx):
    d = need(ctx.pick("balance", prefer=("liabilities", "debt")))
    a = need(ctx.pick("balance", prefer=("assets", "equity"), exclude=(d,)))
    return neg(div(F(d), F(a))), f"-{d} / {a}"


@thesis("asset_growth", "fundamental", "Investment (asset growth)",
        "Firms that grow assets aggressively tend to over-invest and subsequently underperform.",
        "slow", ("SUBINDUSTRY", "INDUSTRY"), ("balance",))
def _b(ctx: Ctx):
    a = need(ctx.pick("balance", prefer=("assets",)))
    k = ctx.win(126, 252)
    return neg(sub(div(F(a), C("ts_delay", F(a), k)), 1)), f"-{k}-day growth of {a}"


@thesis("fundamental_momentum", "fundamental", "Improving fundamentals",
        "Improvements in profitability are priced in slowly; firms whose margins or returns on capital are "
        "rising keep outperforming.", "slow", ("SUBINDUSTRY", "INDUSTRY"), ("flow", "balance"))
def _b(ctx: Ctx):
    f = need(ctx.pick("flow"))
    b = need(ctx.pick("balance", prefer=("assets",)))
    k = ctx.win(63, 126, 252)
    return C("ts_delta", div(F(f), F(b)), k), f"{k}-day change in {f} / {b}"


@thesis("estimate_revisions", "fundamental", "Analyst estimate revisions",
        "Analysts revise gradually and herd, so recent upward revisions predict further revisions and "
        "positive surprises.", "medium", ("INDUSTRY", "SUBINDUSTRY", "MARKET"), ("estimate",))
def _b(ctx: Ctx):
    e = need(ctx.field("estimate"))
    k = ctx.win(10, 20, 40, 60)
    return div(C("ts_delta", e, k), C("abs", C("ts_delay", e, k))), f"{k}-day percentage revision of the estimate"


@thesis("target_upside", "fundamental", "Price-target upside",
        "Consensus price targets far above the current price reflect information that prices incorporate "
        "with a lag.", "medium", ("INDUSTRY", "SUBINDUSTRY"), ("estimate",))
def _b(ctx: Ctx):
    if "est_ptg" not in ctx.pool.available("estimate"):
        raise Skip()
    ctx.used.append("est_ptg")
    return sub(div(C("ts_backfill", F("est_ptg"), 20), F("close")), 1), "consensus target / close - 1"


# ======================================================================
# Alternative data
# ======================================================================
@thesis("sentiment_level", "alternative", "News / social sentiment",
        "Sentiment in news and social data leads order flow: positive coverage attracts buyers over the "
        "following days.", "fast", ("MARKET", "INDUSTRY", "SECTOR"), ("sentiment",))
def _b(ctx: Ctx):
    s = need(ctx.field("sentiment"))
    k = ctx.win(3, 5, 10)
    return C("ts_mean", s, k), f"{k}-day average sentiment"


@thesis("sentiment_surprise", "alternative", "Sentiment surprise",
        "A change in tone relative to a stock's usual coverage carries more information than the level.",
        "fast", ("MARKET", "INDUSTRY"), ("sentiment",))
def _b(ctx: Ctx):
    s = need(ctx.field("sentiment"))
    k = ctx.win(20, 40, 60)
    return sub(s, C("ts_mean", s, k)), f"sentiment minus its {k}-day average"


@thesis("attention_reversal", "alternative", "Attention-driven overpricing",
        "Attention spikes pull in retail buyers who over-pay; returns earned during high-attention "
        "episodes tend to revert.", "fast", ("MARKET", "INDUSTRY"), ("attention", "returns"))
def _b(ctx: Ctx):
    a = need(ctx.field("attention"))
    k = ctx.win(3, 5)
    return neg(mul(C("rank", div(a, C("ts_mean", a, 20))), C("ts_sum", F("returns"), k))), \
        f"attention surge (vs 20-day mean) times {k}-day return, sign flipped"


@thesis("options_signal", "alternative", "Options-implied fear",
        "Informed traders act in options first: rising implied volatility or put demand relative to a "
        "stock's own history precedes weak returns.", "medium", ("MARKET", "INDUSTRY", "SECTOR"), ("option",))
def _b(ctx: Ctx):
    o = need(ctx.field("option"))
    k = ctx.win(20, 60, 120)
    return neg(C("ts_zscore", o, k)), f"-{k}-day z-score of the options measure"


@thesis("short_interest", "alternative", "Short-seller information",
        "Short sellers are informed and pay to be short; heavily shorted or rising-short stocks "
        "underperform.", "medium", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("positioning",))
def _b(ctx: Ctx):
    s = need(ctx.field("positioning"))
    if ctx.coin(0.5):
        k = ctx.win(10, 20, 40)
        return neg(C("ts_delta", s, k)), f"-{k}-day change in short positioning"
    return neg(s), "-short positioning level"


@thesis("model_signal", "alternative", "Vendor model score",
        "Third-party model scores aggregate information the price hasn't fully absorbed; ranking against "
        "each stock's own history isolates fresh changes.", "medium", ("INDUSTRY", "SUBINDUSTRY"), ("model",))
def _b(ctx: Ctx):
    m = need(ctx.field("model"))
    k = ctx.win(20, 60, 120)
    return C("ts_rank", m, k), f"{k}-day time-series rank of the model score"


# ======================================================================
# Event-driven
# ======================================================================
@thesis("post_report", "event_driven", "Fresh-report fundamentals",
        "Fundamental information is most valuable right after it's published; trading it only in the "
        "weeks after a new report avoids stale positions.", "medium", ("SUBINDUSTRY", "INDUSTRY"),
        ("flow", "balance"))
def _b(ctx: Ctx):
    f = need(ctx.pick("flow"))
    b = need(ctx.pick("balance", prefer=("assets", "equity")))
    enter = ctx.win(3, 5, 10)
    hold = ctx.win(40, 60)
    fresh = C("days_from_last_change", F(f))
    return C("trade_when", lt(fresh, enter), C("rank", div(F(f), F(b))), gt(fresh, hold)), \
        f"rank({f}/{b}) entered within {enter} days of a new report and held up to {hold} days"


@thesis("volume_event", "event_driven", "Abnormal-volume events",
        "Only abnormal-volume days carry the liquidity shocks that revert; ignoring quiet days sharpens "
        "the reversal signal and cuts turnover.", "medium", ("MARKET", "INDUSTRY"), ("volume", "adv", "returns"))
def _b(ctx: Ctx):
    m = ctx.choice([1.5, 2.0, 2.5])
    k = ctx.win(2, 3, 5)
    return C("trade_when", gt(F("volume"), mul(m, F("adv20"))), neg(C("ts_sum", F("returns"), k)), -1), \
        f"fade the {k}-day move, updated only on days when volume > {m} x adv20"


@thesis("news_event", "event_driven", "News-day sentiment",
        "Sentiment matters most on days with unusual news flow; holding the reading between events keeps "
        "turnover low.", "medium", ("MARKET", "INDUSTRY"), ("attention", "sentiment"))
def _b(ctx: Ctx):
    a = need(ctx.field("attention"))
    s = need(ctx.field("sentiment"))
    return C("trade_when", gt(a, C("ts_mean", a, 20)), s, -1), "sentiment updated only on above-normal news days"


@thesis("extreme_move", "event_driven", "Extreme-move reversal",
        "Moves several standard deviations beyond normal are disproportionately driven by forced trading "
        "and revert.", "fast", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("returns",))
def _b(ctx: Ctx):
    z = ctx.choice([1.5, 2.0, 2.5])
    k = ctx.win(20, 60)
    return C("trade_when", gt(C("abs", F("returns")), mul(z, C("ts_std_dev", F("returns"), k))),
             neg(F("returns")), -1), f"fade returns larger than {z} x their {k}-day volatility"


# ======================================================================
# Cross-sectional
# ======================================================================
@thesis("quality_value", "cross_sectional", "Quality-at-a-reasonable-price composite",
        "Value alone buys value traps; pairing cheapness with profitability keeps the cheap stocks that "
        "are actually good businesses.", "slow", ("SUBINDUSTRY", "INDUSTRY"), ("flow", "balance", "cap"),
        normalized=True)
def _b(ctx: Ctx):
    f = need(ctx.pick("flow", prefer=("operating_income", "income")))
    a = need(ctx.pick("balance", prefer=("assets",)))
    b = need(ctx.pick("balance", prefer=("equity",), exclude=(a,)))
    return add(C("rank", div(F(b), F("cap"))), C("rank", div(F(f), F(a)))), \
        f"rank({b}/cap) + rank({f}/{a})"


@thesis("peer_relative", "cross_sectional", "Peer-relative fundamentals",
        "Accounting ratios are only comparable within an industry; ranking against peers removes "
        "structural industry differences.", "slow", ("NONE", "MARKET", "SECTOR"), ("flow", "balance"),
        normalized=True)
def _b(ctx: Ctx):
    f = need(ctx.pick("flow"))
    b = need(ctx.pick("balance"))
    g = ctx.choice(["industry", "subindustry", "sector"])
    op = ctx.choice(["group_rank", "group_zscore"])
    return C(op, div(F(f), F(b)), F(g)), f"{op.replace('_', ' ')} of {f}/{b} within {g}"


@thesis("size_neutral_reversal", "cross_sectional", "Size-neutral reversal",
        "Reversal is strongest in small caps; comparing only stocks of similar size keeps the effect "
        "while making it robust in larger universes.", "fast", ("MARKET", "INDUSTRY"), ("returns", "cap"),
        normalized=True)
def _b(ctx: Ctx):
    k = ctx.win(3, 5, 10)
    return C("group_rank", neg(C("ts_sum", F("returns"), k)), C("bucket", C("rank", F("cap")), range="0.1,1,0.1")), \
        f"{k}-day reversal ranked within size deciles"


@thesis("residual_signal", "cross_sectional", "Beta-free idiosyncratic signal",
        "Removing the part of a signal explained by volatility isolates the mispricing from a risk tilt.",
        "medium", ("INDUSTRY", "SUBINDUSTRY"), ("returns", "flow", "balance"), normalized=True)
def _b(ctx: Ctx):
    f = need(ctx.pick("flow"))
    b = need(ctx.pick("balance"))
    return C("regression_neut", C("rank", div(F(f), F(b))), C("rank", C("ts_std_dev", F("returns"), 60))), \
        f"rank({f}/{b}) with 60-day volatility regressed out"


# ======================================================================
# Time-series
# ======================================================================
@thesis("valuation_vs_history", "time_series", "Cheap versus its own history",
        "A stock trading cheap relative to its own valuation history is more likely mispriced than one "
        "that is structurally cheap.", "slow", ("SUBINDUSTRY", "INDUSTRY", "MARKET"), ("flow", "cap"),
        normalized=True)
def _b(ctx: Ctx):
    f = need(ctx.pick("flow"))
    k = ctx.win(126, 252, 504)
    return C("ts_rank", div(F(f), F("cap")), k), f"{k}-day time-series rank of {f}/cap"


@thesis("trend_residual", "time_series", "Deviation from trend",
        "Prices that stray from their own recent linear trend tend to snap back toward it.",
        "fast", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(10, 20, 40)
    p = need(ctx.pick("price", prefer=("close", "vwap")))
    return neg(div(C("ts_regression", F(p), C("ts_step", 1), k), F(p))), \
        f"-{k}-day residual of {p} from its linear trend, scaled by price"


@thesis("zscore_reversion", "time_series", "Own-history mean reversion",
        "Relative to its own recent range, an extended price tends to revert as temporary demand fades.",
        "fast", ("MARKET", "INDUSTRY"), ("price",))
def _b(ctx: Ctx):
    k = ctx.win(10, 20, 40)
    p = need(ctx.pick("price", prefer=("close", "vwap", "low")))
    return neg(C("ts_zscore", F(p), k)), f"-{k}-day z-score of {p}"


@thesis("volume_confirmed_trend", "time_series", "Volume-confirmed trend",
        "Trends accompanied by rising participation carry information; trends on fading volume don't.",
        "medium", ("MARKET", "INDUSTRY", "SECTOR"), ("returns", "volume"))
def _b(ctx: Ctx):
    k = ctx.win(10, 20, 40)
    return mul(C("ts_sum", F("returns"), k), div(C("ts_mean", F("volume"), 5), C("ts_mean", F("volume"), 60))), \
        f"{k}-day return scaled by the 5-day/60-day volume ratio"


# ======================================================================
# Ideas taught in WorldQuant's Learn2Quant lessons (see alpha_bench/course.py)
# ======================================================================
def _have(ctx: Ctx, role: str, *names: str) -> bool:
    return all(n in ctx.pool.available(role) for n in names)


@thesis("open_close_reversion", "price_volume", "Open-to-close reversion vs peers (Learn2Quant L4)",
        "If a stock closes below its open it tends to bounce back and outperform its peers, and vice versa: "
        "intraday selling pressure is temporary.", "fast", ("SUBINDUSTRY", "INDUSTRY"), ("price",), normalized=True)
def _b(ctx: Ctx):
    g = ctx.choice(["subindustry", "industry"])
    k = ctx.win(1, 1, 3, 5)
    core = sub(F("close"), F("open")) if k == 1 else C("ts_mean", sub(F("close"), F("open")), k)
    ctx.used += ["close", "open"]
    return neg(C("group_rank", core, F(g))), \
        f"-group_rank(close - open{'' if k == 1 else f', averaged over {k} days'}) within {g}"


@thesis("cashflow_yield_trend", "fundamental", "Improving cash-flow yield (Learn2Quant L4)",
        "Cash flow from operations relative to market value shows whether a firm is fairly priced for its cash "
        "generation; when that ratio is improving versus its own recent history, the stock tends to outperform.",
        "medium", ("MARKET", "INDUSTRY", "SUBINDUSTRY"), ("flow", "cap"))
def _b(ctx: Ctx):
    f = need(ctx.pick("flow", prefer=("cashflow_op",)))
    k = ctx.win(40, 60, 120)
    return C("ts_zscore", div(F(f), F("cap")), k), f"{k}-day time-series z-score of {f} / cap"


@thesis("buzz_reversal", "alternative", "Attention (buzz) reversal (Learn2Quant L5)",
        "A stock whose sentiment buzz is rising versus its recent average is attracting unusual attention and is "
        "likely overpriced: short it; go long stocks whose buzz is fading.", "fast", ("INDUSTRY", "MARKET", "SECTOR"),
        ("attention",))
def _b(ctx: Ctx):
    name = "snt_buzz" if "snt_buzz" in ctx.pool.available("attention") else need(ctx.pick("attention"))
    ctx.used.append(name)
    k = ctx.win(5, 10, 20)
    return neg(div(F(name), C("ts_mean", F(name), k))), f"-{name} / its {k}-day mean"


@thesis("iv_call_put_spread", "alternative", "Call vs put implied-volatility demand (Learn2Quant L5)",
        "When implied volatility of calls exceeds that of puts, demand for the stock exceeds supply and it tends "
        "to appreciate; the reverse signals weakness.", "medium", ("INDUSTRY", "SUBINDUSTRY", "MARKET"), ("option",))
def _b(ctx: Ctx):
    h = ctx.choice(["720", "120"])
    call, put = f"implied_volatility_call_{h}", f"implied_volatility_put_{h}"
    if not _have(ctx, "option", call, put):
        raise Skip()
    ctx.used += [call, put]
    return sub(F(call), F(put)), f"{call} - {put} ({'2-year' if h == '720' else '4-month'} horizon)"


@thesis("iv_vs_realized", "alternative", "Implied vs realised volatility (Learn2Quant L7)",
        "At-the-money call implied volatility well above realised (Parkinson high-low) volatility reveals call "
        "demand ahead of the move; best used with timely (delay-0) data.", "medium", ("SUBINDUSTRY", "INDUSTRY"),
        ("option",))
def _b(ctx: Ctx):
    if not _have(ctx, "option", "implied_volatility_call_120", "parkinson_volatility_120"):
        raise Skip()
    ctx.used += ["implied_volatility_call_120", "parkinson_volatility_120"]
    return div(F("implied_volatility_call_120"), F("parkinson_volatility_120")), \
        "implied_volatility_call_120 / parkinson_volatility_120"


@thesis("volume_momentum", "price_volume", "Volume momentum (Learn2Quant L6)",
        "Last week's trading volume well above the past year's average signals rising liquidity and bullish "
        "interest, which tends to precede gains.", "medium", ("MARKET", "INDUSTRY", "SECTOR"), ("volume",),
        normalized=True)
def _b(ctx: Ctx):
    a, b = ctx.win(3, 5, 10), ctx.win(120, 240, 250)
    ctx.used.append("volume")
    return C("sqrt", C("rank", div(C("ts_mean", F("volume"), a), C("ts_mean", F("volume"), b)))), \
        f"sqrt(rank({a}-day / {b}-day average volume))"


@thesis("industry_lead_lag", "cross_sectional", "Industry lead-lag catch-up (Learn2Quant L6 idea type)",
        "Lead-lag relationships: news reaches the largest firms in an industry first, and smaller peers catch up. "
        "Stocks lagging their cap-weighted industry move tend to follow it.", "fast", ("MARKET", "SECTOR"),
        ("returns", "cap"))
def _b(ctx: Ctx):
    k = ctx.win(3, 5, 10)
    lead = C("ts_sum", C("group_mean", F("returns"), F("cap"), F("industry")), k)
    return sub(lead, C("ts_sum", F("returns"), k)), f"{k}-day cap-weighted industry return minus own return"


@thesis("vol_regime_reversal", "event_driven", "Reversion only in volatile regimes (Learn2Quant L4 improvement)",
        "Reversion works best when volatility is elevated; updating positions only then (and holding otherwise) "
        "keeps the edge while cutting turnover.", "medium", ("SUBINDUSTRY", "INDUSTRY", "MARKET"), ("returns", "price"))
def _b(ctx: Ctx):
    k = ctx.win(3, 5, 10)
    vol = C("ts_std_dev", F("returns"), 20)
    gate = gt(vol, C("ts_mean", vol, ctx.win(60, 120)))
    return C("trade_when", gate, neg(div(C("ts_delta", F("close"), k), F("close"))), -1), \
        f"{k}-day price reversion, updated only when 20-day volatility is above its longer-run average"


def by_id(tid: str) -> Thesis | None:
    return next((t for t in THESES if t.id == tid), None)
