"""BRAIN operator catalog.

This is the built-in reference list. When signed in, `merge_platform_operators`
overlays the account's real /operators response: operators the account can't
use are marked unavailable and never generated, and platform-only operators are
listed (but can't be evaluated offline until implemented).

Fields per operator:
  window   index of the lookback-days argument (None if no lookback)
  smooth   True if the operator lowers turnover (averaging / decay)
  cs       True if it's cross-sectional (acts across stocks on one day)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperatorSpec:
    name: str
    category: str
    signature: str
    description: str
    min_args: int
    max_args: int
    window: int | None = None
    smooth: bool = False
    cs: bool = False
    group_arg: int | None = None
    available: bool = True          # False when the account's /operators list omits it
    platform_only: bool = False     # listed by BRAIN but not evaluable offline
    kwargs: dict = field(default_factory=dict)


def _op(name, category, signature, description, min_args, max_args=None, **kw) -> OperatorSpec:
    return OperatorSpec(name, category, signature, description, min_args,
                        min_args if max_args is None else max_args, **kw)


A, L, TS, CS, G, T, V = ("Arithmetic", "Logical", "Time Series", "Cross Sectional", "Group",
                         "Transformational", "Vector")

_CATALOG = [
    # Arithmetic
    _op("abs", A, "abs(x)", "Absolute value.", 1),
    _op("add", A, "add(x, y, filter=false)", "x + y (filter=true treats NaN as 0).", 2, 8, kwargs={"filter": False}),
    _op("subtract", A, "subtract(x, y, filter=false)", "x - y.", 2, 2, kwargs={"filter": False}),
    _op("multiply", A, "multiply(x, y, filter=false)", "x * y.", 2, 8, kwargs={"filter": False}),
    _op("divide", A, "divide(x, y)", "x / y (division by zero gives NaN).", 2),
    _op("power", A, "power(x, y)", "x raised to y.", 2),
    _op("signed_power", A, "signed_power(x, y)", "sign(x) * |x|^y; keeps direction while reshaping magnitude.", 2),
    _op("sqrt", A, "sqrt(x)", "Square root.", 1),
    _op("log", A, "log(x)", "Natural log (NaN for x <= 0).", 1),
    _op("exp", A, "exp(x)", "Exponential.", 1),
    _op("sign", A, "sign(x)", "-1, 0 or +1.", 1),
    _op("inverse", A, "inverse(x)", "1 / x.", 1),
    _op("reverse", A, "reverse(x)", "-x.", 1),
    _op("s_log_1p", A, "s_log_1p(x)", "sign(x) * log(1 + |x|): compresses outliers.", 1),
    _op("max", A, "max(x, y, ...)", "Element-wise maximum.", 2, 8),
    _op("min", A, "min(x, y, ...)", "Element-wise minimum.", 2, 8),
    _op("densify", A, "densify(x)", "Re-index a grouping field to consecutive integers.", 1),
    _op("to_nan", A, "to_nan(x, value=0, reverse=false)", "Replace value with NaN (or the reverse).", 1, 1,
        kwargs={"value": 0, "reverse": False}),
    # Logical
    _op("if_else", L, "if_else(cond, x, y)", "x where cond is true, else y.", 3),
    _op("is_nan", L, "is_nan(x)", "1 where x is NaN, else 0.", 1),
    _op("not", L, "not(x)", "Logical negation.", 1),
    _op("and", L, "and(x, y)", "Logical AND.", 2),
    _op("or", L, "or(x, y)", "Logical OR.", 2),
    _op("less", L, "x < y", "Comparison.", 2), _op("greater", L, "x > y", "Comparison.", 2),
    _op("less_equal", L, "x <= y", "Comparison.", 2), _op("greater_equal", L, "x >= y", "Comparison.", 2),
    _op("equal", L, "x == y", "Comparison.", 2), _op("not_equal", L, "x != y", "Comparison.", 2),
    # Time series
    _op("ts_delay", TS, "ts_delay(x, d)", "Value d days ago.", 2, window=1),
    _op("ts_delta", TS, "ts_delta(x, d)", "x - ts_delay(x, d).", 2, window=1),
    _op("ts_mean", TS, "ts_mean(x, d)", "Mean over the past d days.", 2, window=1, smooth=True),
    _op("ts_sum", TS, "ts_sum(x, d)", "Sum over the past d days.", 2, window=1, smooth=True),
    _op("ts_std_dev", TS, "ts_std_dev(x, d)", "Standard deviation over d days.", 2, window=1),
    _op("ts_zscore", TS, "ts_zscore(x, d)", "(x - ts_mean) / ts_std_dev over d days.", 2, window=1),
    _op("ts_rank", TS, "ts_rank(x, d, constant=0)", "Rank of today's value within the past d days, in [0, 1].", 2,
        window=1, kwargs={"constant": 0}),
    _op("ts_min", TS, "ts_min(x, d)", "Minimum over d days.", 2, window=1),
    _op("ts_max", TS, "ts_max(x, d)", "Maximum over d days.", 2, window=1),
    _op("ts_arg_min", TS, "ts_arg_min(x, d)", "Days since the d-day minimum.", 2, window=1),
    _op("ts_arg_max", TS, "ts_arg_max(x, d)", "Days since the d-day maximum.", 2, window=1),
    _op("ts_product", TS, "ts_product(x, d)", "Product over d days.", 2, window=1),
    _op("ts_corr", TS, "ts_corr(x, y, d)", "Rolling correlation of x and y over d days.", 3, window=2),
    _op("ts_covariance", TS, "ts_covariance(y, x, d)", "Rolling covariance over d days.", 3, window=2),
    _op("ts_decay_linear", TS, "ts_decay_linear(x, d, dense=false)", "Linearly weighted average (newest weight d).", 2,
        window=1, smooth=True, kwargs={"dense": False}),
    _op("ts_decay_exp_window", TS, "ts_decay_exp_window(x, d, factor=0.5)", "Exponentially weighted average.", 2, 3,
        window=1, smooth=True, kwargs={"factor": 0.5}),
    _op("ts_av_diff", TS, "ts_av_diff(x, d)", "x - ts_mean(x, d), ignoring NaN.", 2, window=1),
    _op("ts_scale", TS, "ts_scale(x, d, constant=0)", "(x - ts_min) / (ts_max - ts_min) + constant.", 2, window=1,
        kwargs={"constant": 0}),
    _op("ts_quantile", TS, "ts_quantile(x, d, driver=\"gaussian\")", "ts_rank mapped through an inverse CDF.", 2,
        window=1, kwargs={"driver": "gaussian"}),
    _op("ts_backfill", TS, "ts_backfill(x, d)", "Fill NaN with the most recent valid value within d days.", 2, 3,
        window=1, smooth=True),
    _op("ts_count_nans", TS, "ts_count_nans(x, d)", "Number of NaN values in the past d days.", 2, window=1),
    _op("ts_regression", TS, "ts_regression(y, x, d, lag=0, rettype=0)", "Rolling OLS of y on x; rettype 0 = residual, 2 = slope.",
        3, window=2, kwargs={"lag": 0, "rettype": 0}),
    _op("ts_step", TS, "ts_step(1)", "Day counter.", 1),
    _op("ts_skewness", TS, "ts_skewness(x, d)", "Rolling skewness.", 2, window=1),
    _op("ts_kurtosis", TS, "ts_kurtosis(x, d)", "Rolling excess kurtosis.", 2, window=1),
    _op("ts_ir", TS, "ts_ir(x, d)", "ts_mean / ts_std_dev over d days.", 2, window=1),
    _op("ts_max_diff", TS, "ts_max_diff(x, d)", "x - ts_max(x, d).", 2, window=1),
    _op("ts_median", TS, "ts_median(x, d)", "Rolling median.", 2, window=1, smooth=True),
    _op("days_from_last_change", TS, "days_from_last_change(x)", "Days since x last changed.", 1),
    _op("last_diff_value", TS, "last_diff_value(x, d)", "Most recent value within d days that differs from today's.", 2,
        window=1),
    _op("kth_element", TS, "kth_element(x, d, k=1)", "k-th most recent valid value within d days.", 2, 3, window=1,
        kwargs={"k": 1}),
    _op("hump", TS, "hump(x, hump=0.01)", "Only move the position when the change exceeds a threshold: cuts turnover.", 1, 1,
        smooth=True, kwargs={"hump": 0.01}),
    # Cross sectional
    _op("rank", CS, "rank(x, rate=2)", "Cross-sectional rank scaled to [0, 1].", 1, 1, cs=True, kwargs={"rate": 2}),
    _op("zscore", CS, "zscore(x)", "Cross-sectional z-score.", 1, cs=True),
    _op("scale", CS, "scale(x, scale=1)", "Scale so sum(|x|) = scale.", 1, 1, cs=True, kwargs={"scale": 1}),
    _op("normalize", CS, "normalize(x, useStd=false, limit=0.0)", "Subtract the cross-sectional mean (optionally / std).",
        1, 1, cs=True, kwargs={"useStd": False, "limit": 0.0}),
    _op("quantile", CS, "quantile(x, driver=\"gaussian\", sigma=1.0)", "Rank mapped through an inverse CDF.", 1, 1, cs=True,
        kwargs={"driver": "gaussian", "sigma": 1.0}),
    _op("winsorize", CS, "winsorize(x, std=4)", "Clip values beyond std standard deviations.", 1, 1, cs=True,
        kwargs={"std": 4}),
    _op("regression_neut", CS, "regression_neut(y, x)", "Residual of a cross-sectional regression of y on x.", 2, cs=True),
    _op("vector_neut", CS, "vector_neut(x, y)", "Remove the projection of x on y.", 2, cs=True),
    # Group
    _op("group_neutralize", G, "group_neutralize(x, group)", "Subtract the group mean.", 2, cs=True, group_arg=1),
    _op("group_rank", G, "group_rank(x, group)", "Rank within each group.", 2, cs=True, group_arg=1),
    _op("group_zscore", G, "group_zscore(x, group)", "Z-score within each group.", 2, cs=True, group_arg=1),
    _op("group_scale", G, "group_scale(x, group)", "Min-max scale within each group.", 2, cs=True, group_arg=1),
    _op("group_normalize", G, "group_normalize(x, group)", "Scale so each group's sum(|x|) = 1.", 2, cs=True, group_arg=1),
    _op("group_mean", G, "group_mean(x, weight, group)", "(Weighted) group mean broadcast to members.", 2, 3, cs=True,
        group_arg=-1),
    _op("group_sum", G, "group_sum(x, group)", "Group sum broadcast to members.", 2, cs=True, group_arg=1),
    _op("group_max", G, "group_max(x, group)", "Group maximum.", 2, cs=True, group_arg=1),
    _op("group_min", G, "group_min(x, group)", "Group minimum.", 2, cs=True, group_arg=1),
    _op("group_median", G, "group_median(x, group)", "Group median.", 2, cs=True, group_arg=1),
    _op("group_std_dev", G, "group_std_dev(x, group)", "Group standard deviation.", 2, cs=True, group_arg=1),
    _op("group_count", G, "group_count(x, group)", "Number of valid values in the group.", 2, cs=True, group_arg=1),
    _op("group_backfill", G, "group_backfill(x, group, d, std=4.0)", "Fill NaN with the winsorized group mean over d days.",
        3, 3, cs=True, group_arg=1, window=2, kwargs={"std": 4.0}),
    # Transformational
    _op("trade_when", T, "trade_when(x, y, z)", "Update to y when x > 0; exit (NaN) when z > 0; otherwise hold.", 3,
        smooth=True),
    _op("bucket", T, "bucket(x, range=\"0,1,0.1\")", "Bucket values into groups (use with group_* operators).", 1, 1,
        kwargs={"range": "0,1,0.1"}),
    _op("pasteurize", T, "pasteurize(x)", "Set values outside the universe to NaN.", 1),
    _op("tail", T, "tail(x, lower=0, upper=0, newval=0)", "Replace values inside [lower, upper] with newval.", 1, 1,
        kwargs={"lower": 0, "upper": 0, "newval": 0}),
    _op("left_tail", T, "left_tail(x, maximum=0)", "NaN out values above maximum.", 1, 1, kwargs={"maximum": 0}),
    _op("right_tail", T, "right_tail(x, minimum=0)", "NaN out values below minimum.", 1, 1, kwargs={"minimum": 0}),
    _op("clamp", T, "clamp(x, lower=0, upper=0)", "Clip x to [lower, upper].", 1, 1, kwargs={"lower": 0, "upper": 0}),
    # Vector (event / multi-value fields)
    _op("vec_avg", V, "vec_avg(x)", "Mean of a vector field's values.", 1),
    _op("vec_sum", V, "vec_sum(x)", "Sum of a vector field's values.", 1),
    _op("vec_max", V, "vec_max(x)", "Max of a vector field's values.", 1),
    _op("vec_min", V, "vec_min(x)", "Min of a vector field's values.", 1),
    _op("vec_count", V, "vec_count(x)", "Number of values in a vector field.", 1),
    _op("vec_stddev", V, "vec_stddev(x)", "Std dev of a vector field's values.", 1),
]

OPERATORS: dict[str, OperatorSpec] = {op.name: op for op in _CATALOG}
GROUP_FIELDS = {"market", "sector", "industry", "subindustry", "country", "exchange"}


def spec(name: str) -> OperatorSpec | None:
    return OPERATORS.get(name)


def merge_platform_operators(platform: list[dict]) -> dict[str, int]:
    """Overlay the account's /operators list onto the catalog."""
    names = {str(p.get("name")) for p in platform if p.get("name")}
    if not names:
        return {"available": len(OPERATORS), "unavailable": 0, "platform_only": 0}
    for op in OPERATORS.values():
        if op.category == L and op.name in ("less", "greater", "less_equal", "greater_equal", "equal", "not_equal"):
            continue  # infix comparisons are always available
        op.available = op.name in names
    added = 0
    for p in platform:
        name = str(p.get("name") or "")
        if name and name not in OPERATORS:
            OPERATORS[name] = OperatorSpec(name, str(p.get("category") or "Other"),
                                           str(p.get("definition") or name), str(p.get("description") or ""),
                                           0, 99, available=True, platform_only=True)
            added += 1
    return {"available": sum(o.available and not o.platform_only for o in OPERATORS.values()),
            "unavailable": sum(not o.available for o in OPERATORS.values()),
            "platform_only": added}


def catalog_rows() -> list[dict]:
    return [{"name": o.name, "category": o.category, "signature": o.signature, "description": o.description,
             "available": o.available, "platform_only": o.platform_only, "smooth": o.smooth}
            for o in sorted(OPERATORS.values(), key=lambda o: (o.category, o.name))]
