"""Recreate / Modify a pasted alpha.

modify  - keep the alpha, apply the smallest set of edits that fixes its failures.
rebuild - keep the *idea* but produce a new alpha: variants of the original
          (horizon, normalization, field swaps, gating) plus fresh builds of the
          matching theses, required to be decorrelated from the original.
"""
from __future__ import annotations

import time

import numpy as np

from ..engine.evaluator import EvalError, FieldCatalog, parallel_map
from ..engine.simulate import OfflineEngine
from ..expr.ast import BinOp, Call, Ident, Node, inline, to_str, transform, walk
from ..expr.classify import Structure, classify
from ..expr.fingerprint import logic_fingerprint
from ..expr.parser import parse
from ..rules.engine import Rules, evaluate, summarize
from ..settings_model import SimSettings
from .fields import ROLE_FIELDS, build_pool
from .generator import Generator
from .ideas import THESES, C, F
from .repair import assess, auto_repair, propose, rank_fixes, scale_windows
from .scoring import score

THESIS_MATCH = {
    "short-term reversal": ["short_reversal", "zscore_reversion", "trend_residual", "volume_shock_reversal",
                            "size_neutral_reversal", "extreme_move"],
    "short-term continuation": ["momentum", "volume_confirmed_trend"],
    "price-volume divergence": ["pv_divergence", "volume_confirmed_trend", "volume_shock_reversal"],
    "momentum": ["momentum", "near_high", "volume_confirmed_trend"],
    "long-term reversal": ["momentum", "near_high"],
    "volatility anomaly": ["low_volatility", "residual_signal"],
    "volume/liquidity effect": ["illiquidity", "volume_shock_reversal", "volume_event"],
    "price pattern": ["range_position", "vwap_gap", "overnight_gap", "trend_residual"],
    "valuation (value)": ["value_book", "earnings_yield", "valuation_vs_history", "quality_value"],
    "profitability (quality)": ["profitability", "peer_relative", "quality_value", "fundamental_momentum"],
    "accruals / earnings quality": ["accruals", "earnings_yield"],
    "fundamental momentum / growth": ["fundamental_momentum", "asset_growth", "post_report"],
    "fundamental ratio": ["peer_relative", "profitability", "earnings_yield", "residual_signal"],
    "estimate revisions": ["estimate_revisions", "target_upside"],
    "analyst consensus": ["estimate_revisions", "target_upside"],
    "sentiment": ["sentiment_level", "sentiment_surprise", "news_event", "attention_reversal"],
    "positioning / options": ["options_signal", "short_interest"],
    "conditional / event-gated signal": ["post_report", "volume_event", "news_event", "extreme_move"],
    "multi-factor composite": ["quality_value", "residual_signal", "peer_relative"],
}


def _role_of(name: str) -> str | None:
    for role, names in ROLE_FIELDS.items():
        if name in names:
            return role
    return None


def variants(node: Node, catalog: FieldCatalog, rng: np.random.Generator) -> list[tuple[Node, str]]:
    """Structural variations that keep the original idea."""
    tree = inline(node)
    out: list[tuple[Node, str]] = []
    for f, label in ((0.5, "half the lookback horizon"), (2.0, "double the lookback horizon"),
                     (3.0, "triple the lookback horizon")):
        v = scale_windows(tree, f)
        if v != tree:
            out.append((v, f"Same construction at {label}"))
    core = tree.args[0] if isinstance(tree, Call) and tree.name in ("rank", "zscore", "quantile", "scale") \
        and tree.args else tree
    out.append((C("group_rank", core, F("subindustry")), "Same signal, ranked within sub-industry peers"))
    out.append((C("quantile", C("ts_rank", core, 120)), "Same signal, judged against each stock's own history"))
    out.append((C("trade_when", BinOp(">", F("volume"), F("adv20")), C("rank", core), -1),
                "Same signal, updated only on above-average-volume days"))
    out.append((C("rank", C("ts_decay_exp_window", core, 10, factor=0.7)),
                "Same signal, exponentially smoothed"))
    pool = build_pool(catalog)
    names = [n.name for n in walk(tree) if isinstance(n, Ident)]
    for name in dict.fromkeys(names):
        role = _role_of(name) or ("balance" if catalog.category(name) == "fundamental" else None)
        alts = [a for a in pool.available(role or "") if a != name]
        if role and alts:
            alt = alts[int(rng.integers(0, len(alts)))]
            swapped = transform(tree, lambda n, a=alt, o=name: Ident(a) if isinstance(n, Ident) and n.name == o else None)
            out.append((swapped, f"Same construction with {alt} in place of {name}"))
    return out


def recreate(engine: OfflineEngine, rules: Rules, catalog: FieldCatalog, gen: Generator, expression: str,
             settings: SimSettings, mode: str, corr_fn=None) -> dict:
    started = time.time()
    node = parse(expression)
    structure: Structure = classify(node, catalog.category)
    base = assess(engine, rules, node, settings, corr_fn, full=True)
    original = {"expression": to_str(node), "settings": settings.to_dict(), "sim": base["sim"].to_dict(),
                "checks": [c.to_dict() for c in base["checks"]], "summary": summarize(base["checks"])}

    if mode == "modify":
        fixes = propose(node, settings, base["checks"], base["sim"].metrics, catalog, base["correlation"])
        options = rank_fixes(engine, rules, base, fixes, corr_fn, limit=8)
        repaired = auto_repair(engine, rules, node, settings, catalog, corr_fn, rounds=4, current=base)
        final = assess(engine, rules, repaired["node"], repaired["settings"], corr_fn, full=True)
        return {
            "mode": "modify", "structure": structure.to_dict(), "description": structure.describe(),
            "original": original,
            "result": _pack(repaired["node"], repaired["settings"], final),
            "changes": repaired["history"],
            "unchanged": not repaired["history"],
            "options": [_fix_row(f) for f in options],
            "elapsed_s": round(time.time() - started, 1),
        }

    # ---------------------------------------------------------------- rebuild
    rng = np.random.default_rng(time.time_ns() % (2 ** 32))
    base_pnl = base["sim"].daily_pnl
    cands: list[tuple[Node, SimSettings, str]] = [(v, settings, why) for v, why in variants(node, catalog, rng)]
    ids = THESIS_MATCH.get(structure.thesis) or [t.id for t in THESES if t.family == structure.family]
    theses = [t for t in THESES if t.id in ids]
    for c in gen.candidates({t.family for t in theses} or {structure.family}, settings, n=8, theses=theses or None):
        cands.append((c.node, c.settings, f"Fresh build of the '{c.thesis.title}' thesis: " + c.construction[0]))
    orig_fp = logic_fingerprint(node)
    cands = [c for c in cands if logic_fingerprint(c[0]) != orig_fp]

    def screen(c):
        n, s, why = c
        try:
            sim = engine.simulate(n, s, quick=True)
        except EvalError:
            return None
        corr_orig = float(np.corrcoef(sim.daily_pnl, base_pnl)[0, 1]) if np.std(base_pnl) > 0 and \
            np.std(sim.daily_pnl) > 0 else 0.0
        checks = evaluate(sim.metrics, s, rules, correlation=corr_fn(sim) if corr_fn else None)
        sc = score(sim.metrics, checks, rules, s) - (2.0 if abs(corr_orig) >= 0.7 else 0.0)
        return sc, n, s, why, corr_orig

    screened = sorted((r for r in parallel_map(screen, cands) if r is not None), key=lambda r: -r[0])
    if not screened:
        return {"mode": "rebuild", "error": "Couldn't build a working variant of this idea offline."}
    _, n, s, why, corr_orig = screened[0]
    repaired = auto_repair(engine, rules, n, s, catalog, corr_fn, rounds=3)
    final = assess(engine, rules, repaired["node"], repaired["settings"], corr_fn, full=True)
    fpnl = final["sim"].daily_pnl
    corr_final = float(np.corrcoef(fpnl, base_pnl)[0, 1]) if np.std(fpnl) > 0 and np.std(base_pnl) > 0 else 0.0
    return {
        "mode": "rebuild", "structure": structure.to_dict(), "description": structure.describe(),
        "original": original, "result": _pack(repaired["node"], repaired["settings"], final),
        "inspiration": why, "changes": repaired["history"],
        "correlation_with_original": corr_final,
        "decorrelated": abs(corr_final) < 0.7,
        "alternatives": [{"expression": to_str(a[1]), "why": a[3], "corr_with_original": round(a[4], 2),
                          "score": round(a[0], 2)} for a in screened[1:6]],
        "elapsed_s": round(time.time() - started, 1),
    }


def _pack(node: Node, settings: SimSettings, a: dict) -> dict:
    return {"expression": to_str(node), "settings": settings.to_dict(), "sim": a["sim"].to_dict(),
            "checks": [c.to_dict() for c in a["checks"]], "summary": summarize(a["checks"]),
            "correlation": a["correlation"], "fingerprint": logic_fingerprint(node)}


def _fix_row(f) -> dict:
    return {"title": f.title, "detail": f.detail, "change": f.change, "size": f.size, "targets": f.targets,
            **{k: v for k, v in f.result.items()}}
