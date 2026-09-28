"""Settings sweep: decay x truncation x neutralization x universe x delay.

The expression is evaluated once per (delay, universe); every other setting only
changes portfolio construction, so the sweep is cheap.
"""
from __future__ import annotations

import itertools
import time

from ..engine.evaluator import EvalError, parallel_map
from ..engine.simulate import OfflineEngine
from ..expr.ast import Node
from ..rules.engine import Rules, evaluate
from ..settings_model import SimSettings
from .scoring import score

DEFAULT_GRID = {
    "decay": [0, 2, 4, 6, 8, 12],
    "truncation": [0.01, 0.05, 0.08, 0.1],
    "neutralization": ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"],
    "universe": None,   # None = keep the current universe
    "delay": None,      # None = keep the current delay
}


def sweep(engine: OfflineEngine, rules: Rules, node: Node, base: SimSettings, grid: dict | None = None,
          corr_fn=None, max_runs: int = 400) -> dict:
    started = time.time()
    g = {**DEFAULT_GRID, **(grid or {})}
    universes = g["universe"] or [base.universe]
    delays = g["delay"] or [base.delay]
    combos = list(itertools.product(g["decay"], g["truncation"], g["neutralization"]))
    rows = []
    for delay in delays:
        for universe in universes:
            s0 = base.with_(delay=int(delay), universe=universe)
            try:
                alpha, proxied = engine.alpha_matrix(node, s0)
            except EvalError as exc:
                return {"error": str(exc)}

            def run(combo, s0=s0, alpha=alpha, proxied=proxied):
                decay, trunc, neut = combo
                s = s0.with_(decay=int(decay), truncation=float(trunc), neutralization=neut)
                sim = engine.simulate(node, s, alpha=alpha, proxied=proxied, quick=True)
                corr = corr_fn(sim) if corr_fn else None
                checks = evaluate(sim.metrics, s, rules, correlation=corr)
                m = sim.metrics
                return {"settings": s.to_dict(), "score": score(m, checks, rules, s),
                        "sharpe": m["sharpe"], "fitness": m["fitness"], "returns": m["returns"],
                        "turnover": m["turnover"], "drawdown": m["drawdown"], "margin": m["margin"],
                        "max_weight": m["max_weight"], "sub_universe_sharpe": m.get("sub_universe_sharpe"),
                        "fails": [c.name for c in checks if c.result == "FAIL"]}

            rows.extend(parallel_map(run, combos[: max(1, max_runs // (len(delays) * len(universes)))]))
    rows.sort(key=lambda r: -r["score"])
    # Pareto front on (Sharpe, fitness): nothing else is at least as good on both and better on one.
    for r in rows:
        r["pareto"] = not any(o is not r and o["sharpe"] >= r["sharpe"] and o["fitness"] >= r["fitness"]
                              and (o["sharpe"] > r["sharpe"] or o["fitness"] > r["fitness"]) for o in rows)
    base_row = next((r for r in rows if r["settings"] == base.to_dict()), None)
    return {"rows": rows, "best": rows[0] if rows else None, "base": base_row, "runs": len(rows),
            "passing": sum(1 for r in rows if not r["fails"]), "elapsed_s": round(time.time() - started, 1)}
