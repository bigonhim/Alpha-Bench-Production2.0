"""Create: generate a new alpha from scratch without repeating earlier logic.

Each call:
1. Picks theses with probability inversely related to how often they (and their
   fields) were used this session, never repeating the last few theses.
2. Builds, dresses and parameterises candidates; any candidate whose logic
   fingerprint was produced before is discarded.
3. Screens candidates offline, penalising PnL correlation with the user's
   alphas and with everything generated earlier this session.
4. Polishes the best with minimal-change repairs and explains the result.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from ..engine.evaluator import EvalError, FieldCatalog, parallel_map
from ..engine.simulate import OfflineEngine
from ..expr.ast import Node, to_str
from ..expr.fingerprint import logic_fingerprint, operator_set, shape_fingerprint
from ..expr.operators import OPERATORS
from ..rules.engine import Rules, evaluate, summarize
from ..settings_model import SimSettings
from .fields import FieldPool, build_pool
from .ideas import FAMILIES, THESES, C, Ctx, F, Skip, Thesis
from .repair import assess, auto_repair, neutralization_note
from .scoring import OfflinePnlPool, score


@dataclass
class IdeaMemory:
    logic: set[str] = field(default_factory=set)
    shapes: Counter = field(default_factory=Counter)
    theses: Counter = field(default_factory=Counter)
    fields: Counter = field(default_factory=Counter)
    recent: deque = field(default_factory=lambda: deque(maxlen=4))
    buckets: Counter = field(default_factory=Counter)   # Learn2Quant L8 turnover buckets produced so far

    def remember(self, node: Node, thesis_id: str, fields_used: list[str], shape: str) -> None:
        self.logic.add(logic_fingerprint(node))
        self.shapes[shape] += 1
        self.theses[thesis_id] += 1
        self.recent.append(thesis_id)
        for f in fields_used:
            self.fields[f] += 1

    def seen(self, node: Node) -> bool:
        return logic_fingerprint(node) in self.logic


@dataclass
class Candidate:
    node: Node
    settings: SimSettings
    thesis: Thesis
    construction: list[str]
    fields: list[str]


def _allowed(node: Node, allowed_ops: set[str] | None) -> bool:
    ops = operator_set(node)
    if any(o in OPERATORS and not OPERATORS[o].available for o in ops):
        return False
    return allowed_ops is None or ops <= allowed_ops


def _dress(ctx: Ctx, core: Node, th: Thesis) -> tuple[Node, list[str]]:
    notes: list[str] = []
    node = core
    if not th.normalized:
        style = ctx.choice(["rank"] * 4 + ["zscore"] * 2 + ["group_rank"] * 2 + ["quantile", "ts_rank"])
        if style == "rank":
            node, note = C("rank", node), "ranked cross-sectionally (bounded, outlier-proof)"
        elif style == "zscore":
            node, note = C("winsorize", C("zscore", node), std=ctx.choice([3, 4])), \
                "z-scored and winsorized, so strength of the signal matters but outliers are clipped"
        elif style == "group_rank":
            g = ctx.choice(["industry", "subindustry", "sector"])
            node, note = C("group_rank", node, F(g)), f"ranked within each {g}, so stocks compete only with peers"
        elif style == "quantile":
            node, note = C("quantile", node), "mapped to a Gaussian by quantile(), which spreads weight evenly"
        else:
            k = ctx.win(60, 120, 252)
            node, note = C("ts_rank", node, k), f"compared with its own {k}-day history (ts_rank)"
        notes.append(note)
    if th.speed == "fast" and ctx.coin(0.3):
        k = ctx.win(3, 5, 8)
        node = C("ts_decay_linear", node, k)
        notes.append(f"smoothed with ts_decay_linear over {k} days to tame turnover")
    if ctx.coin(0.08) and th.family != "cross_sectional":
        node = C("group_neutralize", node, C("bucket", C("rank", F("cap")), range="0.1,1,0.1"))
        notes.append("neutralized within size deciles so it doesn't just bet on small caps")
    elif ctx.coin(0.06):
        node = C("multiply", node, C("rank", F("adv20")))
        notes.append("tilted toward liquid names (x rank(adv20)) for sub-universe robustness")
    return node, notes


def _settings(ctx: Ctx, th: Thesis, base: SimSettings, auto_universe: bool) -> SimSettings:
    neut = ctx.choice(list(th.neutralizations))
    # The lessons smooth fast signals with decay 7-10.
    decay = {"fast": ctx.win(3, 4, 6, 8, 10), "medium": ctx.win(0, 2, 4, 7), "slow": ctx.win(0, 0, 1, 2)}[th.speed]
    trunc = base.truncation
    universe = base.universe
    if auto_universe:
        universe = ctx.choice(["TOP3000"] * 6 + ["TOP1000"] * 3 + ["TOP500"])
    return base.with_(neutralization=neut, decay=decay, truncation=trunc, universe=universe)


class Generator:
    def __init__(self, engine: OfflineEngine, rules: Rules, catalog: FieldCatalog, pool: OfflinePnlPool,
                 seed: int | None = None):
        self.engine = engine
        self.rules = rules
        self.catalog = catalog
        self.pnl_pool = pool
        self.memory = IdeaMemory()
        self.rng = np.random.default_rng(seed if seed is not None else time.time_ns() % (2 ** 32))
        self._n = 0

    def field_pool(self) -> FieldPool:
        return build_pool(self.catalog)

    def remember_existing(self, node: Node) -> None:
        """Treat an expression from the library as already derived."""
        self.memory.logic.add(logic_fingerprint(node))

    def _pick_theses(self, families: set[str], pool: FieldPool, k: int) -> list[Thesis]:
        opts = [t for t in THESES if t.family in families and all(pool.has(r) for r in t.roles)]
        if not opts:
            return []
        fresh = [t for t in opts if t.id not in self.memory.recent] or opts
        w = np.array([1.0 / (1.0 + 2.0 * self.memory.theses[t.id]) for t in fresh])
        # Encourage breadth across families too.
        fam_use: Counter = Counter()
        for t in THESES:
            fam_use[t.family] += self.memory.theses[t.id]
        w = w / np.array([1.0 + 0.5 * fam_use[t.family] for t in fresh])
        idx = self.rng.choice(len(fresh), size=min(k, len(fresh)), replace=False, p=w / w.sum())
        return [fresh[i] for i in idx]

    def candidates(self, families: set[str], base: SimSettings, *, n: int = 14,
                   allowed_ops: set[str] | None = None, auto_universe: bool = False,
                   theses: list[Thesis] | None = None) -> list[Candidate]:
        pool = self.field_pool()
        chosen = theses or self._pick_theses(families, pool, max(3, n // 3))
        out: list[Candidate] = []
        tries = 0
        while len(out) < n and tries < n * 6 and chosen:
            tries += 1
            th = chosen[tries % len(chosen)]
            ctx = Ctx(self.rng, pool, self.memory.fields)
            try:
                core, construction = th.build(ctx)
            except Skip:
                continue
            node, notes = _dress(ctx, core, th)
            if self.memory.seen(node) or any(logic_fingerprint(c.node) == logic_fingerprint(node) for c in out):
                continue
            if not _allowed(node, allowed_ops):
                continue
            settings = _settings(ctx, th, base, auto_universe)
            out.append(Candidate(node, settings, th, [construction] + notes + ctx.notes, list(dict.fromkeys(ctx.used))))
        return out

    def _corr_fn(self):
        def fn(sim):
            if self.pnl_pool.count() == 0:
                return None
            return self.pnl_pool.max_corr(sim.daily_pnl)
        return fn

    def create(self, families: set[str] | None = None, base: SimSettings | None = None, *,
               allowed_ops: set[str] | None = None, auto_universe: bool = False, budget_s: float = 25.0,
               n: int = 14) -> dict:
        families = set(families or FAMILIES)
        base = base or SimSettings()
        started = time.time()
        cands = self.candidates(families, base, n=n, allowed_ops=allowed_ops, auto_universe=auto_universe)
        if not cands:
            return {"error": "No idea in the selected families can be built from the fields and operators "
                             "available. Widen the idea space or sync fields from BRAIN."}
        corr_fn = self._corr_fn()

        def screen(cand: Candidate):
            try:
                sim = self.engine.simulate(cand.node, cand.settings, sub_universe=True, quick=True)
            except EvalError:
                return None
            checks = evaluate(sim.metrics, cand.settings, self.rules, correlation=corr_fn(sim))
            return score(sim.metrics, checks, self.rules, cand.settings), cand, sim

        screened = [r for r in parallel_map(screen, cands) if r is not None]
        # Diversity across holding periods (Learn2Quant L8): nudge toward the least-produced turnover bucket.
        if self.memory.buckets:
            least = min(("low (<5%)", "medium (5-15%)", "high (>15%)"), key=lambda b: self.memory.buckets[b])
            screened = [(sc + (0.15 if sim.metrics.get("turnover_bucket") == least else 0.0), c, sim)
                        for sc, c, sim in screened]
        if not screened:
            return {"error": "Every candidate failed to evaluate offline. Try again or widen the idea space."}
        screened.sort(key=lambda r: -r[0])

        best = None
        for sc, cand, sim in screened[:2]:
            if time.time() - started > budget_s and best is not None:
                break
            checks = evaluate(sim.metrics, cand.settings, self.rules, correlation=corr_fn(sim))
            seed = {"sim": sim, "checks": checks, "correlation": corr_fn(sim), "score": sc}
            fixed = auto_repair(self.engine, self.rules, cand.node, cand.settings, self.catalog, corr_fn,
                                rounds=3, per_round=8, current=seed)
            if best is None or fixed["assessment"]["score"] > best[1]["assessment"]["score"]:
                best = (cand, fixed)
            if not any(c.result == "FAIL" for c in fixed["assessment"]["checks"]):
                break
        cand, fixed = best
        node, settings = fixed["node"], fixed["settings"]
        a = assess(self.engine, self.rules, node, settings, corr_fn, full=True)
        pool = self.field_pool()
        shape = shape_fingerprint(node, self.catalog.category)
        self.memory.logic.add(logic_fingerprint(cand.node))  # the raw idea, before repairs
        self.memory.remember(node, cand.thesis.id, cand.fields, shape)
        self.memory.buckets[a["sim"].metrics.get("turnover_bucket")] += 1
        self._n += 1
        self.pnl_pool.add(f"session-{self._n}", f"Created #{self._n}: {cand.thesis.title}", a["sim"].daily_pnl,
                          a["sim"].metrics["sharpe"], "session")
        rationale = self.rationale(cand, settings, fixed["history"])
        warnings = list(a["sim"].warnings)
        ph = pool.placeholder_used(cand.fields)
        if ph:
            warnings.append("Uses offline placeholder field names (" + ", ".join(ph) + "). Sign in and sync fields so "
                            "Create uses fields your BRAIN account actually has.")
        return {
            "expression": to_str(node), "settings": settings.to_dict(), "family": cand.thesis.family,
            "family_label": FAMILIES[cand.thesis.family], "thesis": cand.thesis.title, "thesis_id": cand.thesis.id,
            "rationale": rationale, "fields": cand.fields, "repairs": fixed["history"],
            "sim": a["sim"].to_dict(), "checks": [c.to_dict() for c in a["checks"]],
            "summary": summarize(a["checks"]), "correlation": a["correlation"], "warnings": warnings,
            "screened": len(screened), "fingerprint": logic_fingerprint(node),
            "alternatives": [{"expression": to_str(c.node), "thesis": c.thesis.title, "score": round(s, 3),
                              "sharpe": sim.metrics["sharpe"], "fitness": sim.metrics["fitness"]}
                             for s, c, sim in screened[1:5]],
            "elapsed_s": round(time.time() - started, 1),
        }

    @staticmethod
    def rationale(cand: Candidate, settings: SimSettings, repairs: list[dict]) -> dict:
        th = cand.thesis
        decay_note = ("No decay: the signal is already slow-moving." if settings.decay == 0 else
                      f"Decay {settings.decay} averages positions over {settings.decay} days to cut turnover"
                      + (" (a fast signal needs it)." if th.speed == "fast" else "."))
        return {
            "thesis": th.title,
            "why_it_works": th.mechanism,
            "construction": "; ".join(cand.construction) + ".",
            "settings": [neutralization_note(settings.neutralization), decay_note,
                         f"Truncation {settings.truncation} caps any single stock at "
                         f"{settings.truncation:.0%} of the book.",
                         f"Universe {settings.universe}, delay {settings.delay}."],
            "repairs": [f"{r['change']}: {r['detail']}" for r in repairs],
        }


__all__ = ["Generator", "IdeaMemory", "Candidate"]
