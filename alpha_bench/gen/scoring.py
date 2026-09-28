"""Candidate scoring and the offline PnL pool used for correlation screening."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rules.engine import Check, Rules
from ..settings_model import SimSettings

_WEIGHT = {"LOW_SHARPE": 1.6, "LOW_FITNESS": 1.6, "HIGH_TURNOVER": 1.2, "LOW_TURNOVER": 1.2,
           "CONCENTRATED_WEIGHT": 1.0, "LOW_SUB_UNIVERSE_SHARPE": 1.0, "SELF_CORRELATION": 1.4}


def score(metrics: dict, checks: list[Check], rules: Rules, settings: SimSettings) -> float:
    """Higher is better. Failures dominate; margins over the Sharpe/fitness floors break ties."""
    s = 0.0
    for c in checks:
        if c.result == "FAIL":
            s -= _WEIGHT.get(c.name, 0.8)
        elif c.result == "WARN":
            s -= 0.15
    s_need, _ = rules.value("sharpe", settings.delay, settings.region)
    f_need, _ = rules.value("fitness", settings.delay, settings.region)
    s += 0.6 * float(np.clip(metrics.get("sharpe", 0) / s_need, -1, 2.2))
    s += 0.8 * float(np.clip(metrics.get("fitness", 0) / f_need, -1, 2.2))
    s += 0.1 * float(np.clip(metrics.get("margin", 0) / 10.0, -1, 1))
    rd = metrics.get("returns_drawdown")
    if rd is not None:
        s += 0.15 * float(np.clip(rd - 1.0, -1, 1))          # Lesson 3: returns/drawdown > 1
    for c in checks:
        if c.name == "SELF_CORRELATION" and c.value is not None and c.result == "PASS" and c.value > 0.5:
            s -= 0.3 * (c.value - 0.5) / 0.2                   # Lesson 8: < 0.5 diversifies the pool
    return s


@dataclass
class PoolItem:
    id: str
    name: str
    pnl: np.ndarray
    sharpe: float | None
    kind: str  # active | pasted | session


class OfflinePnlPool:
    """Daily PnLs from the offline engine (all on the same synthetic dates)."""

    def __init__(self, window: int = 4 * 252):
        self.window = window
        self.items: dict[str, PoolItem] = {}

    def add(self, id: str, name: str, pnl: np.ndarray, sharpe: float | None, kind: str) -> None:
        self.items[id] = PoolItem(id, name, np.asarray(pnl, dtype=float), sharpe, kind)

    def remove_kind(self, kind: str) -> None:
        self.items = {k: v for k, v in self.items.items() if v.kind != kind}

    def count(self, kind: str | None = None) -> int:
        return sum(1 for v in self.items.values() if kind is None or v.kind == kind)

    def max_corr(self, pnl: np.ndarray, kinds: tuple[str, ...] | None = None, exclude: str | None = None) -> dict:
        items = [v for v in self.items.values() if (kinds is None or v.kind in kinds) and v.id != exclude
                 and len(v.pnl) == len(pnl)]
        if not items:
            return {"max": None, "rows": [], "source": "simulated", "compared": 0}
        w = min(self.window, len(pnl))
        x = pnl[-w:] - pnl[-w:].mean()
        M = np.stack([v.pnl[-w:] for v in items])
        M = M - M.mean(axis=1, keepdims=True)
        den = np.sqrt((M * M).sum(1) * (x * x).sum())
        with np.errstate(all="ignore"):
            corr = np.where(den > 0, (M @ x) / den, 0.0)
        order = np.argsort(-corr)
        rows = [{"id": items[i].id, "name": items[i].name, "corr": float(corr[i]), "sharpe": items[i].sharpe,
                 "kind": items[i].kind} for i in order[:25]]
        top = rows[0]
        return {"max": top["corr"], "with_id": top["id"], "with_name": top["name"], "with_sharpe": top["sharpe"],
                "rows": rows, "source": "simulated", "compared": len(items)}
