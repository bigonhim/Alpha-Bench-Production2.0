"""Self-correlation on daily PnL changes (never on cumulative PnL).

Two tiers:
* simulated: the candidate and your other alphas are all run through the
  offline engine (their expressions are known), then daily PnLs are compared.
* verified: the candidate's BRAIN PnL vs the real BRAIN PnL of your active
  alphas, downloaded from your account.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PnlSeries:
    id: str
    name: str
    dates: list[str]
    daily: np.ndarray
    sharpe: float | None = None

    @classmethod
    def from_cumulative(cls, id: str, name: str, records: Sequence[tuple[str, float]],
                        sharpe: float | None = None) -> "PnlSeries":
        dates = [str(d)[:10] for d, _ in records]
        cum = np.array([float(v) for _, v in records], dtype=float)
        daily = np.diff(cum, prepend=cum[0] if len(cum) else 0.0)
        if len(daily):
            daily[0] = 0.0
        return cls(id, name, dates, daily, sharpe)


def correlation(a: PnlSeries, b: PnlSeries, window_days: int = 4 * 252, min_overlap: int = 120) -> float | None:
    common = sorted(set(a.dates) & set(b.dates))[-window_days:]
    if len(common) < min_overlap:
        return None
    ia = {d: i for i, d in enumerate(a.dates)}
    ib = {d: i for i, d in enumerate(b.dates)}
    x = np.array([a.daily[ia[d]] for d in common])
    y = np.array([b.daily[ib[d]] for d in common])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def max_correlation(candidate: PnlSeries, others: list[PnlSeries], *, source: str,
                    window_days: int = 4 * 252) -> dict:
    rows = []
    for o in others:
        if o.id == candidate.id:
            continue
        c = correlation(candidate, o, window_days)
        if c is not None:
            rows.append({"id": o.id, "name": o.name, "corr": c, "sharpe": o.sharpe})
    rows.sort(key=lambda r: -r["corr"])
    if not rows:
        return {"max": None, "rows": [], "source": source, "compared": 0}
    top = rows[0]
    return {"max": top["corr"], "with_id": top["id"], "with_name": top["name"], "with_sharpe": top["sharpe"],
            "rows": rows[:25], "source": source, "compared": len(rows)}
