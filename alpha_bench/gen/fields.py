"""Field pools by economic role.

Offline (not signed in): the synthetic market's fields. Price/volume and core
fundamental names mirror common BRAIN fields; analyst/sentiment/option names are
placeholders and are flagged as such.

Signed in (catalog synced): only fields the account can access are used. Roles
are filled from exact name matches plus platform fields whose dataset category
and description fit the role.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..engine.evaluator import FieldCatalog
from ..engine.market import FIELD_INFO

ROLE_FIELDS: dict[str, list[str]] = {
    "price": ["close", "open", "high", "low", "vwap"],
    "returns": ["returns"],
    "volume": ["volume"],
    "adv": ["adv20"],
    "cap": ["cap"],
    "balance": ["assets", "liabilities", "equity", "debt", "cash", "retained_earnings"],
    "flow": ["sales", "operating_income", "income", "cashflow_op", "ebitda"],
    "per_share": ["eps", "bookvalue_ps"],
    "ev": ["enterprise_value"],
    "ratio": ["current_ratio", "sales_growth"],
    "estimate": ["est_eps", "est_sales", "est_ptg"],
    "coverage": ["num_analysts"],
    "sentiment": ["news_sentiment"],
    "attention": ["snt_buzz", "news_count", "social_buzz"],
    "option": ["implied_volatility_call_120", "implied_volatility_put_120", "implied_volatility_call_720",
               "implied_volatility_put_720", "parkinson_volatility_120", "implied_volatility", "put_call_ratio"],
    "positioning": ["short_interest"],
    "model": [],
}

# Offline names that are placeholders rather than real BRAIN field ids. (snt_buzz and the
# implied_volatility_*_120/720 and parkinson_volatility_120 fields are real ids, used in Learn2Quant.)
PLACEHOLDER_FIELDS = {"est_eps", "est_sales", "est_ptg", "num_analysts", "news_sentiment", "news_count",
                      "social_buzz", "implied_volatility", "put_call_ratio", "short_interest"}

_ROLE_CATEGORIES = {
    "estimate": {"analyst", "earnings"}, "coverage": {"analyst"},
    "sentiment": {"sentiment", "news", "socialmedia"}, "attention": {"news", "socialmedia", "sentiment"},
    "option": {"option"}, "positioning": {"shortinterest", "insiders", "institutions", "other"},
    "model": {"model", "risk"}, "balance": {"fundamental"}, "flow": {"fundamental"},
}
_BALANCE_WORDS = re.compile(r"asset|liabilit|debt|equity|cash and|inventor|receivable|payable|book value|goodwill", re.I)
_FLOW_WORDS = re.compile(r"income|revenue|sales|earnings|ebit|cash flow|profit|expense|dividend|capex", re.I)
_ATTENTION_WORDS = re.compile(r"count|number of|volume|buzz|mentions|articles", re.I)


@dataclass
class FieldPool:
    roles: dict[str, list[str]]
    platform: bool
    info: dict[str, dict] = field(default_factory=dict)

    def available(self, role: str) -> list[str]:
        return self.roles.get(role, [])

    def has(self, *roles: str) -> bool:
        return all(self.roles.get(r) for r in roles)

    def placeholder_used(self, names) -> list[str]:
        return [] if self.platform else sorted(set(names) & PLACEHOLDER_FIELDS)


def build_pool(catalog: FieldCatalog) -> FieldPool:
    if not catalog.platform:
        return FieldPool(roles={k: list(v) for k, v in ROLE_FIELDS.items()}, platform=False,
                         info={k: {"category": v[0]} for k, v in FIELD_INFO.items()})
    plat = catalog.platform
    roles: dict[str, list[str]] = {}
    for role, names in ROLE_FIELDS.items():
        chosen = [n for n in names if n in plat and plat[n].get("type", "MATRIX") == "MATRIX"]
        cats = _ROLE_CATEGORIES.get(role)
        if cats:
            for fid, meta in plat.items():
                if meta.get("type", "MATRIX") != "MATRIX" or fid in chosen or meta["category"] not in cats:
                    continue
                desc = meta.get("description") or fid
                if role == "balance" and not _BALANCE_WORDS.search(desc):
                    continue
                if role == "flow" and not _FLOW_WORDS.search(desc):
                    continue
                if role == "attention" and not _ATTENTION_WORDS.search(desc):
                    continue
                if role == "sentiment" and _ATTENTION_WORDS.search(desc):
                    continue
                cov = meta.get("coverage")
                if cov is not None and float(cov) < 0.3:
                    continue
                chosen.append(fid)
        roles[role] = chosen
    return FieldPool(roles=roles, platform=True, info=plat)


def novelty_weight(pool: FieldPool, name: str) -> float:
    """Prefer well-covered, less crowded fields (lower platform correlation risk)."""
    meta = pool.info.get(name) or {}
    cov = meta.get("coverage")
    crowd = meta.get("alphaCount") or meta.get("userCount")
    w = 1.0
    if cov is not None:
        w *= 0.4 + float(cov)
    if crowd:
        w *= 1.0 / (1.0 + float(crowd) / 2000.0)
    return w
