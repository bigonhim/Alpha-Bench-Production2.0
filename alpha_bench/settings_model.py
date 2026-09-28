"""Simulation settings shared by the offline engine and the BRAIN API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

UNIVERSES = ["TOP3000", "TOP1000", "TOP500", "TOP200", "TOPSP500"]
UNIVERSE_SIZE = {"TOP3000": 3000, "TOP1000": 1000, "TOP500": 500, "TOP200": 200, "TOPSP500": 500}
# The next smaller universe BRAIN uses for the sub-universe robustness test.
SUB_UNIVERSE = {"TOP3000": "TOP1000", "TOP1000": "TOP500", "TOP500": "TOP200", "TOPSP500": "TOP200"}
NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
# Risk-model neutralizations offered in BRAIN's settings (Learn2Quant Lesson 9 shows them in the
# dropdown). The offline engine approximates them with SUBINDUSTRY; BRAIN applies the real models.
RISK_NEUTRALIZATIONS = ["STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST"]
REGIONS = ["USA", "GLB", "EUR", "ASI", "CHN", "JPN", "KOR", "TWN", "HKG", "AMR"]


@dataclass(frozen=True)
class SimSettings:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "INDUSTRY"
    # BRAIN's "truncation" is the max weight any single stock may take, as a
    # fraction of book size. It is the platform's max-stock-weight control.
    # Learn2Quant's worked examples "restrict the maximum capital on any single stock to 1%".
    truncation: float = 0.01
    pasteurization: str = "ON"
    nan_handling: str = "OFF"
    unit_handling: str = "VERIFY"
    instrument_type: str = "EQUITY"
    language: str = "FASTEXPR"
    max_trade: str = "OFF"
    # Learn2Quant's settings dialog shows a 1-year test period (the last year is held out).
    test_period: str = "P1Y0M0D"

    def with_(self, **changes: Any) -> "SimSettings":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SimSettings":
        raw = raw or {}
        known = cls.__dataclass_fields__
        clean = {k: v for k, v in raw.items() if k in known}
        if "delay" in clean:
            clean["delay"] = int(clean["delay"])
        if "decay" in clean:
            clean["decay"] = int(clean["decay"])
        if "truncation" in clean:
            clean["truncation"] = float(clean["truncation"])
        return cls(**clean)

    def to_brain(self) -> dict[str, Any]:
        """Settings block for POST /simulations."""
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": False,
            "testPeriod": self.test_period,
            "maxTrade": self.max_trade,
        }

    @classmethod
    def from_brain(cls, raw: dict[str, Any]) -> "SimSettings":
        return cls(
            region=raw.get("region", "USA"),
            universe=raw.get("universe", "TOP3000"),
            delay=int(raw.get("delay", 1)),
            decay=int(raw.get("decay", 0)),
            neutralization=raw.get("neutralization", "INDUSTRY"),
            truncation=float(raw.get("truncation", 0.08)),
            pasteurization=raw.get("pasteurization", "ON"),
            nan_handling=raw.get("nanHandling", "OFF"),
            unit_handling=raw.get("unitHandling", "VERIFY"),
            instrument_type=raw.get("instrumentType", "EQUITY"),
            language=raw.get("language", "FASTEXPR"),
            max_trade=raw.get("maxTrade", "OFF"),
            test_period=raw.get("testPeriod", "P0Y0M0D"),
        )

    def test_period_months(self) -> int:
        """Months held out as the test period, from an ISO-8601 duration like P1Y0M0D."""
        import re

        m = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", (self.test_period or "").upper())
        if not m:
            return 0
        return int(m.group(1) or 0) * 12 + int(m.group(2) or 0)
