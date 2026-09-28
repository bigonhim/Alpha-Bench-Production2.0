"""Paths and user preferences.

Preferences live in a small JSON file in the per-user app directory. Nothing
secret is ever written here: credentials go to the OS keyring and session
cookies go to an encrypted file (see brain/session_store.py).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_NAME = "AlphaBench"
KEYRING_SERVICE = "alpha-bench.worldquantbrain"
BRAIN_API_BASE = "https://api.worldquantbrain.com"
BRAIN_WEB_BASE = "https://platform.worldquantbrain.com"


def app_dir() -> Path:
    override = os.environ.get("ALPHA_BENCH_HOME")
    if override:
        path = Path(override)
    elif os.name == "nt":
        path = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "alpha-bench"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Preferences:
    # BRAIN accounts are commonly limited to ~3 concurrent simulations.
    max_concurrent_simulations: int = 3
    # Seconds between automatic "has the user finished verification?" checks.
    verification_poll_seconds: int = 7
    # Give up auto-polling after this long; the user can still click Retry.
    verification_poll_cap_seconds: int = 240
    # Region / universe / delay used when fetching the account's data fields.
    default_region: str = "USA"
    default_universe: str = "TOP3000"
    default_delay: int = 1
    # Offline synthetic market size (stocks x trading days in-sample).
    offline_stocks: int = 600
    offline_days: int = 1260
    offline_seed: int = 7

    @classmethod
    def load(cls, path: Path | None = None) -> "Preferences":
        path = path or app_dir() / "preferences.json"
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        path = path or app_dir() / "preferences.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
