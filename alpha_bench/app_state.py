"""Application state: wires auth, BRAIN client, job queue, offline engine, library and rules."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .brain.auth import AuthManager
from .brain.client import BrainClient, alpha_code, alpha_metrics
from .brain.credentials import CredentialStore, KeyringBackend, MemoryBackend
from .brain.errors import BrainError
from .brain.jobs import SimJob, SimulationQueue
from .brain.session_store import SessionStore
from .brain.transport import RequestsTransport, Transport
from .config import Preferences
from .engine.evaluator import EvalError, FieldCatalog
from .engine.market import load_or_generate
from .engine.simulate import OfflineEngine
from .expr.fingerprint import logic_fingerprint
from .expr.operators import merge_platform_operators
from .expr.parser import ParseError, parse
from .gen.generator import Generator
from .gen.scoring import OfflinePnlPool
from .rules.correlation import PnlSeries, max_correlation
from .rules.engine import Rules, brain_checks_view
from .settings_model import SimSettings
from .store.library import Library

log = logging.getLogger(__name__)


class Tasks:
    """Tiny registry for background tasks the UI polls (sync fields, load alphas...)."""

    def __init__(self) -> None:
        self._t: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Callable[[float, str], None]], Any]) -> dict:
        tid = uuid.uuid4().hex[:10]
        task = {"id": tid, "kind": kind, "state": "running", "progress": 0.0, "message": "Starting...",
                "result": None, "error": None, "started": time.time()}
        with self._lock:
            self._t[tid] = task

        def progress(p: float, msg: str) -> None:
            task["progress"], task["message"] = float(p), msg

        def run():
            try:
                task["result"] = fn(progress)
                task["state"], task["progress"] = "done", 1.0
            except BrainError as exc:
                task["state"], task["error"] = "error", exc.user_message
            except Exception as exc:  # noqa: BLE001
                log.exception("task %s failed", kind)
                task["state"], task["error"] = "error", f"{type(exc).__name__}: {exc}"
        threading.Thread(target=run, name=f"task-{kind}", daemon=True).start()
        return task

    def get(self, tid: str) -> dict | None:
        return self._t.get(tid)

    def active(self) -> list[dict]:
        cutoff = time.time() - 600
        return [dict(t, result=None) for t in self._t.values() if t["state"] == "running" or t["started"] > cutoff]


class AppState:
    def __init__(self, home: Path, prefs: Preferences, *, transport: Transport | None = None,
                 use_keyring: bool = True):
        self.home = home
        self.prefs = prefs
        backend = KeyringBackend() if use_keyring else MemoryBackend()
        if use_keyring and not backend.available:
            log.warning("No OS keyring available: credentials and sessions won't be remembered.")
        self.transport = transport or RequestsTransport()
        self.creds = CredentialStore(backend)
        self.auth = AuthManager(self.transport, self.creds, SessionStore(home / "session.bin", backend),
                                state_file=home / "signin_guard.json")
        self.client = BrainClient(self.transport, self.auth)
        self.library = Library(home / "library.db")
        self.rules = Rules(home)
        self.catalog = FieldCatalog()
        self.tasks = Tasks()
        self.pnl_pool = OfflinePnlPool()
        self.brain_pnls: dict[str, PnlSeries] = {}
        self.queue = SimulationQueue(self.client, self.auth, limit=prefs.max_concurrent_simulations,
                                     on_update=self._on_job)
        self.engine: OfflineEngine | None = None
        self.generator: Generator | None = None
        self.ready = threading.Event()
        self.load_error: str | None = None
        self.compute_lock = threading.Lock()   # one heavy offline job at a time (memory)
        self._load_platform_cache()

    # ------------------------------------------------------------ startup
    def start(self) -> None:
        threading.Thread(target=self._boot, name="boot", daemon=True).start()

    def _boot(self) -> None:
        try:
            self.auth.restore()
        except Exception:  # noqa: BLE001
            log.exception("session restore failed")
        try:
            market = load_or_generate(self.home, self.prefs.offline_stocks, self.prefs.offline_days,
                                      self.prefs.offline_seed)
            self.engine = OfflineEngine(market, self.catalog)
            self.generator = Generator(self.engine, self.rules, self.catalog, self.pnl_pool)
            for fp in self.library.fingerprints():
                self.generator.memory.logic.add(fp)
            self.ready.set()
            self._rebuild_offline_pool()
        except Exception as exc:  # noqa: BLE001
            log.exception("offline engine failed to load")
            self.load_error = f"The offline engine failed to start: {exc}"
        for row in self.library.unfinished_jobs():
            job = SimJob(expression=row["expression"], settings=SimSettings.from_dict(row["settings"]),
                         library_id=row["alpha_id"], id=row["id"], progress_url=row["progress_url"],
                         created_at=row["created_at"] or time.time())
            self.queue.resume(job)
        for k in self.library.known("brain"):
            if k.get("pnl"):
                self.brain_pnls[k["id"]] = PnlSeries.from_cumulative(k["id"], k["name"], k["pnl"], k.get("sharpe"))

    def require_engine(self) -> OfflineEngine:
        if not self.ready.wait(timeout=120):
            raise RuntimeError(self.load_error or "The offline engine is still loading.")
        return self.engine  # type: ignore[return-value]

    # ------------------------------------------------------- platform meta
    def _meta_path(self) -> Path:
        p = self.prefs
        return self.home / f"platform_fields_{p.default_region}_{p.default_universe}_d{p.default_delay}.json"

    def _load_platform_cache(self) -> None:
        path = self._meta_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.catalog.load_platform(data.get("fields") or [])
                merge_platform_operators(data.get("operators") or [])
            except (OSError, ValueError):
                pass

    def sync_platform(self, progress) -> dict:
        p = self.prefs
        progress(0.05, "Fetching operators available to your account...")
        ops = self.client.list_operators()
        stats = merge_platform_operators(ops)
        progress(0.2, "Fetching data fields (this can take a minute)...")
        fields = self.client.list_data_fields(p.default_region, p.default_universe, p.default_delay)
        self.catalog.load_platform(fields)
        self._meta_path().write_text(json.dumps({"fields": fields, "operators": ops, "synced": time.time()}),
                                     encoding="utf-8")
        progress(1.0, "Done")
        return {"fields": len(fields), "operators": stats}

    # ------------------------------------------------------- correlation
    def _corr_fn(self):
        pool = self.pnl_pool

        def fn(sim):
            return pool.max_corr(sim.daily_pnl, kinds=("active", "pasted")) if pool.count() else None
        return fn

    def _rebuild_offline_pool(self) -> None:
        """Simulate known alphas (yours on BRAIN + pasted) offline so candidates can be screened for correlation."""
        engine = self.engine
        if engine is None:
            return
        for k in self.library.known():
            try:
                sim = engine.simulate(k["expression"], SimSettings.from_dict(k["settings"]), sub_universe=False,
                                      quick=True)
            except (EvalError, ParseError, ValueError):
                continue
            kind = "active" if k["source"] == "brain" else "pasted"
            self.pnl_pool.add(k["id"], k["name"], sim.daily_pnl, k.get("sharpe") or sim.metrics["sharpe"], kind)

    def load_active_alphas(self, progress) -> dict:
        progress(0.02, "Listing your submitted (active) alphas...")
        alphas = list(self.client.iter_user_alphas(stage="OS", status="ACTIVE"))
        self.library.clear_known("brain")
        self.brain_pnls.clear()
        self.pnl_pool.remove_kind("active")
        for i, a in enumerate(alphas):
            aid = a.get("id")
            progress(0.05 + 0.9 * i / max(1, len(alphas)), f"Downloading PnL {i + 1}/{len(alphas)} ({aid})")
            code = alpha_code(a)
            settings = SimSettings.from_brain(a.get("settings") or {})
            sharpe = (a.get("is") or {}).get("sharpe")
            try:
                pnl = self.client.get_pnl(aid)
            except BrainError:
                pnl = None
            name = a.get("name") or aid
            self.library.upsert_known(id=aid, name=name, source="brain", expression=code, settings=settings.to_dict(),
                                      sharpe=sharpe, pnl=pnl)
            if pnl:
                self.brain_pnls[aid] = PnlSeries.from_cumulative(aid, name, pnl, sharpe)
        self._rebuild_offline_pool()
        return {"alphas": len(alphas), "with_pnl": len(self.brain_pnls)}

    def add_pasted(self, items: list[dict]) -> dict:
        added, errors = 0, []
        base = len(self.library.known("pasted"))
        for i, it in enumerate(items):
            expr = (it.get("expression") or "").strip()
            if not expr:
                continue
            try:
                parse(expr)
            except ParseError as exc:
                errors.append(f"#{i + 1}: {exc}")
                continue
            pid = f"pasted-{base + i + 1}"
            self.library.upsert_known(id=pid, name=it.get("name") or f"Pasted #{base + i + 1}", source="pasted",
                                      expression=expr, settings=SimSettings.from_dict(it.get("settings")).to_dict(),
                                      sharpe=it.get("sharpe"), pnl=None)
            added += 1
        if self.engine:
            self._rebuild_offline_pool()
        return {"added": added, "errors": errors}

    def verified_correlation(self, brain_alpha_id: str) -> dict | None:
        """Real daily-PnL correlation of a BRAIN-simulated alpha against your active alphas."""
        if not self.brain_pnls:
            return None
        pnl = self.client.get_pnl(brain_alpha_id)
        if not pnl:
            return None
        cand = PnlSeries.from_cumulative(brain_alpha_id, brain_alpha_id, pnl)
        return max_correlation(cand, list(self.brain_pnls.values()), source="brain",
                               window_days=4 * 252)

    # ---------------------------------------------------------------- jobs
    def submit_live(self, expression: str, settings: SimSettings, library_id: int | None) -> SimJob:
        return self.queue.submit(expression, settings, library_id=library_id)

    def _on_job(self, job: SimJob) -> None:
        self.library.save_job(job.to_dict())
        if job.state != "done" or not job.result or not job.library_id:
            if job.state == "error" and job.library_id:
                self.library.update_alpha(job.library_id, status="brain-error", notes=job.error)
            return
        alpha = job.result
        checks = brain_checks_view(alpha)
        metrics = alpha_metrics(alpha)
        failed = [c["name"] for c in checks if str(c.get("result")).upper() == "FAIL"]
        # Every live result recalibrates the offline engine: log offline-vs-BRAIN for this alpha.
        item = self.library.get(job.library_id)
        offline = ((item or {}).get("offline") or {}).get("metrics") or {}
        if offline:
            keys = ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin", "long_count", "short_count")
            rows = [{"metric": k, "offline": offline.get(k), "brain": metrics.get(k)} for k in keys]
            self.library.log_reconciliation(job.library_id, job.brain_alpha_id, rows, (item or {}).get("family"))
        self.library.update_alpha(job.library_id, brain_alpha_id=job.brain_alpha_id,
                                  brain={"metrics": metrics, "checks": checks, "fetched": time.time()},
                                  status="brain-pass" if not failed else "brain-fail")
        try:
            self.rules.learn_from_brain(checks, job.settings.delay, job.settings.region)
        except Exception:  # noqa: BLE001
            log.exception("learning limits failed")

    # ------------------------------------------------------------- helpers
    def save_result(self, *, expression: str, settings: dict, mode: str, parent_id: int | None = None,
                    **extra) -> int:
        try:
            fp = logic_fingerprint(parse(expression))
        except ParseError:
            fp = None
        return self.library.add_alpha(expression=expression, settings=settings, mode=mode, parent_id=parent_id,
                                      fingerprint=fp, **extra)


def jsonable(x: Any) -> Any:
    """Make numpy scalars/arrays JSON-safe (NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
