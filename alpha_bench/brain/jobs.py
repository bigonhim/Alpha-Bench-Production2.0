"""Live-simulation job queue.

* At most `limit` simulations are in flight at once (BRAIN accounts are
  commonly capped around 3; configurable).
* A 401 mid-job pauses the job without losing its place: the progress URL is
  kept, and the job resumes polling once the user has signed in again.
* Every state change is reported through `on_update` so the app can persist
  jobs and survive a restart.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..settings_model import SimSettings
from .auth import AuthManager
from .client import BrainClient, Cancelled
from .errors import BrainError, NotSignedIn, SessionExpired

log = logging.getLogger(__name__)


@dataclass
class SimJob:
    expression: str
    settings: SimSettings
    library_id: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "queued"   # queued | starting | running | paused | done | error | cancelled
    progress_url: str | None = None
    brain_alpha_id: str | None = None
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "expression": self.expression, "settings": self.settings.to_dict(),
                "library_id": self.library_id, "state": self.state, "progress_url": self.progress_url,
                "brain_alpha_id": self.brain_alpha_id, "error": self.error,
                "created_at": self.created_at, "updated_at": self.updated_at}


class SimulationQueue:
    def __init__(self, client: BrainClient, auth: AuthManager, *, limit: int = 3,
                 on_update: Callable[[SimJob], None] | None = None):
        self.client = client
        self.auth = auth
        self.limit = max(1, int(limit))
        self.on_update = on_update
        self._jobs: dict[str, SimJob] = {}
        self._slots = threading.Condition()
        self._active = 0

    def set_limit(self, limit: int) -> None:
        with self._slots:
            self.limit = max(1, min(10, int(limit)))
            self._slots.notify_all()

    def submit(self, expression: str, settings: SimSettings, *, library_id: int | None = None,
               job: SimJob | None = None) -> SimJob:
        job = job or SimJob(expression=expression, settings=settings, library_id=library_id)
        self._jobs[job.id] = job
        self._emit(job)
        threading.Thread(target=self._run, args=(job,), name=f"sim-{job.id}", daemon=True).start()
        return job

    def resume(self, job: SimJob) -> SimJob:
        """Re-attach a job persisted before a restart (keeps its progress URL)."""
        job.state = "queued"
        job.error = None
        return self.submit(job.expression, job.settings, job=job)

    def get(self, job_id: str) -> SimJob | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: -j.created_at)]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.state in ("done", "error", "cancelled"):
            return False
        job.cancel_requested = True
        return True

    # ----------------------------------------------------------- internals
    def _emit(self, job: SimJob) -> None:
        job.updated_at = time.time()
        if self.on_update:
            try:
                self.on_update(job)
            except Exception:  # noqa: BLE001
                log.exception("job update hook failed")

    def _acquire(self, job: SimJob) -> bool:
        with self._slots:
            while self._active >= self.limit:
                if job.cancel_requested:
                    return False
                self._slots.wait(timeout=1.0)
            self._active += 1
            return True

    def _release(self) -> None:
        with self._slots:
            self._active = max(0, self._active - 1)
            self._slots.notify_all()

    def _pause_for_auth(self, job: SimJob) -> None:
        job.state = "paused"
        job.error = "Paused: waiting for you to sign in to BRAIN again. Nothing is lost."
        self._emit(job)
        while not job.cancel_requested and not self.auth.wait_until_signed_in(timeout=2.0):
            pass
        if not job.cancel_requested:
            job.state = "queued"
            job.error = None
            self._emit(job)

    def _run(self, job: SimJob) -> None:
        while True:
            if job.cancel_requested:
                job.state = "cancelled"
                self._emit(job)
                return
            if not self.auth.is_signed_in:
                self._pause_for_auth(job)
                continue
            if not self._acquire(job):
                continue
            try:
                if not job.progress_url:
                    job.state = "starting"
                    self._emit(job)
                    job.progress_url = self.client.start_simulation(job.expression, job.settings)
                job.state = "running"
                self._emit(job)
                done = self.client.wait_simulation(job.progress_url,
                                                   should_stop=lambda: job.cancel_requested)
                job.brain_alpha_id = done.get("alpha")
                job.result = self.client.get_alpha(job.brain_alpha_id) if job.brain_alpha_id else done
                job.state = "done"
                job.error = None
                self._emit(job)
                return
            except (SessionExpired, NotSignedIn):
                # Free the slot while waiting; the progress URL is kept so polling resumes.
                self._release()
                self._pause_for_auth(job)
                continue
            except Cancelled:
                job.state = "cancelled"
                self._emit(job)
                return
            except BrainError as exc:
                job.state = "error"
                job.error = exc.user_message
                self._emit(job)
                return
            except Exception:  # noqa: BLE001
                log.exception("simulation job crashed")
                job.state = "error"
                job.error = "Unexpected error while running the simulation; see the log for details."
                self._emit(job)
                return
            finally:
                if job.state in ("done", "error", "cancelled"):
                    self._release()
