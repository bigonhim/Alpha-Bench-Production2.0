"""Authenticated BRAIN API client.

Deliberately has no "submit alpha" method: final submission is always a manual
action the user takes on the BRAIN website, so a bug here can never burn a
submission slot.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, Iterator

from ..config import BRAIN_API_BASE, BRAIN_WEB_BASE
from ..settings_model import SimSettings
from .auth import AuthManager
from .errors import (BrainRequestError, DailyLimitReached, NetworkError, NotSignedIn, RateLimited, ServerError,
                     SessionExpired)

# Correlation and check endpoints answer in their newer format with this Accept header.
V2 = {"Accept": "application/json;version=2.0"}
FAILED_STATUSES = {"ERROR", "FAIL", "FAILED", "TIMEOUT", "CANCELLED"}
from .transport import HttpResponse, Transport

log = logging.getLogger(__name__)


def parse_retry_after(resp: HttpResponse) -> float | None:
    raw = resp.header("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        from email.utils import parsedate_to_datetime

        try:
            return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class Cancelled(Exception):
    pass


class BrainClient:
    def __init__(self, transport: Transport, auth: AuthManager, *, api_base: str = BRAIN_API_BASE,
                 sleep: Callable[[float], None] = time.sleep, max_retries: int = 5,
                 backoff_base: float = 2.0, backoff_cap: float = 120.0,
                 min_poll_interval: float = 2.0, max_poll_seconds: float = 60 * 60):
        self.transport = transport
        self.auth = auth
        self.api = api_base.rstrip("/")
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.min_poll_interval = min_poll_interval
        self.max_poll_seconds = max_poll_seconds
        self._io_lock = threading.Lock()  # requests.Session isn't guaranteed thread-safe
        self.rate_limit: dict[str, str] = {}  # last x-ratelimit-* headers, shown in the UI

    # ---------------------------------------------------------------- core
    def _url(self, path_or_url: str) -> str:
        return path_or_url if path_or_url.startswith("https://") else self.api + "/" + path_or_url.lstrip("/")

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.backoff_cap, retry_after)
        return min(self.backoff_cap, self.backoff_base * (2 ** attempt)) * (0.75 + random.random() * 0.5)

    def request(self, method: str, path_or_url: str, *, json: Any = None,
                params: dict | None = None, headers: dict | None = None) -> HttpResponse:
        if not self.auth.is_signed_in:
            raise NotSignedIn()
        url = self._url(path_or_url)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._io_lock:
                    resp = self.transport.request(method, url, json=json, params=params, headers=headers)
            except NetworkError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self.sleep(self._backoff(attempt, None))
                    continue
                raise
            for h in ("x-ratelimit-remaining", "x-ratelimit-reset", "x-ratelimit-limit"):
                if resp.header(h) is not None:
                    self.rate_limit[h] = resp.header(h)
            if resp.status == 401:
                self.auth.mark_expired()
                raise SessionExpired()
            if resp.status == 429 and "DAILY" in resp.text_snippet(400).upper():
                raise DailyLimitReached()
            if resp.status == 429 or resp.status >= 500:
                retry_after = parse_retry_after(resp)
                last_error = RateLimited(retry_after) if resp.status == 429 else ServerError()
                if attempt < self.max_retries:
                    wait = self._backoff(attempt, retry_after)
                    log.info("BRAIN returned %s; backing off %.1fs", resp.status, wait)
                    self.sleep(wait)
                    continue
                raise last_error
            if resp.status >= 400:
                raise BrainRequestError(resp.status, _error_detail(resp))
            return resp
        raise last_error or ServerError()

    def get_json(self, path: str, params: dict | None = None) -> Any:
        return self.request("GET", path, params=params).json()

    def poll(self, url: str, *, should_stop: Callable[[], bool] | None = None,
             on_progress: Callable[[dict], None] | None = None, headers: dict | None = None) -> HttpResponse:
        """Poll a progress URL, sleeping exactly as long as BRAIN's Retry-After asks."""
        started = time.monotonic()
        while True:
            if should_stop and should_stop():
                raise Cancelled()
            resp = self.request("GET", url, headers=headers)
            retry_after = parse_retry_after(resp)
            if not retry_after:
                return resp
            if on_progress:
                body = resp.json()
                if isinstance(body, dict):
                    on_progress(body)
            if time.monotonic() - started > self.max_poll_seconds:
                raise ServerError("BRAIN didn't finish in time. The job stays queued; you can retry it.")
            self.sleep(max(self.min_poll_interval, retry_after))

    # ------------------------------------------------------------ account
    def whoami(self) -> dict:
        return self.get_json("authentication")

    # --------------------------------------------------------- simulation
    def start_simulation(self, expression: str, settings: SimSettings) -> str:
        """POST a regular simulation; returns the progress URL to poll."""
        payload = {"type": "REGULAR", "settings": settings.to_brain(), "regular": expression}
        resp = self.request("POST", "simulations", json=payload)
        location = resp.header("Location")
        if not location:
            raise BrainRequestError(resp.status, "BRAIN accepted the simulation but gave no progress URL.")
        return self._url(location)

    def wait_simulation(self, progress_url: str, **kw) -> dict:
        body = self.poll(progress_url, **kw).json() or {}
        status = str(body.get("status", "")).upper()
        if status in FAILED_STATUSES or not body.get("alpha"):
            detail = body.get("message") or f"Simulation ended with status {status or 'unknown'}."
            loc = body.get("location")
            if isinstance(loc, dict) and loc.get("property"):
                detail += f" (at {loc.get('property')})"
            raise BrainRequestError(400, detail)
        return body

    # -------------------------------------------------------------- alphas
    def get_alpha(self, alpha_id: str) -> dict:
        return self.get_json(f"alphas/{alpha_id}")

    def get_recordset(self, alpha_id: str, name: str) -> dict:
        return self.poll(self._url(f"alphas/{alpha_id}/recordsets/{name}")).json() or {}

    def get_pnl(self, alpha_id: str) -> list[tuple[str, float]]:
        """Cumulative PnL series as (date, pnl) pairs."""
        data = self.get_recordset(alpha_id, "pnl")
        props = [p.get("name") for p in (data.get("schema") or {}).get("properties", [])]
        di = props.index("date") if "date" in props else 0
        pi = props.index("pnl") if "pnl" in props else 1
        out = []
        for row in data.get("records") or []:
            try:
                out.append((str(row[di]), float(row[pi])))
            except (TypeError, ValueError, IndexError):
                continue
        return out

    def get_self_correlations(self, alpha_id: str) -> dict:
        return self.poll(self._url(f"alphas/{alpha_id}/correlations/self"), headers=V2).json() or {}

    def get_prod_correlations(self, alpha_id: str) -> dict:
        return self.poll(self._url(f"alphas/{alpha_id}/correlations/prod"), headers=V2).json() or {}

    def get_checks(self, alpha_id: str) -> dict:
        """Submission-eligibility checks without submitting (GET /alphas/{id}/check)."""
        return self.poll(self._url(f"alphas/{alpha_id}/check"), headers=V2).json() or {}

    def iter_user_alphas(self, *, stage: str | None = "OS", status: str | None = None,
                         page_size: int = 100, max_items: int = 2000) -> Iterator[dict]:
        offset = 0
        while offset < max_items:
            params: dict[str, Any] = {"limit": min(100, page_size), "offset": offset, "order": "-dateCreated",
                                      "hidden": "false"}
            if stage:
                params["stage"] = stage
            if status:
                params["status"] = status
            page = self.get_json("users/self/alphas", params=params) or {}
            results = page.get("results") or []
            yield from results
            offset += len(results)
            if not results or not page.get("next"):
                return

    # ------------------------------------------------------------ metadata
    def list_datasets(self, region: str, universe: str, delay: int, instrument: str = "EQUITY") -> list[dict]:
        params = {"instrumentType": instrument, "region": region, "delay": delay, "universe": universe,
                  "limit": 50, "offset": 0}
        return self._paged("data-sets", params)

    def list_data_fields(self, region: str, universe: str, delay: int, *, dataset_id: str | None = None,
                         search: str | None = None, instrument: str = "EQUITY",
                         max_items: int = 3000) -> list[dict]:
        params: dict[str, Any] = {"instrumentType": instrument, "region": region, "delay": delay,
                                  "universe": universe, "limit": 50, "offset": 0}
        if dataset_id:
            params["dataset.id"] = dataset_id
        if search:
            params["search"] = search
        return self._paged("data-fields", params, max_items=max_items)

    def list_operators(self) -> list[dict]:
        data = self.get_json("operators")
        if isinstance(data, dict):
            return data.get("results") or []
        return data or []

    def _paged(self, path: str, params: dict, max_items: int = 2000) -> list[dict]:
        out: list[dict] = []
        while len(out) < max_items:
            page = self.get_json(path, params=params) or {}
            results = page.get("results") or []
            out.extend(results)
            params = {**params, "offset": params["offset"] + len(results)}
            if not results or len(out) >= int(page.get("count", 0) or 0):
                break
        return out

    @staticmethod
    def alpha_web_url(alpha_id: str) -> str:
        return f"{BRAIN_WEB_BASE}/alpha/{alpha_id}"


def alpha_code(alpha: dict) -> str:
    """Expression text of a BRAIN alpha. In GET /alphas/{id}, `regular` is {code, description, ...}."""
    reg = alpha.get("regular")
    if isinstance(reg, dict):
        return str(reg.get("code") or "")
    return str(reg or "")


def alpha_metrics(alpha: dict) -> dict:
    """Normalise the `is` block of a BRAIN alpha into Alpha Bench metric names."""
    is_ = alpha.get("is") or {}
    out = {k: is_.get(k) for k in ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin")}
    out["long_count"] = is_.get("longCount")
    out["short_count"] = is_.get("shortCount")
    out["pnl_total"] = is_.get("pnl")
    out["book_size"] = float(is_.get("bookSize") or 20_000_000)
    by_check = {"LOW_SUB_UNIVERSE_SHARPE": "sub_universe_sharpe", "CONCENTRATED_WEIGHT": "max_weight",
                "LOW_2Y_SHARPE": "sharpe_2y", "SELF_CORRELATION": "self_correlation",
                "PROD_CORRELATION": "prod_correlation"}
    for c in is_.get("checks") or []:
        key = by_check.get(c.get("name"))
        if key and c.get("value") is not None:
            out[key] = c.get("value")
    return {k: v for k, v in out.items() if v is not None}


def _error_detail(resp: HttpResponse) -> str:
    body = resp.json()
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            if body.get(key):
                return str(body[key])[:300]
        return "; ".join(f"{k}: {v}" for k, v in list(body.items())[:4])[:300]
    return resp.text_snippet(200)
