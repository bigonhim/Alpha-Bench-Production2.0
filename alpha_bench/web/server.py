"""Local web UI + JSON API for Alpha Bench.

Security model (the server holds a live BRAIN session):
* Binds to 127.0.0.1 only.
* A random per-launch key is delivered once via the launch URL (?k=...), then
  kept in an HttpOnly, SameSite=Strict cookie. Every /api call needs that cookie
  *and* an `X-Alpha-Bench: 1` header, so other websites can't drive the API
  (no CORS is enabled, and custom headers force a preflight that fails).
* The Host header must be the loopback address (blocks DNS-rebinding attacks).
* Validation errors never echo request bodies (so a password can't bounce back).
"""
from __future__ import annotations

import base64
import hmac
import logging
import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..app_state import AppState, jsonable
from ..brain.client import BrainClient, alpha_code, alpha_metrics
from ..brain.errors import BrainError
from ..brain.redact import redact
from ..engine.evaluator import APPROXIMATIONS, EvalError
from ..engine.simulate import FIDELITY_NOTES
from ..expr.classify import classify
from ..expr.operators import catalog_rows
from ..expr.parser import ParseError, parse
from ..gen.fields import build_pool
from ..gen.ideas import FAMILIES, THESES
from ..gen.optimize import sweep
from ..gen.reconcile import Calibration, ocr_image, parse_pasted_metrics, reconcile
from ..gen.recreate import recreate
from ..gen.repair import assess, propose, rank_fixes
from ..rules.engine import brain_checks_view, brain_outlook, summarize
from .. import course
from ..settings_model import NEUTRALIZATIONS, REGIONS, RISK_NEUTRALIZATIONS, UNIVERSES, SimSettings

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
COOKIE = "ab_key"


# ---------------------------------------------------------------- models
class SignIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=512)
    remember: bool = True


class ExprIn(BaseModel):
    expression: str = Field(max_length=20000)
    settings: dict = Field(default_factory=dict)


class CreateIn(BaseModel):
    families: list[str] | None = None
    settings: dict = Field(default_factory=dict)
    allowed_ops: list[str] | None = None
    auto_universe: bool = False


class RecreateIn(ExprIn):
    mode: str = "modify"


class OptimizeIn(ExprIn):
    grid: dict = Field(default_factory=dict)


class SaveIn(ExprIn):
    mode: str = "manual"
    parent_id: int | None = None
    name: str | None = None
    family: str | None = None
    thesis: str | None = None
    rationale: Any = None
    offline: Any = None


class LiveIn(BaseModel):
    library_id: int | None = None
    expression: str | None = None
    settings: dict = Field(default_factory=dict)


class ReconcileIn(BaseModel):
    library_id: int | None = None
    brain_alpha_id: str | None = None
    expression: str | None = None
    settings: dict = Field(default_factory=dict)
    pasted_text: str | None = None


class OcrIn(BaseModel):
    image_b64: str = Field(max_length=20_000_000)


class PasteIn(BaseModel):
    items: list[dict]


class PatchIn(BaseModel):
    name: str | None = None
    notes: str | None = None
    status: str | None = None


class RiskIn(BaseModel):
    max_turnover: float | None = None
    max_weight: float | None = None
    max_sector_net: float | None = None


class PrefsIn(BaseModel):
    max_concurrent_simulations: int | None = None
    default_region: str | None = None
    default_universe: str | None = None
    default_delay: int | None = None


def create_app(state: AppState, *, port: int, key: str | None = None) -> FastAPI:
    key = key or secrets.token_urlsafe(24)
    app = FastAPI(title="Alpha Bench", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.launch_key = key
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    def outlook_for(metrics: dict, check_dicts: list[dict], settings: SimSettings, family: str | None):
        cal = Calibration(state.library.reconciliation_rows()).adjust(metrics, family)
        return brain_outlook(metrics, check_dicts, state.rules, settings, calibrated=cal), cal

    def ok(data: Any = None, **kw) -> JSONResponse:
        return JSONResponse(jsonable(data if data is not None else kw))

    def fail(message: str, status: int = 400, **extra) -> JSONResponse:
        return JSONResponse({"error": redact(message), **extra}, status_code=status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"error": "Bad host."}, status_code=421)
        path = request.url.path
        if path.startswith("/api/"):
            cookie = request.cookies.get(COOKIE, "")
            if not hmac.compare_digest(cookie, key) or request.headers.get("x-alpha-bench") != "1":
                return JSONResponse({"error": "Not authorised. Open Alpha Bench from the launcher link."},
                                    status_code=403)
        resp = await call_next(request)
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation(_: Request, exc: RequestValidationError):
        fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()})
        return JSONResponse({"error": "Some inputs were missing or invalid: " + ", ".join(fields)}, status_code=422)

    @app.exception_handler(BrainError)
    async def brain_error(_: Request, exc: BrainError):
        return JSONResponse({"error": exc.user_message, "kind": type(exc).__name__}, status_code=400)

    @app.exception_handler(ParseError)
    async def parse_error(_: Request, exc: ParseError):
        return JSONResponse({"error": f"Expression error: {exc}", "pos": exc.pos}, status_code=400)

    @app.exception_handler(EvalError)
    async def eval_error(_: Request, exc: EvalError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    # ------------------------------------------------------------ pages
    @app.get("/")
    def index(request: Request, k: str | None = None):
        if k is not None:
            if not hmac.compare_digest(k, key):
                return HTMLResponse(_locked_page(), status_code=403)
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, key, httponly=True, samesite="strict", path="/")
            return resp
        if not hmac.compare_digest(request.cookies.get(COOKIE, ""), key):
            return HTMLResponse(_locked_page(), status_code=403)
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ------------------------------------------------------------ state
    @app.get("/api/state")
    def get_state():
        jobs = state.queue.jobs()
        return ok({"auth": state.auth.status(), "engine_ready": state.ready.is_set(), "load_error": state.load_error,
                   "jobs": jobs[:30], "tasks": state.tasks.active(), "rate_limit": state.client.rate_limit,
                   "pool": {"active": state.pnl_pool.count("active"), "pasted": state.pnl_pool.count("pasted"),
                            "session": state.pnl_pool.count("session"), "brain_pnls": len(state.brain_pnls)},
                   "platform_fields": len(state.catalog.platform),
                   "prefs": vars(state.prefs)})

    # ------------------------------------------------------------- auth
    @app.post("/api/auth/signin")
    def signin(body: SignIn):
        return ok(state.auth.sign_in(body.email.strip(), body.password, body.remember))

    @app.post("/api/auth/signin-saved")
    def signin_saved():
        return ok(state.auth.sign_in_saved())

    @app.post("/api/auth/verify-check")
    def verify_check():
        return ok(state.auth.check_verification())

    @app.post("/api/auth/restart")
    def restart():
        return ok(state.auth.restart_sign_in())

    @app.post("/api/auth/refresh")
    def refresh():
        return ok(state.auth.refresh())

    @app.post("/api/auth/signout")
    def signout():
        return ok(state.auth.sign_out())

    # ------------------------------------------------------------- meta
    @app.get("/api/meta")
    def meta(region: str = "USA"):
        pool = build_pool(state.catalog)
        return ok({"operators": catalog_rows(), "families": FAMILIES,
                   "theses": [{"id": t.id, "family": t.family, "title": t.title, "mechanism": t.mechanism}
                              for t in THESES],
                   "fields": {"platform": pool.platform, "roles": pool.roles,
                              "platform_count": len(state.catalog.platform)},
                   "universes": UNIVERSES, "neutralizations": NEUTRALIZATIONS, "regions": REGIONS,
                   "risk_neutralizations": RISK_NEUTRALIZATIONS,
                   "course": {"source": course.SOURCE, "principles": course.PRINCIPLES},
                   "fidelity": FIDELITY_NOTES, "approximations": APPROXIMATIONS,
                   "rules": state.rules.effective(region), "rule_conflicts": state.rules.conflicts(region),
                   "risk": state.rules.risk(),
                   "rules_path": str(state.rules.user_path), "defaults": SimSettings().to_dict()})

    @app.post("/api/meta/sync")
    def meta_sync():
        return ok(state.tasks.start("sync-fields", state.sync_platform))

    @app.get("/api/tasks/{tid}")
    def task(tid: str):
        t = state.tasks.get(tid)
        return ok(t) if t else fail("Unknown task.", 404)

    # ------------------------------------------------------------ modes
    @app.post("/api/create")
    def create(body: CreateIn):
        state.require_engine()
        with state.compute_lock:
            gen = state.generator
            gen.pnl_pool = state.pnl_pool
            res = gen.create(set(body.families) if body.families else None,
                             SimSettings.from_dict({**SimSettings().to_dict(), **body.settings}),
                             allowed_ops=set(body.allowed_ops) if body.allowed_ops else None,
                             auto_universe=body.auto_universe)
        if "error" in res:
            return fail(res["error"])
        res["outlook"], res["calibrated"] = outlook_for(res["sim"]["metrics"], res["checks"],
                                                        SimSettings.from_dict(res["settings"]), res["family"])
        res["library_id"] = state.save_result(expression=res["expression"], settings=res["settings"], mode="create",
                                              family=res["family"], thesis=res["thesis"], rationale=res["rationale"],
                                              offline={"metrics": res["sim"]["metrics"], "summary": res["summary"]})
        return ok(res)

    @app.post("/api/recreate")
    def do_recreate(body: RecreateIn):
        engine = state.require_engine()
        settings = SimSettings.from_dict(body.settings)
        with state.compute_lock:
            res = recreate(engine, state.rules, state.catalog, state.generator, body.expression, settings,
                           "rebuild" if body.mode == "rebuild" else "modify", state._corr_fn())
        if "error" in res:
            return fail(res["error"])
        parent = state.save_result(expression=res["original"]["expression"], settings=res["original"]["settings"],
                                   mode="imported", offline={"metrics": res["original"]["sim"]["metrics"],
                                                             "summary": res["original"]["summary"]})
        res["library_id"] = state.save_result(
            expression=res["result"]["expression"], settings=res["result"]["settings"], mode=res["mode"],
            parent_id=parent, family=res["structure"]["family"], thesis=res["structure"]["thesis"],
            rationale={"description": res["description"], "changes": res.get("changes"),
                       "inspiration": res.get("inspiration")},
            offline={"metrics": res["result"]["sim"]["metrics"], "summary": res["result"]["summary"]})
        res["result"]["outlook"], res["result"]["calibrated"] = outlook_for(
            res["result"]["sim"]["metrics"], res["result"]["checks"],
            SimSettings.from_dict(res["result"]["settings"]), res["structure"]["family"])
        res["parent_library_id"] = parent
        return ok(res)

    @app.post("/api/simulate")
    def simulate(body: ExprIn):
        engine = state.require_engine()
        node = parse(body.expression)
        settings = SimSettings.from_dict(body.settings)
        with state.compute_lock:
            a = assess(engine, state.rules, node, settings, state._corr_fn(), full=True)
        s = classify(node, state.catalog.category)
        check_dicts = [c.to_dict() for c in a["checks"]]
        outlook, calibrated = outlook_for(a["sim"].metrics, check_dicts, settings, s.family)
        return ok({"sim": a["sim"].to_dict(), "checks": check_dicts,
                   "summary": summarize(a["checks"]), "correlation": a["correlation"],
                   "structure": s.to_dict(), "description": s.describe(),
                   "outlook": outlook, "calibrated": calibrated})

    @app.post("/api/why")
    def why(body: ExprIn):
        engine = state.require_engine()
        node = parse(body.expression)
        settings = SimSettings.from_dict(body.settings)
        with state.compute_lock:
            base = assess(engine, state.rules, node, settings, state._corr_fn())
            fixes = propose(node, settings, base["checks"], base["sim"].metrics, state.catalog, base["correlation"])
            ranked = rank_fixes(engine, state.rules, base, fixes, state._corr_fn(), limit=8)
        failing = [c.to_dict() for c in base["checks"] if c.result in ("FAIL", "WARN")]
        return ok({"failing": failing, "summary": summarize(base["checks"]),
                   "fixes": [{"title": f.title, "detail": f.detail, "change": f.change, "size": f.size,
                              "targets": f.targets, **f.result} for f in ranked]})

    @app.post("/api/optimize")
    def optimize(body: OptimizeIn):
        engine = state.require_engine()
        with state.compute_lock:
            res = sweep(engine, state.rules, parse(body.expression), SimSettings.from_dict(body.settings),
                        body.grid or None, state._corr_fn())
        return fail(res["error"]) if "error" in res else ok(res)

    # ---------------------------------------------------------- library
    @app.post("/api/library")
    def save(body: SaveIn):
        parse(body.expression)
        lid = state.save_result(expression=body.expression, settings=SimSettings.from_dict(body.settings).to_dict(),
                                mode=body.mode, parent_id=body.parent_id, name=body.name, family=body.family,
                                thesis=body.thesis, rationale=body.rationale, offline=body.offline)
        return ok({"library_id": lid})

    @app.get("/api/library")
    def library(search: str | None = None):
        return ok({"items": state.library.list(search=search)})

    @app.get("/api/library/{lid}")
    def library_item(lid: int):
        item = state.library.get(lid)
        if not item:
            return fail("Not found.", 404)
        return ok({"item": item, "lineage": state.library.lineage(lid),
                   "brain_url": BrainClient.alpha_web_url(item["brain_alpha_id"]) if item.get("brain_alpha_id") else None})

    @app.patch("/api/library/{lid}")
    def library_patch(lid: int, body: PatchIn):
        state.library.update_alpha(lid, **{k: v for k, v in body.model_dump().items() if v is not None})
        return ok({"ok": True})

    @app.delete("/api/library/{lid}")
    def library_delete(lid: int):
        state.library.delete(lid)
        return ok({"ok": True})

    @app.get("/api/library/{lid}/export")
    def library_export(lid: int):
        data = state.library.export(lid)
        return ok(data) if data else fail("Not found.", 404)

    # ------------------------------------------------------------ BRAIN
    @app.post("/api/brain/simulate")
    def live(body: LiveIn):
        if not state.auth.is_signed_in:
            return fail("Sign in to BRAIN first. Live simulations run on your account.", 401)
        if body.library_id:
            item = state.library.get(body.library_id)
            if not item:
                return fail("Library item not found.", 404)
            expr, settings, lid = item["expression"], SimSettings.from_dict(item["settings"]), item["id"]
        else:
            parse(body.expression or "")
            expr, settings = body.expression, SimSettings.from_dict(body.settings)
            offline = None
            if state.ready.is_set():
                # Snapshot the offline prediction so the live result can recalibrate the engine.
                try:
                    with state.compute_lock:
                        quick = state.require_engine().simulate(expr, settings, quick=True)
                    offline = {"metrics": jsonable(quick.metrics)}
                except EvalError:
                    offline = None
            lid = state.save_result(expression=expr, settings=settings.to_dict(), mode="manual", offline=offline)
        state.library.update_alpha(lid, status="brain-queued")
        job = state.submit_live(expr, settings, lid)
        return ok({"job": job.to_dict(), "library_id": lid})

    @app.get("/api/jobs")
    def jobs():
        return ok({"jobs": state.queue.jobs()})

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str):
        return ok({"cancelled": state.queue.cancel(jid)})

    @app.post("/api/brain/check/{alpha_id}")
    def brain_check(alpha_id: str):
        def run(progress):
            progress(0.1, "Asking BRAIN to run submission checks (this does NOT submit)...")
            checks = state.client.get_checks(alpha_id)
            items = ((checks.get("is") or {}).get("checks")) or []
            progress(0.7, "Checking correlation with your active alphas...")
            corr = state.verified_correlation(alpha_id)
            item = state.library.by_brain_id(alpha_id)
            if item:
                brain = item.get("brain") or {}
                brain["checks"] = [{"name": c.get("name"), "result": c.get("result"), "limit": c.get("limit"),
                                    "value": c.get("value"), "source": "verified on BRAIN"} for c in items]
                brain["verified_correlation"] = corr
                state.library.update_alpha(item["id"], brain=brain)
            return {"checks": items, "verified_correlation": corr}
        return ok(state.tasks.start("brain-check", run))

    @app.get("/api/brain/alpha/{alpha_id}")
    def brain_alpha(alpha_id: str):
        a = state.client.get_alpha(alpha_id)
        return ok({"id": alpha_id, "expression": alpha_code(a), "settings": SimSettings.from_brain(a.get("settings") or {}).to_dict(),
                   "metrics": alpha_metrics(a), "checks": brain_checks_view(a), "url": BrainClient.alpha_web_url(alpha_id)})

    # -------------------------------------------------------- reconcile
    @app.post("/api/reconcile")
    def do_reconcile(body: ReconcileIn):
        engine = state.require_engine()
        item = state.library.get(body.library_id) if body.library_id else None
        if body.brain_alpha_id and not item:
            item = state.library.by_brain_id(body.brain_alpha_id)
        expr = body.expression or (item or {}).get("expression")
        settings = SimSettings.from_dict(body.settings or (item or {}).get("settings") or {})
        brain_m: dict = {}
        brain_checks: list[dict] | None = None
        source = "pasted"
        if body.brain_alpha_id and state.auth.is_signed_in:
            a = state.client.get_alpha(body.brain_alpha_id)
            expr = expr or alpha_code(a)
            if not body.settings and not item:
                settings = SimSettings.from_brain(a.get("settings") or {})
            brain_m, brain_checks, source = alpha_metrics(a), brain_checks_view(a), "brain"
        if body.pasted_text:
            parsed = parse_pasted_metrics(body.pasted_text)
            brain_m = {**parsed["metrics"], **brain_m}
            if not brain_checks and parsed["checks"]:
                brain_checks = parsed["checks"]
        if not expr:
            return fail("Give the alpha's expression, pick it from the library, or enter its BRAIN alpha ID.")
        if not brain_m and not brain_checks:
            return fail("No BRAIN results found. Paste the metrics from BRAIN, upload a screenshot, or sign in and "
                        "enter the alpha ID.")
        with state.compute_lock:
            corr = None
            if body.brain_alpha_id and state.auth.is_signed_in and state.brain_pnls:
                try:
                    corr = state.verified_correlation(body.brain_alpha_id)
                except BrainError:
                    corr = None
            res = reconcile(engine, state.rules, state.catalog, expression=expr, settings=settings,
                            brain_metrics=brain_m, brain_checks=brain_checks, corr_fn=state._corr_fn(),
                            brain_correlation=corr)
        family = res["structure"]["family"]
        lib_id = (item or {}).get("id") or state.save_result(expression=expr, settings=settings.to_dict(),
                                                             mode="imported")
        state.library.log_reconciliation(lib_id, body.brain_alpha_id, res["rows"], family)
        if body.brain_alpha_id:
            state.library.update_alpha(lib_id, brain_alpha_id=body.brain_alpha_id,
                                       brain={"metrics": brain_m, "checks": brain_checks or []})
        res.update({"library_id": lib_id, "source": source, "expression": expr, "settings": settings.to_dict(),
                    "verified_correlation": corr,
                    "calibration": Calibration(state.library.reconciliation_rows()).summary()})
        return ok(res)

    @app.post("/api/reconcile/ocr")
    def do_ocr(body: OcrIn):
        try:
            data = base64.b64decode(body.image_b64.split(",")[-1], validate=False)
        except ValueError:
            return fail("That image couldn't be read.")
        text, err = ocr_image(data)
        if err:
            return fail(err)
        return ok({"text": text, **parse_pasted_metrics(text or "")})

    @app.get("/api/course")
    def course_examples():
        return ok({"source": course.SOURCE, "examples": course.examples(), "principles": course.PRINCIPLES})

    @app.post("/api/course/calibrate")
    def course_calibrate():
        """Reconcile every lesson example against the BRAIN numbers shown in the videos and log the gaps."""
        engine = state.require_engine()
        rows = []
        with state.compute_lock:
            for ex in course.EXAMPLES:
                if not ex.brain.get("sharpe"):
                    continue
                settings = SimSettings.from_dict(ex.settings)
                res = reconcile(engine, state.rules, state.catalog, expression=ex.expression, settings=settings,
                                brain_metrics=dict(ex.brain), brain_checks=None)
                lib_id = state.save_result(expression=ex.expression, settings=settings.to_dict(), mode="imported",
                                           name=f"{ex.lesson}: {ex.title}",
                                           rationale={"description": ex.idea, "inspiration": ex.improvement})
                state.library.update_alpha(lib_id, brain={"metrics": ex.brain, "checks": [], "source": course.SOURCE})
                state.library.log_reconciliation(lib_id, None, res["rows"], res["structure"]["family"])
                by = {r["metric"]: r for r in res["rows"]}
                rows.append({"id": ex.id, "title": f"{ex.lesson}: {ex.title}",
                             **{f"{k}_{src}": by.get(k, {}).get(src) for k in ("sharpe", "turnover", "fitness")
                                for src in ("offline", "brain")}})
        return ok({"rows": rows, "calibration": Calibration(state.library.reconciliation_rows()).summary()})

    @app.get("/api/calibration")
    def calibration():
        return ok(Calibration(state.library.reconciliation_rows()).summary())

    # ------------------------------------------------------ correlation
    @app.post("/api/correlation/load-active")
    def load_active():
        if not state.auth.is_signed_in:
            return fail("Sign in to BRAIN to load your active alphas.", 401)
        return ok(state.tasks.start("load-active", state.load_active_alphas))

    @app.post("/api/correlation/paste")
    def paste(body: PasteIn):
        state.require_engine()
        with state.compute_lock:
            return ok(state.add_pasted(body.items))

    @app.delete("/api/correlation/pasted")
    def clear_pasted():
        state.library.clear_known("pasted")
        state.pnl_pool.remove_kind("pasted")
        return ok({"ok": True})

    @app.get("/api/correlation/pool")
    def pool():
        known = state.library.known()
        return ok({"items": [{"id": k["id"], "name": k["name"], "source": k["source"], "expression": k["expression"],
                              "sharpe": k.get("sharpe"), "has_pnl": bool(k.get("pnl"))} for k in known],
                   "counts": {"active": state.pnl_pool.count("active"), "pasted": state.pnl_pool.count("pasted"),
                              "session": state.pnl_pool.count("session")}})

    @app.post("/api/correlation/check")
    def corr_check(body: ExprIn):
        engine = state.require_engine()
        with state.compute_lock:
            sim = engine.simulate(body.expression, SimSettings.from_dict(body.settings), quick=True)
        return ok(state.pnl_pool.max_corr(sim.daily_pnl))

    # ------------------------------------------------------------ prefs
    @app.get("/api/rules")
    def rules(region: str = "USA"):
        return ok({"effective": state.rules.effective(region), "conflicts": state.rules.conflicts(region),
                   "path": str(state.rules.user_path)})

    @app.post("/api/rules/risk")
    def rules_risk(body: RiskIn):
        return ok(state.rules.set_risk(body.model_dump()))

    @app.post("/api/rules/reload")
    def rules_reload():
        state.rules.reload()
        return ok({"ok": True})

    @app.post("/api/prefs")
    def prefs(body: PrefsIn):
        p = state.prefs
        if body.max_concurrent_simulations is not None:
            p.max_concurrent_simulations = max(1, min(10, body.max_concurrent_simulations))
            state.queue.set_limit(p.max_concurrent_simulations)
        if body.default_region:
            p.default_region = body.default_region
        if body.default_universe:
            p.default_universe = body.default_universe
        if body.default_delay is not None:
            p.default_delay = int(body.default_delay)
        p.save(state.home / "preferences.json")
        return ok(vars(p))

    return app


def _locked_page() -> str:
    return ("<!doctype html><meta charset=utf-8><title>Alpha Bench</title>"
            "<body style='font-family:system-ui;background:#0f1115;color:#e6e6e6;padding:48px'>"
            "<h2>Alpha Bench is locked</h2><p>Open it from the link printed by the launcher "
            "(it contains a one-time key). This protects your BRAIN session from other websites.</p></body>")
