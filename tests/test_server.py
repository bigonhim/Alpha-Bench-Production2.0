"""API-level tests: security guard, sign-in flow through the server, and the main modes.

BRAIN is faked; nothing touches the network or the OS keyring.
"""
import time

import pytest
from fastapi.testclient import TestClient

from alpha_bench.app_state import AppState
from alpha_bench.config import Preferences
from alpha_bench.web.server import create_app

from .fakes import API, AUTH, FakeTransport, resp

PORT = 8765
KEY = "test-key-123"
USER = {"user": {"id": "AB12345"}, "token": {"expiry": 14400.0}, "permissions": []}


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    prefs = Preferences(offline_stocks=150, offline_days=420)
    transport = FakeTransport()
    state = AppState(home, prefs, transport=transport, use_keyring=False)
    state.start()
    assert state.ready.wait(120)
    app = create_app(state, port=PORT, key=KEY)
    client = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")
    r = client.get(f"/?k={KEY}", follow_redirects=False)
    assert r.status_code == 303
    client.cookies.set("ab_key", KEY)
    client.headers["X-Alpha-Bench"] = "1"
    return state, transport, client


def test_guard_blocks_foreign_requests(app_env):
    state, _, client = app_env
    raw = TestClient(client.app, base_url=f"http://127.0.0.1:{PORT}")
    assert raw.get("/api/state").status_code == 403                      # no key cookie
    raw.cookies.set("ab_key", KEY)
    assert raw.get("/api/state").status_code == 403                      # no custom header
    assert raw.get("/api/state", headers={"X-Alpha-Bench": "1"}).status_code == 200
    evil = TestClient(client.app, base_url="http://evil.example:80")
    evil.cookies.set("ab_key", KEY)
    assert evil.get("/api/state", headers={"X-Alpha-Bench": "1"}).status_code == 421  # DNS rebinding
    assert raw.get("/?k=wrong").status_code == 403
    r = raw.get("/api/state", headers={"X-Alpha-Bench": "1"})
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_validation_errors_never_echo_password(app_env):
    _, _, client = app_env
    r = client.post("/api/auth/signin", json={"email": "x@y.com", "password": ["S3cret-in-a-list"]})
    assert r.status_code == 422
    assert "S3cret" not in r.text


def test_signin_verification_and_signout_via_api(app_env):
    state, transport, client = app_env
    transport.routes.clear()
    transport.on("POST", AUTH, resp(401, {"inquiry": "inq_42"}))
    st = client.post("/api/auth/signin", json={"email": "me@x.com", "password": "pw123456", "remember": False}).json()
    assert st["state"] == "verification_needed" and st["verification_url"].endswith("inquiry=inq_42")
    transport.on("POST", AUTH + "/persona", resp(401, {"inquiry": "inq_42"}), resp(201, USER))
    assert client.post("/api/auth/verify-check").json()["state"] == "verification_needed"
    assert client.post("/api/auth/verify-check").json()["state"] == "signed_in"
    assert client.get("/api/state").json()["auth"]["label"] == "Signed in"


def test_create_simulate_why_optimize(app_env):
    _, _, client = app_env
    r = client.post("/api/create", json={"families": ["price_volume"]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["expression"] and res["library_id"] and res["checks"] and res["sim"]["series"]["dates"]
    sim = client.post("/api/simulate", json={"expression": res["expression"], "settings": res["settings"]}).json()
    assert "LOW_SHARPE" in {c["name"] for c in sim["checks"]}
    why = client.post("/api/why", json={"expression": "rank(ts_sum(returns, 5))", "settings": {}}).json()
    assert why["fixes"]
    opt = client.post("/api/optimize", json={"expression": "rank(-returns)", "settings": {},
                                             "grid": {"decay": [0, 4], "truncation": [0.08],
                                                      "neutralization": ["MARKET"]}}).json()
    assert opt["runs"] == 2
    bad = client.post("/api/simulate", json={"expression": "rank(close", "settings": {}})
    assert bad.status_code == 400 and "Expression error" in bad.json()["error"]


def test_recreate_and_library(app_env):
    _, _, client = app_env
    r = client.post("/api/recreate", json={"expression": "rank(ts_sum(returns, 5))", "settings": {}, "mode": "modify"})
    assert r.status_code == 200, r.text
    d = r.json()
    item = client.get(f"/api/library/{d['library_id']}").json()
    assert item["lineage"]["ancestors"][0]["id"] == d["parent_library_id"]
    exp = client.get(f"/api/library/{d['library_id']}/export").json()
    assert exp["alpha_bench_export"] == 1


def test_live_simulation_job_and_reconcile(app_env):
    state, transport, client = app_env
    if not state.auth.is_signed_in:
        transport.on("POST", AUTH, resp(201, USER))
        client.post("/api/auth/signin", json={"email": "me@x.com", "password": "pw123456", "remember": False})
    alpha = {"id": "ZZ99", "regular": {"code": "rank(-returns)"}, "settings": {"region": "USA", "universe": "TOP3000",
             "delay": 1, "decay": 0, "neutralization": "MARKET", "truncation": 0.08},
             "is": {"sharpe": 1.1, "fitness": 0.6, "turnover": 0.82, "returns": 0.08, "bookSize": 20000000,
                    "checks": [{"name": "HIGH_TURNOVER", "result": "FAIL", "limit": 0.7, "value": 0.82},
                               {"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.25, "value": 1.1}]}}
    transport.on("POST", API + "/simulations", resp(201, None, {"Location": "/simulations/s1"}))
    transport.on("GET", API + "/simulations/s1", resp(200, {"status": "COMPLETE", "alpha": "ZZ99"}))
    transport.on("GET", API + "/alphas/ZZ99", resp(200, alpha))
    r = client.post("/api/brain/simulate", json={"expression": "rank(-returns)",
                                                 "settings": {"neutralization": "MARKET"}}).json()
    lid = r["library_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        item = client.get(f"/api/library/{lid}").json()["item"]
        if item.get("brain"):
            break
        time.sleep(0.1)
    assert item["brain_alpha_id"] == "ZZ99" and item["status"] == "brain-fail"
    rec = client.post("/api/reconcile", json={"brain_alpha_id": "ZZ99"}).json()
    assert set(rec["failing"]) == {"HIGH_TURNOVER", "LOW_SHARPE"} and rec["source"] == "brain"
    assert client.get("/api/calibration").json()


def test_course_gallery_and_calibration(app_env):
    _, _, client = app_env
    d = client.get("/api/course").json()
    assert any(e["id"] == "l4-open-close" for e in d["examples"]) and d["principles"]
    cal = client.post("/api/course/calibrate").json()
    assert len(cal["rows"]) >= 6 and cal["calibration"]["sharpe"]["n"] >= 6
    sim = client.post("/api/simulate", json={"expression": "rank(-returns)", "settings": {}}).json()
    assert "sharpe" in sim["calibrated"]  # results now carry a calibrated estimate
    meta = client.get("/api/meta").json()
    assert "SLOW_AND_FAST" in meta["risk_neutralizations"] and meta["defaults"]["truncation"] == 0.01


def test_outlook_and_auto_calibration(app_env):
    state, transport, client = app_env
    sim = client.post("/api/simulate", json={"expression": "rank(-ts_sum(returns, 5))", "settings": {}}).json()
    assert sim["outlook"]["level"] in ("fail", "likely-fail", "borderline", "comfortable")
    # the live simulation earlier in the suite compared offline vs BRAIN automatically
    rows = state.library.reconciliation_rows()
    assert rows  # auto-logged; Calibration feeds outlook
    res = client.post("/api/create", json={"families": ["price_volume"]}).json()
    assert "outlook" in res and isinstance(res["outlook"]["reasons"], list)


def test_signout_wipes(app_env):
    state, transport, client = app_env
    transport.on("DELETE", AUTH, resp(204))
    assert client.post("/api/auth/signout").json()["state"] == "signed_out"
    assert client.post("/api/brain/simulate", json={"expression": "rank(close)"}).status_code == 401
