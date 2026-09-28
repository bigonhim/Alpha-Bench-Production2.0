import logging
import time

import pytest

from alpha_bench.brain.auth import AuthManager, AuthState, parse_challenge
from alpha_bench.brain.client import BrainClient
from alpha_bench.brain.credentials import CredentialStore, MemoryBackend
from alpha_bench.brain.errors import (InvalidCredentials, RateLimited, SessionExpired, SignInLocked,
                                      VerificationExpired)
from alpha_bench.brain.jobs import SimulationQueue
from alpha_bench.brain.redact import RedactingFilter, redact
from alpha_bench.brain.session_store import SessionStore
from alpha_bench.settings_model import SimSettings

from .fakes import API, AUTH, FakeTransport, resp

USER = {"user": {"id": "AB12345"}, "token": {"expiry": 14400.0}, "permissions": ["TOP_CONSULTANT"]}


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    backend = MemoryBackend()
    transport = FakeTransport()
    clock = Clock()
    auth = AuthManager(transport, CredentialStore(backend), SessionStore(tmp_path / "s.bin", backend),
                       state_file=tmp_path / "guard.json", clock=clock)
    return transport, auth, backend, clock, tmp_path


def test_success_path_signs_in_and_remembers(env):
    transport, auth, backend, _, tmp = env
    transport.on("POST", AUTH, resp(201, USER))
    status = auth.sign_in("me@example.com", "hunter22", remember=True)
    assert status["state"] == "signed_in"
    assert "AB12345" in status["message"]
    assert transport.calls[0]["basic_auth"] == ("me@example.com", "hunter22")
    assert CredentialStore(backend).load() == ("me@example.com", "hunter22")
    assert (tmp / "s.bin").exists()
    assert b"session-cookie-value" not in (tmp / "s.bin").read_bytes()  # encrypted at rest


def test_dont_remember_me_stores_nothing(env):
    transport, auth, backend, _, tmp = env
    transport.on("POST", AUTH, resp(201, USER))
    auth.sign_in("me@example.com", "hunter22", remember=False)
    assert auth.is_signed_in
    assert CredentialStore(backend).load() is None
    assert not (tmp / "s.bin").exists()


def test_inquiry_verification_flow(env):
    transport, auth, _, _, _ = env
    inquiry = {"inquiry": "inq_ABC123"}
    transport.on("POST", AUTH, resp(401, inquiry))
    status = auth.sign_in("me@example.com", "pw123456", remember=True)
    assert status["state"] == "verification_needed"
    assert status["verification_url"] == AUTH + "/persona?inquiry=inq_ABC123"

    # Not done yet: friendly message, no error, state unchanged.
    transport.on("POST", AUTH + "/persona", resp(401, inquiry), resp(201, USER))
    status = auth.check_verification()
    assert status["state"] == "verification_needed"
    assert "Not verified yet" in status["message"]

    status = auth.check_verification()
    assert status["state"] == "signed_in"
    followups = [c for c in transport.calls if c["url"] == AUTH + "/persona"]
    assert followups[-1]["json"] == inquiry  # original payload posted back


def test_location_header_verification_flow(env):
    transport, auth, _, _, _ = env
    headers = {"WWW-Authenticate": "persona", "Location": "/authentication/persona?inquiry=inq_9"}
    transport.on("POST", AUTH, resp(401, None, headers))
    status = auth.sign_in("me@example.com", "pw123456", remember=False)
    assert status["verification_url"] == API + "/authentication/persona?inquiry=inq_9"
    transport.on("POST", API + "/authentication/persona?inquiry=inq_9", resp(201, None))
    transport.on("GET", AUTH, resp(200, USER))
    assert auth.check_verification()["state"] == "signed_in"


def test_untrusted_verification_host_is_rejected():
    r = resp(401, None, {"WWW-Authenticate": "persona", "Location": "https://evil.example.com/x"})
    assert parse_challenge(r, AUTH) is None


def test_verification_link_expiry(env):
    transport, auth, _, clock, _ = env
    transport.on("POST", AUTH, resp(401, {"inquiry": "inq_1"}))
    auth.sign_in("me@example.com", "pw123456", remember=False)
    clock.now += 60 * 60
    with pytest.raises(VerificationExpired):
        auth.check_verification()
    assert auth.state == AuthState.VERIFICATION_EXPIRED


def test_wrong_password_caps_attempts_and_locks(env):
    transport, auth, _, clock, tmp = env
    transport.on("POST", AUTH, resp(401, {"detail": "Incorrect credentials"}))
    for left in (2, 1, 0):
        with pytest.raises(InvalidCredentials) as err:
            auth.sign_in("me@example.com", "wrong", remember=False)
        assert err.value.attempts_left == left
    with pytest.raises(SignInLocked):
        auth.sign_in("me@example.com", "wrong", remember=False)
    assert transport.count("POST", AUTH) == 3  # no 4th request reached BRAIN
    # The lockout survives an app restart.
    backend = MemoryBackend()
    again = AuthManager(transport, CredentialStore(backend), SessionStore(tmp / "x.bin", backend),
                        state_file=tmp / "guard.json", clock=clock)
    assert again.status()["state"] == "locked"
    clock.now += 16 * 60
    assert again.status()["state"] == "signed_out"


def test_rate_limit_on_sign_in_does_not_count_as_failure(env):
    transport, auth, _, _, _ = env
    transport.on("POST", AUTH, resp(429, None, {"Retry-After": "30"}))
    with pytest.raises(RateLimited) as err:
        auth.sign_in("me@example.com", "pw123456", remember=False)
    assert err.value.retry_after == 30
    assert auth.failed_attempts == 0


def test_restore_valid_and_expired_session(env):
    transport, auth, _, _, _ = env
    transport.on("POST", AUTH, resp(201, USER))
    auth.sign_in("me@example.com", "pw123456", remember=True)
    transport.cookies = []

    transport.routes.clear()
    transport.on("GET", AUTH, resp(200, USER))
    assert auth.restore()["state"] == "signed_in"
    assert transport.count("POST", AUTH) == 1  # no new login needed

    transport.routes.clear()
    transport.on("GET", AUTH, resp(401, {"detail": "expired"}))
    assert auth.restore()["state"] == "session_expired"
    # The dead session was wiped, so the next launch simply starts signed out.
    assert auth.restore()["state"] == "signed_out"


def test_sign_out_wipes_everything(env):
    transport, auth, backend, _, tmp = env
    transport.on("POST", AUTH, resp(201, USER))
    transport.on("DELETE", AUTH, resp(204))
    auth.sign_in("me@example.com", "pw123456", remember=True)
    auth.sign_out()
    assert auth.state == AuthState.SIGNED_OUT
    assert CredentialStore(backend).load() is None
    assert not (tmp / "s.bin").exists()
    assert transport.cookies == []


def _signed_in_client(env, **kw):
    transport, auth, _, _, _ = env
    transport.on("POST", AUTH, resp(201, USER))
    auth.sign_in("me@example.com", "pw123456", remember=False)
    sleeps: list[float] = []
    client = BrainClient(transport, auth, sleep=sleeps.append, **kw)
    return transport, auth, client, sleeps


def test_client_401_marks_session_expired(env):
    transport, auth, client, _ = _signed_in_client(env)
    transport.on("GET", API + "/alphas/abc", resp(401, {"detail": "expired"}))
    with pytest.raises(SessionExpired):
        client.get_alpha("abc")
    assert auth.state == AuthState.SESSION_EXPIRED


def test_client_respects_retry_after_on_429(env):
    transport, _, client, sleeps = _signed_in_client(env)
    transport.on("GET", API + "/alphas/abc", resp(429, None, {"Retry-After": "7"}), resp(200, {"id": "abc"}))
    assert client.get_alpha("abc")["id"] == "abc"
    assert sleeps == [7.0]


def test_client_exponential_backoff_on_5xx(env):
    transport, _, client, sleeps = _signed_in_client(env, backoff_base=1.0)
    transport.on("GET", API + "/alphas/abc", resp(503), resp(502), resp(200, {"id": "abc"}))
    client.get_alpha("abc")
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_simulation_polls_with_retry_after(env):
    transport, _, client, sleeps = _signed_in_client(env)
    progress = API + "/simulations/sim1"
    transport.on("POST", API + "/simulations", resp(201, None, {"Location": "/simulations/sim1"}))
    transport.on("GET", progress, resp(200, {"progress": 0.3}, {"Retry-After": "5"}),
                 resp(200, {"status": "COMPLETE", "alpha": "A1"}))
    url = client.start_simulation("rank(close)", SimSettings())
    assert url == progress
    assert client.wait_simulation(url)["alpha"] == "A1"
    assert sleeps == [5.0]
    payload = transport.calls[1]["json"]
    assert payload["type"] == "REGULAR" and payload["regular"] == "rank(close)"
    assert payload["settings"]["neutralization"] == "INDUSTRY"


def test_job_pauses_on_401_and_resumes_after_sign_in(env):
    transport, auth, client, _ = _signed_in_client(env)
    progress = API + "/simulations/sim1"
    transport.on("POST", API + "/simulations", resp(201, None, {"Location": "/simulations/sim1"}))
    transport.on("GET", progress, resp(401, {"detail": "expired"}), resp(200, {"status": "COMPLETE", "alpha": "A1"}))
    transport.on("GET", API + "/alphas/A1", resp(200, {"id": "A1", "is": {"sharpe": 1.5}}))
    states: list[str] = []
    queue = SimulationQueue(client, auth, limit=3, on_update=lambda j: states.append(j.state))
    job = queue.submit("rank(close)", SimSettings())

    deadline = time.time() + 5
    while job.state != "paused" and time.time() < deadline:
        time.sleep(0.02)
    assert job.state == "paused"
    assert job.progress_url == progress  # state kept for resume

    transport.on("POST", AUTH, resp(201, USER))
    auth.sign_in("me@example.com", "pw123456", remember=False)
    deadline = time.time() + 8
    while job.state != "done" and time.time() < deadline:
        time.sleep(0.02)
    assert job.state == "done"
    assert job.result["id"] == "A1"
    assert transport.count("POST", API + "/simulations") == 1  # resumed, not re-submitted


def test_client_has_no_submit_method():
    assert not any("submit" in name for name in dir(BrainClient))


def test_secrets_never_reach_logs(env, caplog):
    transport, auth, _, _, _ = env
    transport.on("POST", AUTH, resp(401, {"detail": "bad"}))
    logger = logging.getLogger("alpha_bench.test")
    logger.addFilter(RedactingFilter())
    with pytest.raises(InvalidCredentials) as err:
        auth.sign_in("me@example.com", "S3cretPassw0rd!", remember=False)
    assert "S3cretPassw0rd!" not in err.value.user_message
    with caplog.at_level(logging.INFO):
        logger.info("debug dump password=S3cretPassw0rd! cookie: t=abc")
    assert "S3cretPassw0rd!" not in caplog.text
    assert redact("Authorization: Basic dXNlcjpwYXNz") == "Authorization: ***"
