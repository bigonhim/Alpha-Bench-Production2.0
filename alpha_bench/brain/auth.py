"""BRAIN sign-in and session management.

Flow
----
1. POST {api}/authentication with HTTP Basic auth on a cookie-keeping session.
2. A JSON body with `user` means the session cookie is valid.
3. Otherwise BRAIN may demand biometric (Persona) identity verification. Two
   response shapes are handled (see `parse_challenge`). The user completes the
   face check themselves in their own browser; Alpha Bench only shows the link
   and later asks BRAIN whether it's done. It never automates, replays, or
   captures any part of the verification.
4. Cookies are persisted encrypted so a valid session survives restarts.
5. A 401 mid-session flips the state to SESSION_EXPIRED; listeners (the job
   queue) pause and resume after the user signs in again.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urljoin, urlparse

from ..config import BRAIN_API_BASE
from .credentials import CredentialStore
from .errors import (BrainRequestError, InvalidCredentials, NetworkError, RateLimited, ServerError,
                     SignInLocked, VerificationExpired)
from .redact import forget_secrets, register_secret
from .session_store import SessionStore
from .transport import HttpResponse, Transport

log = logging.getLogger(__name__)

# Hosts the verification link may point at. Anything else is refused rather
# than opened in the user's browser.
_TRUSTED_VERIFICATION_HOSTS = ("worldquantbrain.com", "withpersona.com")


class AuthState(str, Enum):
    SIGNED_OUT = "signed_out"
    SIGNED_IN = "signed_in"
    VERIFICATION_NEEDED = "verification_needed"
    VERIFICATION_EXPIRED = "verification_expired"
    SESSION_EXPIRED = "session_expired"
    LOCKED = "locked"
    OFFLINE = "offline"


STATE_LABELS = {
    AuthState.SIGNED_OUT: "Signed out",
    AuthState.SIGNED_IN: "Signed in",
    AuthState.VERIFICATION_NEEDED: "Verification needed",
    AuthState.VERIFICATION_EXPIRED: "Verification expired",
    AuthState.SESSION_EXPIRED: "Session expired",
    AuthState.LOCKED: "Sign-in paused",
    AuthState.OFFLINE: "Offline",
}


@dataclass
class VerificationChallenge:
    browser_url: str          # what the user opens (or copies to a phone)
    followup_url: str         # where we POST once they say they're done
    followup_json: Any        # body for that POST (None for the Location flavour)
    kind: str                 # "location" | "inquiry"
    created_at: float = field(default_factory=time.time)


def _trusted_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == h or host.endswith("." + h)
                                            for h in _TRUSTED_VERIFICATION_HOSTS)


def parse_challenge(resp: HttpResponse, auth_url: str) -> VerificationChallenge | None:
    """Recognise a biometric-verification demand in an /authentication response.

    Shape A (used by WorldQuant's own ace_lib helper): HTTP 401 with
      `WWW-Authenticate: persona` and a `Location` header. The verification page
      is urljoin(response_url, Location); completing it is confirmed by POSTing
      to that same URL, which returns 201 once the check has passed.
    Shape B (seen in other open-source clients): a JSON body with an `inquiry`
      id and no `user`. The page is {auth_url}/persona?inquiry=<id>, confirmed by
      POSTing the original payload back to {auth_url}/persona.

    The exact format should be confirmed against the live response; both are
    parsed defensively and any URL outside BRAIN/Persona is rejected.
    """
    www = (resp.header("WWW-Authenticate") or "").lower()
    location = resp.header("Location")
    if "persona" in www and location:
        url = urljoin(resp.url or auth_url, location)
        if _trusted_url(url):
            return VerificationChallenge(browser_url=url, followup_url=url, followup_json=None,
                                         kind="location")
    body = resp.json()
    if isinstance(body, dict) and body.get("inquiry") and not body.get("user"):
        inquiry = quote(str(body["inquiry"]), safe="")
        base = auth_url.rstrip("/")
        return VerificationChallenge(browser_url=f"{base}/persona?inquiry={inquiry}",
                                     followup_url=f"{base}/persona", followup_json=body,
                                     kind="inquiry")
    return None


def _retry_after(resp: HttpResponse) -> float | None:
    raw = resp.header("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


@dataclass
class _Pending:
    email: str
    password: str
    remember: bool


class AuthManager:
    def __init__(self, transport: Transport, credentials: CredentialStore, sessions: SessionStore,
                 *, api_base: str = BRAIN_API_BASE, state_file: Path | None = None,
                 max_failed_attempts: int = 3, lockout_seconds: float = 15 * 60,
                 verification_ttl: float = 20 * 60, clock: Callable[[], float] = time.time):
        self.transport = transport
        self.credentials = credentials
        self.sessions = sessions
        self.auth_url = api_base.rstrip("/") + "/authentication"
        self.state_file = state_file
        self.max_failed_attempts = max_failed_attempts
        self.lockout_seconds = lockout_seconds
        self.verification_ttl = verification_ttl
        self.clock = clock

        self._lock = threading.RLock()
        self._signed_in = threading.Condition(self._lock)
        self._listeners: list[Callable[[dict], None]] = []

        self.state = AuthState.SIGNED_OUT
        self.message = "Sign in with your BRAIN account to run live simulations."
        self.user: dict[str, Any] | None = None
        self.email: str | None = None
        self.token_expiry: float | None = None
        self.challenge: VerificationChallenge | None = None
        self._pending: _Pending | None = None
        self.remembered = False
        self.failed_attempts = 0
        self.locked_until = 0.0
        self._load_guard_state()

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_lock_state()
            left = max(0, self.max_failed_attempts - self.failed_attempts)
            return {
                "state": self.state.value,
                "label": STATE_LABELS[self.state],
                "message": self.message,
                "user_id": (self.user or {}).get("id"),
                "email": self.email,
                "remembered_email": self.credentials.remembered_email(),
                "has_saved_password": self.credentials.has_saved_password(),
                "can_remember": self.credentials.can_remember,
                "verification_url": self.challenge.browser_url if self.challenge else None,
                "verification_age_seconds": (self.clock() - self.challenge.created_at) if self.challenge else None,
                "attempts_left": left,
                "locked_seconds": max(0.0, self.locked_until - self.clock()),
                "token_expiry_seconds": self.token_expiry,
            }

    @property
    def is_signed_in(self) -> bool:
        return self.state == AuthState.SIGNED_IN

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def _set_state(self, state: AuthState, message: str) -> None:
        self.state = state
        self.message = message
        if state == AuthState.SIGNED_IN:
            self._signed_in.notify_all()
        snapshot = self.status()
        for fn in list(self._listeners):
            try:
                fn(snapshot)
            except Exception:  # noqa: BLE001 - a bad listener must not break auth
                log.exception("auth listener failed")

    def wait_until_signed_in(self, timeout: float | None = None) -> bool:
        with self._signed_in:
            return self._signed_in.wait_for(lambda: self.state == AuthState.SIGNED_IN, timeout=timeout)

    # ---------------------------------------------------- lockout bookkeeping
    def _load_guard_state(self) -> None:
        if not self.state_file or not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.failed_attempts = int(raw.get("failed_attempts", 0))
            self.locked_until = float(raw.get("locked_until", 0.0))
        except (OSError, ValueError):
            pass
        self._refresh_lock_state()

    def _save_guard_state(self) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.write_text(json.dumps({"failed_attempts": self.failed_attempts,
                                                   "locked_until": self.locked_until}), encoding="utf-8")
        except OSError:
            pass

    def _refresh_lock_state(self) -> None:
        if self.locked_until and self.clock() >= self.locked_until:
            self.locked_until = 0.0
            self.failed_attempts = 0
            self._save_guard_state()
            if self.state == AuthState.LOCKED:
                self.state = AuthState.SIGNED_OUT
                self.message = "You can try signing in again."
        elif self.locked_until and self.state not in (AuthState.SIGNED_IN, AuthState.LOCKED):
            self.state = AuthState.LOCKED
            self.message = SignInLocked(self.locked_until - self.clock()).user_message

    # ------------------------------------------------------------- sign in
    def restore(self) -> dict[str, Any]:
        """On startup: reuse the saved encrypted session if BRAIN still accepts it."""
        with self._lock:
            cookies = self.sessions.load()
            if not cookies:
                self._set_state(AuthState.SIGNED_OUT, "Sign in with your BRAIN account to run live simulations.")
                return self.status()
            for c in cookies:
                register_secret(c.get("value"))
            self.transport.import_cookies(cookies)
            try:
                who = self._whoami()
            except NetworkError:
                self._set_state(AuthState.OFFLINE, "Couldn't reach BRAIN to check your saved session. "
                                                   "Offline tools still work; retry when you're back online.")
                return self.status()
            except (RateLimited, ServerError) as exc:
                self._set_state(AuthState.OFFLINE, exc.user_message)
                return self.status()
            if who:
                self.remembered = True
                self.email = self.credentials.remembered_email()
                self._accept_user(who)
                self._set_state(AuthState.SIGNED_IN, f"Signed in as {self._display_name()} (saved session).")
            else:
                self.transport.clear_cookies()
                self.sessions.wipe()
                self._set_state(AuthState.SESSION_EXPIRED, "Your saved session has expired. Sign in again.")
            return self.status()

    def sign_in(self, email: str, password: str, remember: bool) -> dict[str, Any]:
        with self._lock:
            self._refresh_lock_state()
            if self.state == AuthState.LOCKED:
                raise SignInLocked(self.locked_until - self.clock())
            register_secret(password)
            self.challenge = None
            self._pending = None
            resp = self.transport.request("POST", self.auth_url, basic_auth=(email, password))
            return self._handle_login(resp, _Pending(email, password, remember))

    def sign_in_saved(self) -> dict[str, Any]:
        creds = self.credentials.load()
        if not creds:
            raise InvalidCredentials(self.max_failed_attempts - self.failed_attempts)
        return self.sign_in(creds[0], creds[1], remember=True)

    def _handle_login(self, resp: HttpResponse, pending: _Pending) -> dict[str, Any]:
        body = resp.json()
        if resp.status in (200, 201) and isinstance(body, dict) and body.get("user"):
            return self._complete(body, pending)

        challenge = parse_challenge(resp, self.auth_url)
        if challenge:
            # The password was accepted; only the face check is outstanding.
            self.failed_attempts = 0
            self._save_guard_state()
            self.challenge = challenge
            self._pending = pending
            self.email = pending.email
            self._set_state(AuthState.VERIFICATION_NEEDED,
                            "BRAIN needs you to confirm your identity with a quick face check. "
                            "Open the link, finish the check, then come back.")
            return self.status()

        if resp.status == 429:
            raise RateLimited(_retry_after(resp))
        if resp.status >= 500:
            raise ServerError()
        if resp.status in (401, 403):
            self.failed_attempts += 1
            if self.failed_attempts >= self.max_failed_attempts:
                self.locked_until = self.clock() + self.lockout_seconds
            self._save_guard_state()
            self._refresh_lock_state()
            if self.state != AuthState.LOCKED:
                self._set_state(AuthState.SIGNED_OUT, "Wrong email or password.")
            raise InvalidCredentials(max(0, self.max_failed_attempts - self.failed_attempts))
        raise BrainRequestError(resp.status, resp.text_snippet(160))

    def check_verification(self) -> dict[str, Any]:
        """Ask BRAIN whether the user has finished the face check.

        Safe to call repeatedly (the UI polls it gently); returns the current status
        with a friendly "not verified yet" message until BRAIN confirms.
        """
        with self._lock:
            if self.state != AuthState.VERIFICATION_NEEDED or not self.challenge or not self._pending:
                return self.status()
            if self.clock() - self.challenge.created_at > self.verification_ttl:
                self._expire_challenge()
                raise VerificationExpired()

            resp = self.transport.request("POST", self.challenge.followup_url,
                                          json=self.challenge.followup_json)
            body = resp.json()
            if resp.status in (200, 201):
                if isinstance(body, dict) and body.get("user"):
                    return self._complete(body, self._pending)
                who = self._whoami()
                if who:
                    return self._complete(who, self._pending)
            elif resp.status in (404, 410):
                self._expire_challenge()
                raise VerificationExpired()
            elif resp.status == 429:
                raise RateLimited(_retry_after(resp))
            elif resp.status >= 500:
                raise ServerError()
            else:
                fresh = parse_challenge(resp, self.auth_url)
                if fresh and fresh.browser_url != self.challenge.browser_url:
                    self.challenge = fresh
                    self._set_state(AuthState.VERIFICATION_NEEDED,
                                    "BRAIN issued a new verification link. Please open it and finish the check.")
                    return self.status()

            self.message = ("Not verified yet. Finish the face check in your browser; "
                            "Alpha Bench will notice automatically, or click Retry.")
            return self.status()

    def restart_sign_in(self) -> dict[str, Any]:
        """Fresh login attempt (e.g. after an expired verification link)."""
        with self._lock:
            pending = self._pending
            self.challenge = None
            if pending:
                return self.sign_in(pending.email, pending.password, pending.remember)
            if self.credentials.has_saved_password():
                return self.sign_in_saved()
            self._set_state(AuthState.SIGNED_OUT, "Enter your BRAIN email and password to sign in again.")
            return self.status()

    def _expire_challenge(self) -> None:
        self.challenge = None
        self._set_state(AuthState.VERIFICATION_EXPIRED, VerificationExpired.user_message)

    def _complete(self, who: dict[str, Any], pending: _Pending) -> dict[str, Any]:
        self._accept_user(who)
        self.email = pending.email
        self.failed_attempts = 0
        self.locked_until = 0.0
        self._save_guard_state()
        self.challenge = None
        self.remembered = pending.remember and self.credentials.can_remember
        if self.remembered:
            try:
                self.credentials.save(pending.email, pending.password)
            except Exception:  # noqa: BLE001
                log.warning("Could not save credentials to the OS keyring.")
            self.sessions.save(self.transport.export_cookies())
        else:
            # "Don't remember me": nothing touches disk, and any old copy goes.
            self.credentials.delete()
            self.sessions.wipe()
        for c in self.transport.export_cookies():
            register_secret(c.get("value"))
        self._pending = None
        self._set_state(AuthState.SIGNED_IN, f"Signed in as {self._display_name()}.")
        return self.status()

    def _accept_user(self, who: dict[str, Any]) -> None:
        self.user = who.get("user") if isinstance(who.get("user"), dict) else {"id": str(who.get("user"))}
        token = who.get("token") or {}
        self.token_expiry = token.get("expiry") if isinstance(token, dict) else None

    def _display_name(self) -> str:
        uid = (self.user or {}).get("id")
        if self.email and uid:
            return f"{self.email} ({uid})"
        return self.email or uid or "your BRAIN account"

    # --------------------------------------------------------- session checks
    def _whoami(self) -> dict[str, Any] | None:
        resp = self.transport.request("GET", self.auth_url)
        if resp.status == 429:
            raise RateLimited(_retry_after(resp))
        if resp.status >= 500:
            raise ServerError()
        body = resp.json()
        if resp.status == 200 and isinstance(body, dict) and body.get("user"):
            return body
        return None

    def refresh(self) -> dict[str, Any]:
        """Lightweight authenticated call to confirm the session is still alive."""
        with self._lock:
            if self.state != AuthState.SIGNED_IN:
                return self.status()
            who = self._whoami()
            if who:
                self._accept_user(who)
            else:
                self.mark_expired()
            return self.status()

    def mark_expired(self) -> None:
        """Called by the API client when BRAIN answers 401 mid-session."""
        with self._lock:
            if self.state == AuthState.SIGNED_IN:
                self.transport.clear_cookies()
                self.sessions.wipe()
                self._set_state(AuthState.SESSION_EXPIRED,
                                "Your BRAIN session expired. Running jobs are paused; sign in to resume them.")

    def sign_out(self) -> dict[str, Any]:
        """Sign out and wipe the local session and any stored credentials."""
        with self._lock:
            if self.state == AuthState.SIGNED_IN:
                try:
                    self.transport.request("DELETE", self.auth_url, timeout=10)
                except Exception:  # noqa: BLE001 - best effort; local wipe still happens
                    pass
            self.transport.clear_cookies()
            self.sessions.wipe()
            self.credentials.delete()
            forget_secrets()
            self.user = None
            self.email = None
            self.token_expiry = None
            self.challenge = None
            self._pending = None
            self.remembered = False
            self._set_state(AuthState.SIGNED_OUT, "Signed out. Local session and saved credentials were wiped.")
            return self.status()
