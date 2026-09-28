"""Errors raised by the BRAIN layer.

Every error carries `user_message`: plain language that is safe to show in the
UI. None of them ever include credentials, cookies or raw response bodies.
"""
from __future__ import annotations


class BrainError(Exception):
    user_message = "Something went wrong talking to BRAIN."

    def __init__(self, user_message: str | None = None):
        if user_message:
            self.user_message = user_message
        super().__init__(self.user_message)


class NotSignedIn(BrainError):
    user_message = "You're not signed in to BRAIN. Sign in to run live simulations."


class InvalidCredentials(BrainError):
    user_message = "BRAIN rejected that email or password."

    def __init__(self, attempts_left: int):
        self.attempts_left = attempts_left
        if attempts_left > 0:
            msg = (f"BRAIN rejected that email or password. {attempts_left} attempt"
                   f"{'s' if attempts_left != 1 else ''} left before Alpha Bench pauses sign-in "
                   "to protect your account from lockout.")
        else:
            msg = ("BRAIN rejected that email or password. Sign-in is paused for a while to protect "
                   "your account from lockout. Check your password on the BRAIN website first.")
        super().__init__(msg)


class SignInLocked(BrainError):
    def __init__(self, seconds_left: float):
        self.seconds_left = seconds_left
        minutes = max(1, round(seconds_left / 60))
        super().__init__(
            f"Sign-in is paused for about {minutes} more minute{'s' if minutes != 1 else ''} after "
            "repeated wrong passwords, to avoid locking your BRAIN account."
        )


class VerificationExpired(BrainError):
    user_message = ("The verification link has expired or the check didn't pass. "
                    "Restart sign-in to get a fresh link.")


class SessionExpired(BrainError):
    user_message = ("Your BRAIN session has expired. Running jobs are paused and will resume "
                    "once you sign in again.")


class RateLimited(BrainError):
    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        wait = f" Try again in about {int(retry_after)} seconds." if retry_after else ""
        super().__init__("BRAIN is rate-limiting requests right now." + wait)


class DailyLimitReached(BrainError):
    user_message = ("BRAIN's daily simulation limit for your account has been reached. Live simulations "
                    "resume after the daily reset; offline tools keep working.")


class ServerError(BrainError):
    user_message = "BRAIN's servers returned an error. Alpha Bench backed off and retried, without success."


class NetworkError(BrainError):
    user_message = ("Couldn't reach BRAIN. Check your internet connection. "
                    "Your work is saved locally, so you can retry.")


class BrainRequestError(BrainError):
    """A 4xx that isn't auth or rate limiting (e.g. a malformed expression)."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"BRAIN refused the request ({status}): {detail}")
