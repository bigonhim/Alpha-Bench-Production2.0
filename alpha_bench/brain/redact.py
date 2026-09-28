"""Keep secrets out of logs, error text and exports."""
from __future__ import annotations

import logging
import re
import threading

_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*)(basic|bearer)\s+[A-Za-z0-9+/=._-]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^\n]+"),
    re.compile(r"(?i)(set-cookie\s*[:=]\s*)[^\n]+"),
    re.compile(r"(?i)(\"?password\"?\s*[:=]\s*)\"?[^\s\",}]+"),
    re.compile(r"(?i)(\"?t\"?\s*[:=]\s*)\"?[A-Za-z0-9._-]{16,}"),
]

_known_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str | None) -> None:
    """Remember a live secret (password, cookie value) so it is masked verbatim anywhere."""
    if value and len(value) >= 4:
        with _lock:
            _known_secrets.add(value)


def forget_secrets() -> None:
    with _lock:
        _known_secrets.clear()


def redact(text: str) -> str:
    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + "***", text)
    with _lock:
        secrets = sorted(_known_secrets, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def mask_email(email: str | None) -> str:
    if not email or "@" not in email:
        return "***"
    name, domain = email.split("@", 1)
    return (name[:2] + "***@" + domain) if len(name) > 2 else "***@" + domain


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging crash the app
            return True
        record.msg = redact(message)
        record.args = ()
        return True


def install_log_redaction() -> None:
    root = logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for f in root.filters):
        root.addFilter(RedactingFilter())
    for handler in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
