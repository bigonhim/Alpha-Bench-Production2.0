"""Credential storage: the OS keyring (Windows Credential Locker, macOS Keychain,
Secret Service) or environment variables. Never a plain file.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

from ..config import KEYRING_SERVICE

log = logging.getLogger(__name__)

ENV_EMAIL = "BRAIN_EMAIL"
ENV_PASSWORD = "BRAIN_PASSWORD"
_EMAIL_KEY = "__account_email__"


class SecretBackend(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...


class KeyringBackend:
    """Thin wrapper over the `keyring` package, scoped to one service name."""

    def __init__(self, service: str = KEYRING_SERVICE):
        self.service = service
        try:
            import keyring
            from keyring.backends import fail

            self._keyring = keyring
            self.available = not isinstance(keyring.get_keyring(), fail.Keyring)
        except Exception:  # noqa: BLE001 - keyring can fail to import on odd systems
            self._keyring = None
            self.available = False

    def get(self, key: str) -> str | None:
        if not self.available:
            return None
        try:
            return self._keyring.get_password(self.service, key)
        except Exception:  # noqa: BLE001
            log.warning("Could not read from the OS keyring.")
            return None

    def set(self, key: str, value: str) -> None:
        if not self.available:
            raise RuntimeError("No OS keyring is available on this system.")
        self._keyring.set_password(self.service, key, value)

    def delete(self, key: str) -> None:
        if not self.available:
            return
        try:
            self._keyring.delete_password(self.service, key)
        except Exception:  # noqa: BLE001 - deleting a missing entry raises
            pass


class MemoryBackend:
    """In-process backend for tests or systems without a keyring."""

    def __init__(self) -> None:
        self.available = True
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class CredentialStore:
    def __init__(self, backend: SecretBackend):
        self.backend = backend

    @property
    def can_remember(self) -> bool:
        return bool(getattr(self.backend, "available", False))

    def load(self) -> tuple[str, str] | None:
        """Remembered keyring credentials first, then environment variables."""
        email = self.backend.get(_EMAIL_KEY)
        if email:
            password = self.backend.get(email)
            if password:
                return email, password
        env_email, env_password = os.environ.get(ENV_EMAIL), os.environ.get(ENV_PASSWORD)
        if env_email and env_password:
            return env_email, env_password
        return None

    def remembered_email(self) -> str | None:
        return self.backend.get(_EMAIL_KEY) or os.environ.get(ENV_EMAIL)

    def has_saved_password(self) -> bool:
        return self.load() is not None

    def save(self, email: str, password: str) -> None:
        old = self.backend.get(_EMAIL_KEY)
        if old and old != email:
            self.backend.delete(old)
        self.backend.set(_EMAIL_KEY, email)
        self.backend.set(email, password)

    def delete(self) -> None:
        email = self.backend.get(_EMAIL_KEY)
        if email:
            self.backend.delete(email)
        self.backend.delete(_EMAIL_KEY)
