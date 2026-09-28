"""Encrypted-at-rest session cookie persistence.

Cookies are encrypted with a Fernet key that itself lives in the OS keyring, so
the file on disk is useless without the user's OS login. If no keyring is
available the session is simply not persisted (the user signs in each launch).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .credentials import SecretBackend

log = logging.getLogger(__name__)

_KEY_NAME = "__session_cookie_key__"


class SessionStore:
    def __init__(self, path: Path, backend: SecretBackend):
        self.path = path
        self.backend = backend

    @property
    def available(self) -> bool:
        return bool(getattr(self.backend, "available", False))

    def _fernet(self, create: bool):
        from cryptography.fernet import Fernet

        key = self.backend.get(_KEY_NAME)
        if not key:
            if not create:
                return None
            key = Fernet.generate_key().decode("ascii")
            self.backend.set(_KEY_NAME, key)
        return Fernet(key.encode("ascii"))

    def save(self, cookies: list[dict[str, Any]]) -> bool:
        if not self.available or not cookies:
            return False
        try:
            token = self._fernet(create=True).encrypt(json.dumps(cookies).encode("utf-8"))
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(token)
            tmp.replace(self.path)
            return True
        except Exception:  # noqa: BLE001
            log.warning("Could not save the encrypted session; you'll need to sign in next launch.")
            return False

    def load(self) -> list[dict[str, Any]] | None:
        if not self.available or not self.path.exists():
            return None
        try:
            fernet = self._fernet(create=False)
            if fernet is None:
                return None
            return json.loads(fernet.decrypt(self.path.read_bytes()).decode("utf-8"))
        except Exception:  # noqa: BLE001 - corrupted file or rotated key: start fresh
            log.info("Saved session could not be decrypted; discarding it.")
            self.wipe()
            return None

    def wipe(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self.backend.delete(_KEY_NAME)
