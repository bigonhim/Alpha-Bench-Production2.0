"""HTTP transport interface.

AuthManager and BrainClient only talk to `Transport`, so tests swap in a fake
that returns canned responses (success, inquiry required, 401, 429) without
touching live BRAIN or spending simulation quota.
"""
from __future__ import annotations

import json as jsonlib
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from .errors import NetworkError


@dataclass
class HttpResponse:
    status: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def __post_init__(self) -> None:
        self.headers = {k.lower(): v for k, v in self.headers.items()}

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())

    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return jsonlib.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def text_snippet(self, limit: int = 300) -> str:
        """Short, printable excerpt of the body for error messages (never cookies)."""
        try:
            text = self.body.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
        return text[:limit]


class Transport(Protocol):
    def request(self, method: str, url: str, *, json: Any = None, params: dict | None = None,
                basic_auth: tuple[str, str] | None = None, timeout: float = 60.0,
                headers: dict | None = None) -> HttpResponse: ...

    def export_cookies(self) -> list[dict[str, Any]]: ...

    def import_cookies(self, cookies: list[dict[str, Any]]) -> None: ...

    def clear_cookies(self) -> None: ...


def require_https(url: str) -> None:
    if urlparse(url).scheme != "https":
        raise ValueError("Alpha Bench only talks to BRAIN over HTTPS.")


class RequestsTransport:
    """Production transport: a persistent requests.Session that keeps cookies.

    Certificate verification is always on and cannot be disabled.
    """

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self._session = requests.Session()
        self._session.verify = True
        self._session.headers["User-Agent"] = "AlphaBench/0.1"

    def request(self, method: str, url: str, *, json: Any = None, params: dict | None = None,
                basic_auth: tuple[str, str] | None = None, timeout: float = 60.0,
                headers: dict | None = None) -> HttpResponse:
        require_https(url)
        try:
            resp = self._session.request(method, url, json=json, params=params, auth=basic_auth,
                                         headers=headers, timeout=timeout, verify=True, allow_redirects=True)
        except (self._requests.ConnectionError, self._requests.Timeout) as exc:
            raise NetworkError() from exc
        return HttpResponse(status=resp.status_code, url=resp.url, headers=dict(resp.headers),
                            body=resp.content)

    def export_cookies(self) -> list[dict[str, Any]]:
        out = []
        for c in self._session.cookies:
            out.append({"name": c.name, "value": c.value, "domain": c.domain, "path": c.path,
                        "expires": c.expires, "secure": c.secure})
        return out

    def import_cookies(self, cookies: list[dict[str, Any]]) -> None:
        for c in cookies:
            self._session.cookies.set(c["name"], c["value"], domain=c.get("domain"),
                                      path=c.get("path") or "/", expires=c.get("expires"),
                                      secure=c.get("secure", True))

    def clear_cookies(self) -> None:
        self._session.cookies.clear()
