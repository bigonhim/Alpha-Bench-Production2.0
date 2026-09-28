"""Scriptable fake HTTP transport: no network, no simulation quota."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from alpha_bench.brain.transport import HttpResponse

API = "https://api.worldquantbrain.com"
AUTH = API + "/authentication"


def resp(status: int, body: Any = None, headers: dict | None = None, url: str = AUTH) -> HttpResponse:
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    return HttpResponse(status=status, url=url, headers=headers or {}, body=raw)


class FakeTransport:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[HttpResponse | Callable[..., HttpResponse]]] = defaultdict(list)
        self.calls: list[dict[str, Any]] = []
        self.cookies: list[dict[str, Any]] = []

    def on(self, method: str, url: str, *responses) -> "FakeTransport":
        self.routes[(method.upper(), url)].extend(responses)
        return self

    def request(self, method, url, *, json=None, params=None, basic_auth=None, timeout=60.0, headers=None):
        self.calls.append({"method": method.upper(), "url": url, "json": json, "params": params,
                           "basic_auth": basic_auth, "headers": headers})
        queue = self.routes.get((method.upper(), url))
        if not queue:
            return resp(404, {"detail": "not found"}, url=url)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        result = item(json=json, params=params) if callable(item) else item
        if result.status in (200, 201) and url == AUTH and method.upper() == "POST":
            self.cookies = [{"name": "t", "value": "session-cookie-value-1234567890", "domain": "api.worldquantbrain.com",
                             "path": "/", "expires": None, "secure": True}]
        return result

    def count(self, method: str, url: str) -> int:
        return sum(1 for c in self.calls if c["method"] == method.upper() and c["url"] == url)

    def export_cookies(self):
        return list(self.cookies)

    def import_cookies(self, cookies):
        self.cookies = list(cookies)

    def clear_cookies(self):
        self.cookies = []
