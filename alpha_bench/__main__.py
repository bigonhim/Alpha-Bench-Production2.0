"""Launch Alpha Bench: `python -m alpha_bench` (or the `alpha-bench` script)."""
from __future__ import annotations

import argparse
import logging
import secrets
import socket
import sys
import threading
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__
from .brain.redact import RedactingFilter, install_log_redaction
from .config import Preferences, app_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _logging(home: Path, verbose: bool) -> None:
    handler = RotatingFileHandler(home / "alpha-bench.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler, console])
    # urllib3/requests debug output can include headers; keep it quiet.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    install_log_redaction()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="alpha-bench", description="Alpha Bench for WorldQuant BRAIN")
    ap.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free port)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    ap.add_argument("--home", help="data folder (default: per-user app data folder)")
    ap.add_argument("--no-keyring", action="store_true",
                    help="never store credentials or sessions (sign in every launch)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.home:
        import os
        os.environ["ALPHA_BENCH_HOME"] = args.home
    home = app_dir()
    _logging(home, args.verbose)

    import uvicorn

    from .app_state import AppState
    from .web.server import create_app

    prefs = Preferences.load(home / "preferences.json")
    state = AppState(home, prefs, use_keyring=not args.no_keyring)
    state.start()
    port = args.port or _free_port()
    key = secrets.token_urlsafe(24)
    app = create_app(state, port=port, key=key)
    url = f"http://127.0.0.1:{port}/?k={key}"
    print(f"\n  Alpha Bench {__version__}\n  Open: {url}\n  Data folder: {home}\n  Press Ctrl+C to quit.\n",
          flush=True)
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
