"""Alpha library: versions, lineage, BRAIN results, jobs and the reconciliation log.

SQLite in the app folder. Nothing secret is stored here (no credentials, no
cookies); exports are plain JSON of expressions, settings and metrics.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    name TEXT,
    mode TEXT,                 -- create | modify | rebuild | manual | fix | sweep | imported
    expression TEXT NOT NULL,
    settings TEXT NOT NULL,
    family TEXT,
    thesis TEXT,
    rationale TEXT,
    parent_id INTEGER REFERENCES alphas(id),
    fingerprint TEXT,
    offline TEXT,              -- JSON: metrics + check summary
    brain_alpha_id TEXT,
    brain TEXT,                -- JSON: metrics + checks from BRAIN
    status TEXT DEFAULT 'draft',
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_alphas_parent ON alphas(parent_id);
CREATE INDEX IF NOT EXISTS ix_alphas_brain ON alphas(brain_alpha_id);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    alpha_id INTEGER,
    expression TEXT NOT NULL,
    settings TEXT NOT NULL,
    state TEXT NOT NULL,
    progress_url TEXT,
    brain_alpha_id TEXT,
    error TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS reconciliations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_id INTEGER,
    brain_alpha_id TEXT,
    created_at REAL,
    metric TEXT,
    offline REAL,
    brain REAL,
    family TEXT
);
CREATE TABLE IF NOT EXISTS known_alphas (
    id TEXT PRIMARY KEY,       -- BRAIN alpha id, or 'pasted-<n>'
    name TEXT,
    source TEXT,               -- brain | pasted
    expression TEXT,
    settings TEXT,
    sharpe REAL,
    pnl TEXT,                  -- JSON [[date, cumulative pnl], ...] (BRAIN alphas only)
    updated_at REAL
);
"""


class Library:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------- alphas
    def add_alpha(self, *, expression: str, settings: dict, mode: str, name: str | None = None,
                  family: str | None = None, thesis: str | None = None, rationale: Any = None,
                  parent_id: int | None = None, fingerprint: str | None = None, offline: Any = None,
                  notes: str | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO alphas (created_at, name, mode, expression, settings, family, thesis, rationale, "
                "parent_id, fingerprint, offline, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), name, mode, expression, json.dumps(settings), family, thesis,
                 json.dumps(rationale) if rationale is not None else None, parent_id, fingerprint,
                 json.dumps(offline) if offline is not None else None, notes))
            self._db.commit()
            return int(cur.lastrowid)

    def update_alpha(self, alpha_id: int, **fields: Any) -> None:
        allowed = {"name", "brain_alpha_id", "brain", "status", "notes", "offline"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k} = ?")
            vals.append(json.dumps(v) if k in ("brain", "offline") and v is not None else v)
        if not sets:
            return
        with self._lock:
            self._db.execute(f"UPDATE alphas SET {', '.join(sets)} WHERE id = ?", (*vals, alpha_id))
            self._db.commit()

    def _row(self, r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in ("settings", "rationale", "offline", "brain"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        return d

    def get(self, alpha_id: int) -> dict | None:
        with self._lock:
            return self._row(self._db.execute("SELECT * FROM alphas WHERE id = ?", (alpha_id,)).fetchone())

    def by_brain_id(self, brain_alpha_id: str) -> dict | None:
        with self._lock:
            return self._row(self._db.execute("SELECT * FROM alphas WHERE brain_alpha_id = ? ORDER BY id DESC",
                                              (brain_alpha_id,)).fetchone())

    def list(self, *, search: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM alphas"
        args: list[Any] = []
        if search:
            q += " WHERE expression LIKE ? OR name LIKE ? OR thesis LIKE ? OR brain_alpha_id LIKE ?"
            args = [f"%{search}%"] * 4
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [self._row(r) for r in self._db.execute(q, args).fetchall()]

    def fingerprints(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT fingerprint FROM alphas WHERE fingerprint IS NOT NULL")]

    def lineage(self, alpha_id: int) -> dict:
        """Ancestors (oldest first) and all descendants of an alpha."""
        with self._lock:
            chain, cur = [], self.get(alpha_id)
            while cur and cur.get("parent_id"):
                cur = self.get(cur["parent_id"])
                if cur:
                    chain.insert(0, {"id": cur["id"], "mode": cur["mode"], "expression": cur["expression"]})
            kids, frontier = [], [alpha_id]
            while frontier:
                pid = frontier.pop()
                for r in self._db.execute("SELECT id, mode, expression, parent_id FROM alphas WHERE parent_id = ?",
                                          (pid,)).fetchall():
                    kids.append(dict(r))
                    frontier.append(r["id"])
            return {"ancestors": chain, "descendants": kids}

    def delete(self, alpha_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE alphas SET parent_id = NULL WHERE parent_id = ?", (alpha_id,))
            self._db.execute("DELETE FROM alphas WHERE id = ?", (alpha_id,))
            self._db.commit()

    def export(self, alpha_id: int) -> dict | None:
        a = self.get(alpha_id)
        if not a:
            return None
        return {"alpha_bench_export": 1, **{k: a.get(k) for k in (
            "id", "name", "mode", "expression", "settings", "family", "thesis", "rationale", "parent_id",
            "offline", "brain_alpha_id", "brain", "status", "notes", "created_at")},
            "lineage": self.lineage(alpha_id)}

    # --------------------------------------------------------------- jobs
    def save_job(self, job: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, alpha_id, expression, settings, state, progress_url, brain_alpha_id, error, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "state=excluded.state, progress_url=excluded.progress_url, brain_alpha_id=excluded.brain_alpha_id, "
                "error=excluded.error, updated_at=excluded.updated_at",
                (job["id"], job.get("library_id"), job["expression"], json.dumps(job["settings"]), job["state"],
                 job.get("progress_url"), job.get("brain_alpha_id"), job.get("error"), job.get("created_at"),
                 job.get("updated_at")))
            self._db.commit()

    def unfinished_jobs(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE state NOT IN ('done', 'error', 'cancelled')").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["settings"] = json.loads(d["settings"])
            out.append(d)
        return out

    # ------------------------------------------------------ reconciliation
    def log_reconciliation(self, alpha_id: int | None, brain_alpha_id: str | None, rows: list[dict],
                           family: str | None) -> None:
        now = time.time()
        with self._lock:
            for r in rows:
                if r.get("offline") is None or r.get("brain") is None:
                    continue
                self._db.execute("INSERT INTO reconciliations (alpha_id, brain_alpha_id, created_at, metric, offline, "
                                 "brain, family) VALUES (?,?,?,?,?,?,?)",
                                 (alpha_id, brain_alpha_id, now, r["metric"], float(r["offline"]), float(r["brain"]),
                                  family))
            self._db.commit()

    def reconciliation_rows(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM reconciliations ORDER BY id").fetchall()]

    # -------------------------------------------------------- known alphas
    def upsert_known(self, *, id: str, name: str, source: str, expression: str, settings: dict,
                     sharpe: float | None, pnl: list | None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO known_alphas (id, name, source, expression, settings, sharpe, pnl, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, "
                "expression=excluded.expression, settings=excluded.settings, sharpe=excluded.sharpe, "
                "pnl=COALESCE(excluded.pnl, known_alphas.pnl), updated_at=excluded.updated_at",
                (id, name, source, expression, json.dumps(settings), sharpe,
                 json.dumps(pnl) if pnl is not None else None, time.time()))
            self._db.commit()

    def known(self, source: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM known_alphas", []
        if source:
            q += " WHERE source = ?"
            args.append(source)
        with self._lock:
            rows = [dict(r) for r in self._db.execute(q, args).fetchall()]
        for d in rows:
            d["settings"] = json.loads(d["settings"]) if d.get("settings") else {}
            d["pnl"] = json.loads(d["pnl"]) if d.get("pnl") else None
        return rows

    def clear_known(self, source: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM known_alphas WHERE source = ?", (source,))
            self._db.commit()
