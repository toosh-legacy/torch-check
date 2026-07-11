"""Persistence layer: SQLite for run metadata, parquet for per-sample data.

Metadata lives in a small, queryable SQLite table. The potentially large
per-sample DataFrames are written to parquet files keyed by run id, so the
SQL table stays lean.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .result import EvalResult

DEFAULT_STORE_DIR = ".torchcheck"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    tag               TEXT,
    notes             TEXT,
    git_commit        TEXT,
    dataset_id        TEXT,
    timestamp         TEXT,
    n_samples         INTEGER,
    device            TEXT,
    inference_seconds REAL,
    throughput        REAL,
    summary_json      TEXT,
    per_sample_path   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_tag ON runs(tag);
"""


def _git_commit(cwd: Optional[str] = None) -> Optional[str]:
    """Return the current git commit hash, or None if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _dataset_id(dataloader: Any) -> Optional[str]:
    """Best-effort stable identifier for the dataset behind a dataloader."""
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return f"{type(dataset).__name__}:n={len(dataset)}"
    except TypeError:
        return type(dataset).__name__


class RunStore:
    """SQLite + parquet store for evaluation runs."""

    def __init__(self, path: str | Path = DEFAULT_STORE_DIR):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.per_sample_dir = self.dir / "per_sample"
        self.per_sample_dir.mkdir(exist_ok=True)
        self.db_path = self.dir / "runs.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ save
    def save(
        self,
        result: EvalResult,
        dataset_id: Optional[str] = None,
        git_commit: Optional[str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        meta = result.meta

        per_sample_path = ""
        if result.per_sample is not None:
            per_sample_path = str(self.per_sample_dir / f"{run_id}.parquet")
            result.per_sample.to_parquet(per_sample_path)

        row = {
            "run_id": run_id,
            "tag": meta.get("tag"),
            "notes": meta.get("notes"),
            "git_commit": git_commit if git_commit is not None else _git_commit(),
            "dataset_id": dataset_id if dataset_id is not None else meta.get("dataset_id"),
            "timestamp": meta.get("timestamp"),
            "n_samples": meta.get("n_samples"),
            "device": meta.get("device"),
            "inference_seconds": meta.get("inference_seconds"),
            "throughput": meta.get("throughput_samples_per_sec"),
            "summary_json": json.dumps({k: float(v) for k, v in result.summary.items()}),
            "per_sample_path": per_sample_path,
        }
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO runs ({','.join(row)}) "
                f"VALUES ({','.join(':' + k for k in row)})",
                row,
            )
        # stamp back onto the result for callers
        result.meta["run_id"] = run_id
        result.meta["git_commit"] = row["git_commit"]
        result.meta["dataset_id"] = row["dataset_id"]
        return run_id

    # --------------------------------------------------------------- queries
    def history(self, tag: Optional[str] = None) -> pd.DataFrame:
        """Return past runs (newest first) with summary metrics expanded
        into their own columns for trend analysis."""
        query = "SELECT * FROM runs"
        params: tuple = ()
        if tag is not None:
            query += " WHERE tag = ?"
            params = (tag,)
        query += " ORDER BY timestamp DESC"

        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            summary = json.loads(r.pop("summary_json") or "{}")
            r.pop("per_sample_path", None)
            records.append({**r, **summary})
        return pd.DataFrame(records)

    def get_run(self, run_id: str) -> Optional[dict]:
        """Return the full metadata row for a run id (or None)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        return d

    def resolve(self, ref: str) -> Optional[dict]:
        """Resolve a run by exact run_id, else by newest run with that tag."""
        run = self.get_run(ref)
        if run is not None:
            return run
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE tag = ? ORDER BY timestamp DESC LIMIT 1",
                (ref,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        return d

    def load_per_sample(self, run_id: str) -> Optional[pd.DataFrame]:
        run = self.get_run(run_id)
        if not run or not run.get("per_sample_path"):
            return None
        path = Path(run["per_sample_path"])
        if not path.exists():
            return None
        return pd.read_parquet(path)
