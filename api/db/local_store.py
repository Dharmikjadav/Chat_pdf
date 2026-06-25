"""SQLite-backed document registry (single source of truth for the API).

Thread-safe: a fresh connection is opened per operation and writes are
serialized with a lock, which is robust on Windows where a single sqlite3
connection must not be shared across threads.
"""

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    status           TEXT NOT NULL,
    total_pages      INTEGER NOT NULL DEFAULT 0,
    processed_pages  INTEGER NOT NULL DEFAULT 0,
    current_page     INTEGER NOT NULL DEFAULT 0,
    progress_percent REAL    NOT NULL DEFAULT 0.0,
    stage            TEXT    NOT NULL DEFAULT 'uploaded',
    errors           TEXT    NOT NULL DEFAULT '[]',
    error            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
)
"""


class LocalStore:
    """A tiny SQLite CRUD layer for document records."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        try:
            data["errors"] = json.loads(data.get("errors") or "[]")
        except (json.JSONDecodeError, TypeError):
            data["errors"] = []
        return data

    def create(self, doc_id: str, filename: str, status: str, created_at: str, updated_at: str) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, filename, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, status, created_at, updated_at),
            )
            conn.commit()

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            return self._to_dict(row) if row else None

    def list_all(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [self._to_dict(r) for r in rows]

    def update(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "errors" in fields and not isinstance(fields["errors"], str):
            fields["errors"] = json.dumps(fields["errors"], ensure_ascii=False)
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [doc_id]
        with self._lock, closing(self._connect()) as conn:
            conn.execute(f"UPDATE documents SET {assignments} WHERE doc_id = ?", values)
            conn.commit()

    def delete(self, doc_id: str) -> bool:
        with self._lock, closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0
