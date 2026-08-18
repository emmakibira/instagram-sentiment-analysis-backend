"""SQLite-backed storage for analysis history.

Analyses are stored as JSON blobs keyed by a UUID so that past results can be
recalled from the Streamlit sidebar without re-scraping Instagram.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_history.db",
)


class HistoryStore:
    """Persist and retrieve past analyses."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the store, creating the table when needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path or os.getenv("HISTORY_DB_PATH", _DEFAULT_DB)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    post_url TEXT,
                    result TEXT,
                    created_at REAL
                )
                """
            )

    def save(self, result: dict[str, Any]) -> str:
        """Persist an analysis result and return its id."""
        analysis_id = str(uuid.uuid4())
        post_url = result.get("post_info", {}).get("url", "")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (id, post_url, result, created_at) VALUES (?, ?, ?, ?)",
                (analysis_id, post_url, json.dumps(result), time.time()),
            )
        return analysis_id

    def list(self, limit: int = 50) -> builtins.list[dict[str, Any]]:
        """Return the most recent analyses (id, url, created_at, summary)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, post_url, result, created_at FROM analyses "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                result = json.loads(row["result"])
            except (json.JSONDecodeError, TypeError):
                result = {}
            summary = result.get("sentiment_summary", {})
            items.append(
                {
                    "id": row["id"],
                    "post_url": row["post_url"],
                    "created_at": row["created_at"],
                    "satisfaction_rate": summary.get("satisfaction_rate", 0),
                    "total_analyzed": summary.get("total_analyzed", 0),
                }
            )
        return items

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        """Retrieve a single stored analysis."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["result"])
        except (json.JSONDecodeError, TypeError):
            return None

    def delete(self, analysis_id: str) -> None:
        """Delete a stored analysis."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))

    def clear(self) -> None:
        """Delete all stored analyses."""
        with self._connect() as conn:
            conn.execute("DELETE FROM analyses")
