"""The `threads` table: which conversations exist and what to call them.

LangGraph's checkpointer owns the `checkpoints` and `writes` tables (it creates them for
us). This module owns one extra table so the sidebar can list and order conversations,
because checkpoints store conversation *state*, not a title.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime

from chatbot import config

TITLE_MAX_LENGTH = 40

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads (updated_at DESC);
"""


@dataclass(frozen=True)
class Thread:
    id: str
    title: str
    created_at: str
    updated_at: str


def init_db() -> None:
    with closing(_connect()) as connection, connection:
        connection.executescript(SCHEMA)


def list_threads() -> list[Thread]:
    """Newest activity first."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
    return [Thread(*row) for row in rows]


def ensure_thread(thread_id: str, first_message: str) -> None:
    """Create the thread on its first message, otherwise just bump `updated_at`."""
    now = _now()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO threads (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (thread_id, title_from(first_message), now, now),
        )


def delete_thread(thread_id: str) -> None:
    with closing(_connect()) as connection, connection:
        connection.execute("DELETE FROM threads WHERE id = ?", (thread_id,))


def title_from(text: str) -> str:
    """A short, single-line title derived from the first user message."""
    flat = " ".join(text.split())
    if len(flat) <= TITLE_MAX_LENGTH:
        return flat or "New chat"
    return flat[:TITLE_MAX_LENGTH].rstrip() + "…"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(config.db_path())


def _now() -> str:
    return datetime.now(UTC).isoformat()
