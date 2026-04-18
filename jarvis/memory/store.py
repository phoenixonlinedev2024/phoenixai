"""Persistent memory store — SQLite-backed long-term memory for JARVIS."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.config import cfg


class MemoryStore:
    """Stores facts, conversation history, learned lessons, and user preferences."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or cfg.MEMORY_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    key         TEXT UNIQUE NOT NULL,
                    value       TEXT NOT NULL,
                    source      TEXT DEFAULT 'user',
                    confidence  REAL DEFAULT 1.0,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lessons (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson      TEXT NOT NULL,
                    context     TEXT,
                    applied     INTEGER DEFAULT 0,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skills (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT UNIQUE NOT NULL,
                    cron        TEXT NOT NULL,
                    prompt      TEXT NOT NULL,
                    enabled     INTEGER DEFAULT 1,
                    last_run    TEXT,
                    next_run    TEXT
                );

                CREATE TABLE IF NOT EXISTS capability_gaps (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    context     TEXT,
                    resolved    INTEGER DEFAULT 0,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ab_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_hash   TEXT NOT NULL,
                    variant_a   TEXT NOT NULL,
                    variant_b   TEXT NOT NULL,
                    winner      TEXT,
                    score_a     REAL,
                    score_b     REAL,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitored_targets (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT UNIQUE NOT NULL,
                    target_type TEXT NOT NULL,
                    target      TEXT NOT NULL,
                    last_hash   TEXT,
                    action      TEXT NOT NULL,
                    enabled     INTEGER DEFAULT 1
                );
            """)

    # ------------------------------------------------------------------ #
    # Conversation history
    # ------------------------------------------------------------------ #

    def save_message(self, session_id: str, role: str, content: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts),
            )

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ------------------------------------------------------------------ #
    # Facts (long-term knowledge)
    # ------------------------------------------------------------------ #

    def store_fact(self, key: str, value: Any, source: str = "inferred", confidence: float = 0.9) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        serialised = json.dumps(value) if not isinstance(value, str) else value
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO facts (key, value, source, confidence, timestamp)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value=excluded.value, source=excluded.source,
                       confidence=excluded.confidence, timestamp=excluded.timestamp""",
                (key, serialised, source, confidence, ts),
            )

    def recall_fact(self, key: str) -> Any | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def search_facts(self, query: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value, confidence FROM facts WHERE key LIKE ? OR value LIKE ? LIMIT 20",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
        return [{"key": r["key"], "value": r["value"], "confidence": r["confidence"]} for r in rows]

    def all_facts(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return [{"key": r["key"], "value": r["value"]} for r in rows]

    # ------------------------------------------------------------------ #
    # Lessons learned
    # ------------------------------------------------------------------ #

    def store_lesson(self, lesson: str, context: str = "") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO lessons (lesson, context, timestamp) VALUES (?, ?, ?)",
                (lesson, context, ts),
            )

    def get_lessons(self, limit: int = 20) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT lesson FROM lessons ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["lesson"] for r in rows]

    # ------------------------------------------------------------------ #
    # Skills tracking
    # ------------------------------------------------------------------ #

    def record_skill_use(self, name: str, description: str, success: bool) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO skills (name, description, usage_count, success_rate, timestamp)
                   VALUES (?, ?, 1, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       usage_count = usage_count + 1,
                       success_rate = (success_rate * usage_count + ?) / (usage_count + 1),
                       timestamp = excluded.timestamp""",
                (name, description, 1.0 if success else 0.0, ts, 1.0 if success else 0.0),
            )

    # ------------------------------------------------------------------ #
    # Scheduled tasks
    # ------------------------------------------------------------------ #

    def add_scheduled_task(self, name: str, cron: str, prompt: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO scheduled_tasks (name, cron, prompt)
                   VALUES (?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET cron=excluded.cron, prompt=excluded.prompt""",
                (name, cron, prompt),
            )

    def get_scheduled_tasks(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled=1"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_task_run(self, name: str, last_run: str, next_run: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE scheduled_tasks SET last_run=?, next_run=? WHERE name=?",
                (last_run, next_run, name),
            )

    # ------------------------------------------------------------------ #
    # Capability gaps
    # ------------------------------------------------------------------ #

    def log_gap(self, description: str, context: str = "") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO capability_gaps (description, context, timestamp) VALUES (?, ?, ?)",
                (description, context, ts),
            )

    def get_open_gaps(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, description, context FROM capability_gaps WHERE resolved=0 ORDER BY id DESC LIMIT 20"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_gap(self, gap_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE capability_gaps SET resolved=1 WHERE id=?", (gap_id,))

    # ------------------------------------------------------------------ #
    # A/B test results
    # ------------------------------------------------------------------ #

    def record_ab_result(
        self, task_hash: str, variant_a: str, variant_b: str,
        winner: str, score_a: float, score_b: float,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ab_results (task_hash, variant_a, variant_b, winner, score_a, score_b, timestamp) VALUES (?,?,?,?,?,?,?)",
                (task_hash, variant_a, variant_b, winner, score_a, score_b, ts),
            )

    def get_ab_winner(self, task_hash: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT winner FROM ab_results WHERE task_hash=? ORDER BY id DESC LIMIT 1",
                (task_hash,),
            ).fetchone()
        return row["winner"] if row else None

    # ------------------------------------------------------------------ #
    # Monitored targets
    # ------------------------------------------------------------------ #

    def add_monitor_target(self, name: str, target_type: str, target: str, action: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO monitored_targets (name, target_type, target, action)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       target_type=excluded.target_type, target=excluded.target, action=excluded.action""",
                (name, target_type, target, action),
            )

    def get_monitor_targets(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM monitored_targets WHERE enabled=1").fetchall()
        return [dict(r) for r in rows]

    def update_monitor_hash(self, name: str, new_hash: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE monitored_targets SET last_hash=? WHERE name=?", (new_hash, name))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        with self._conn() as conn:
            n_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            n_lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            n_skills = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            n_tasks = conn.execute("SELECT COUNT(*) FROM scheduled_tasks WHERE enabled=1").fetchone()[0]
            n_gaps = conn.execute("SELECT COUNT(*) FROM capability_gaps WHERE resolved=0").fetchone()[0]
            n_monitors = conn.execute("SELECT COUNT(*) FROM monitored_targets WHERE enabled=1").fetchone()[0]
        return (
            f"Memory: {n_facts} facts | {n_lessons} lessons | {n_skills} skills | "
            f"{n_tasks} scheduled tasks | {n_gaps} open gaps | {n_monitors} monitors"
        )
