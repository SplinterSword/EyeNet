# src/core/database.py
"""SQLite persistence layer with WAL mode for concurrent reads."""

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from src.config import Config

logger = logging.getLogger(__name__)

_local = threading.local()


@contextmanager
def get_db():
    """Thread-local SQLite connection with WAL journaling."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(Config.DB_PATH) or ".", exist_ok=True)
        _local.conn = sqlite3.connect(Config.DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    yield _local.conn


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                event_type    TEXT    NOT NULL,
                severity      INTEGER NOT NULL DEFAULT 1,
                description   TEXT,
                track_id      INTEGER,
                image_path    TEXT,
                metadata_json TEXT,
                acknowledged  INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_type      ON alerts(event_type);
            CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON alerts(severity);

            CREATE TABLE IF NOT EXISTS students (
                roll            TEXT PRIMARY KEY,
                name            TEXT,
                email           TEXT,
                encoding_path   TEXT,
                registered_at   TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                fps             REAL,
                detection_count INTEGER,
                face_count      INTEGER,
                processing_ms   REAL
            );
        """)
        conn.commit()
    logger.info("Database initialized at %s", Config.DB_PATH)


# ── Alert helpers ───────────────────────────────────────────────────────

def insert_alert(
    event_type: str,
    severity: int,
    description: str,
    track_id: int | None = None,
    image_path: str = "",
    metadata: dict | None = None,
) -> int:
    """Insert an alert and return its row id."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO alerts (event_type, severity, description, track_id, image_path, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_type, severity, description, track_id, image_path,
             json.dumps(metadata or {})),
        )
        conn.commit()
        return cur.lastrowid


def query_alerts(
    event_type: str | None = None,
    min_severity: int | None = None,
    since: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Query alerts with optional filters."""
    sql = "SELECT * FROM alerts WHERE 1=1"
    params: list[Any] = []
    if event_type:
        sql += " AND event_type = ?"
        params.append(event_type)
    if min_severity:
        sql += " AND severity >= ?"
        params.append(min_severity)
    if since:
        sql += " AND timestamp >= ?"
        params.append(since)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(min(limit, 500))

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    with get_db() as conn:
        conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        conn.commit()


# ── Metrics helpers ─────────────────────────────────────────────────────

def insert_metrics(fps: float, detection_count: int, face_count: int, processing_ms: float):
    """Record a metrics snapshot."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO metrics (fps, detection_count, face_count, processing_ms) VALUES (?,?,?,?)",
            (fps, detection_count, face_count, processing_ms),
        )
        conn.commit()


def prune_old_metrics(hours: int = 24):
    """Delete metrics older than *hours*."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM metrics WHERE timestamp < datetime('now', ? || ' hours', 'localtime')",
            (f"-{hours}",),
        )
        conn.commit()


def get_latest_metrics() -> dict | None:
    """Return the most recent metrics row as a dict."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_hourly_alert_stats() -> list[dict]:
    """Alerts per hour for the last 24 hours."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM alerts
            WHERE timestamp >= datetime('now', '-24 hours', 'localtime')
            GROUP BY hour ORDER BY hour
        """).fetchall()
    return [dict(r) for r in rows]
