"""
database/db_manager.py
----------------------
Production-grade SQLite database manager for Campus Sentinel.
Supports concurrent WAL reads, normalized vector BLOBs, robust foreign keys,
and accurate local-time analytics.
"""

import sqlite3
from datetime import datetime, timedelta
import numpy as np

from config import settings

EMBEDDING_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Connection & Schema Management
# ---------------------------------------------------------------------------

def get_connection():
    """Open a SQLite connection with WAL mode and foreign key enforcement."""
    settings.ensure_directories()
    conn = sqlite3.connect(settings.DB_PATH, timeout=20.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Create all tables and indexes. Idempotent and safe to run at every startup."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS stakeholders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                stakeholder_uid TEXT UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                role            TEXT NOT NULL,          -- Student / Faculty / Staff / Authorized
                embedding       BLOB NOT NULL,          -- float32 bytes, L2-normalized
                image_path      TEXT,
                registered_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS visit_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                stakeholder_id  INTEGER NOT NULL REFERENCES stakeholders(id) ON DELETE CASCADE,
                camera_location TEXT NOT NULL,
                similarity      REAL NOT NULL,
                timestamp       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unknown_persons (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path      TEXT,
                embedding       BLOB NOT NULL,
                camera_location TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                verified        INTEGER NOT NULL DEFAULT 0   -- 0 = pending review, 1 = verified
            );

            -- Indexes for fast lookups, joins, and timeline queries
            CREATE INDEX IF NOT EXISTS idx_visits_time ON visit_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_visits_stakeholder ON visit_logs(stakeholder_id);
            CREATE INDEX IF NOT EXISTS idx_unknown_time ON unknown_persons(timestamp);
            CREATE INDEX IF NOT EXISTS idx_unknown_verified ON unknown_persons(verified, timestamp);
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Embedding (De)serialization
# ---------------------------------------------------------------------------

def embedding_to_blob(embedding):
    """Convert a numpy array to bytes for BLOB storage."""
    return np.asarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def blob_to_embedding(blob):
    """Convert raw BLOB bytes back to a 1-D float32 numpy array."""
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


# ---------------------------------------------------------------------------
# Stakeholder Management
# ---------------------------------------------------------------------------

def add_stakeholder(stakeholder_uid, name, role, embedding, image_path=None):
    """Insert or update (by UID) a stakeholder. Returns the stakeholder row id."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stakeholders (stakeholder_uid, name, role, embedding, image_path, registered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(stakeholder_uid) DO UPDATE SET
                name=excluded.name,
                role=excluded.role,
                embedding=excluded.embedding,
                image_path=coalesce(excluded.image_path, stakeholders.image_path)
            """,
            (stakeholder_uid, name, role, embedding_to_blob(embedding), image_path, now),
        )
        conn.commit()
        cur.execute("SELECT id FROM stakeholders WHERE stakeholder_uid=?", (stakeholder_uid,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def load_stakeholder_gallery():
    """Load the full recognition gallery from database into memory."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, role, embedding FROM stakeholders ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return [], [], [], None

    ids, names, roles, embs = [], [], [], []
    for rid, name, role, blob in rows:
        ids.append(rid)
        names.append(name)
        roles.append(role)
        embs.append(blob_to_embedding(blob))
    return ids, names, roles, np.vstack(embs)


def list_stakeholders():
    """Return all stakeholder rows for display."""
    conn = get_connection()
    try:
        return conn.execute(
            """SELECT id, stakeholder_uid, name, role, image_path, registered_at
               FROM stakeholders ORDER BY registered_at DESC"""
        ).fetchall()
    finally:
        conn.close()


def get_stakeholder_by_uid(stakeholder_uid):
    """Fetch single stakeholder details."""
    conn = get_connection()
    try:
        return conn.execute(
            """SELECT id, stakeholder_uid, name, role, image_path, registered_at
               FROM stakeholders WHERE stakeholder_uid=?""",
            (stakeholder_uid,)
        ).fetchone()
    finally:
        conn.close()


def delete_stakeholder(stakeholder_uid):
    """Remove a stakeholder (and cascade-delete associated visit logs)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM stakeholders WHERE stakeholder_uid=?", (stakeholder_uid,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Visit Logging
# ---------------------------------------------------------------------------

def log_visit(stakeholder_id, camera_location, similarity):
    """Insert one visit record for a recognized stakeholder."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO visit_logs (stakeholder_id, camera_location, similarity, timestamp)
               VALUES (?, ?, ?, ?)""",
            (stakeholder_id, camera_location, float(similarity), now),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_visits(limit=500, offset=0, name_filter=None, role_filter=None):
    """Recent visits joined with stakeholder info (newest first)."""
    query = """
        SELECT v.id, v.timestamp, s.name, s.role, s.stakeholder_uid,
               v.camera_location, ROUND(v.similarity, 3), s.image_path
        FROM visit_logs v
        JOIN stakeholders s ON s.id = v.stakeholder_id
    """
    params = []
    conditions = []
    if name_filter and isinstance(name_filter, str) and name_filter.strip():
        conditions.append("(s.name LIKE ? OR s.stakeholder_uid LIKE ?)")
        params.extend([f"%{name_filter.strip()}%", f"%{name_filter.strip()}%"])
    if role_filter and isinstance(role_filter, str) and role_filter.strip():
        conditions.append("s.role = ?")
        params.append(role_filter.strip())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY v.timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = get_connection()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unknown Persons Management
# ---------------------------------------------------------------------------

def log_unknown(image_path, embedding, camera_location):
    """Insert an unknown-person record; returns the new row id."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO unknown_persons (image_path, embedding, camera_location, timestamp)
               VALUES (?, ?, ?, ?)""",
            (image_path, embedding_to_blob(embedding), camera_location, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def fetch_unknowns(limit=200, offset=0, only_unverified=False):
    """Recent unknown-person records (newest first)."""
    query = """SELECT id, timestamp, camera_location, image_path, verified
               FROM unknown_persons"""
    params = []
    if only_unverified:
        query += " WHERE verified = 0"
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = get_connection()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def mark_unknown_verified(unknown_id):
    """Flag an unknown-person record as verified/reviewed."""
    conn = get_connection()
    try:
        conn.execute("UPDATE unknown_persons SET verified=1 WHERE id=?", (unknown_id,))
        conn.commit()
    finally:
        conn.close()


def delete_unknown_person(unknown_id):
    """Delete an unknown person record."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM unknown_persons WHERE id=?", (unknown_id,))
        conn.commit()
    finally:
        conn.close()


def get_recent_unknown_embeddings(seconds=120):
    """Fetch embeddings of unknowns logged in the last N seconds for warm-start deduplication."""
    cutoff = (datetime.now() - timedelta(seconds=seconds)).isoformat(timespec="seconds")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT embedding, timestamp FROM unknown_persons WHERE timestamp >= ?",
            (cutoff,)
        ).fetchall()
        results = []
        for blob, ts_str in rows:
            try:
                dt = datetime.fromisoformat(ts_str)
                results.append((blob_to_embedding(blob), dt.timestamp()))
            except Exception:
                pass
        return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Statistics & Analytics Reporting
# ---------------------------------------------------------------------------

def get_stats():
    """Return headline dashboard counts with local-time day boundary."""
    today_start = datetime.now().strftime("%Y-%m-%d 00:00:00")
    today_iso_prefix = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        cur = conn.cursor()
        stakeholders_count = cur.execute("SELECT COUNT(*) FROM stakeholders").fetchone()[0]
        visits_count = cur.execute("SELECT COUNT(*) FROM visit_logs").fetchone()[0]
        unknowns_count = cur.execute("SELECT COUNT(*) FROM unknown_persons").fetchone()[0]
        unverified_unknowns = cur.execute("SELECT COUNT(*) FROM unknown_persons WHERE verified = 0").fetchone()[0]
        
        visits_today = cur.execute(
            "SELECT COUNT(*) FROM visit_logs WHERE timestamp >= ?",
            (today_iso_prefix,)
        ).fetchone()[0]

        return {
            "stakeholders": stakeholders_count,
            "visits": visits_count,
            "unknowns": unknowns_count,
            "unverified_unknowns": unverified_unknowns,
            "visits_today": visits_today,
        }
    finally:
        conn.close()


def get_reports_data():
    """Return consolidated aggregated metrics for reporting charts."""
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Daily visits over last 30 days
        daily_rows = cur.execute(
            """
            SELECT substr(timestamp, 1, 10) as visit_date, COUNT(*) as count
            FROM visit_logs
            GROUP BY visit_date
            ORDER BY visit_date DESC
            LIMIT 30
            """
        ).fetchall()
        daily_visits = [{"date": r[0], "count": r[1]} for r in reversed(daily_rows)]

        # Visits per camera location
        cam_rows = cur.execute(
            """
            SELECT camera_location, COUNT(*) as count
            FROM visit_logs
            GROUP BY camera_location
            ORDER BY count DESC
            """
        ).fetchall()
        cam_visits = [{"location": r[0], "count": r[1]} for r in cam_rows]

        # Visits per role
        role_rows = cur.execute(
            """
            SELECT s.role, COUNT(*) as count
            FROM visit_logs v
            JOIN stakeholders s ON s.id = v.stakeholder_id
            GROUP BY s.role
            ORDER BY count DESC
            """
        ).fetchall()
        role_visits = [{"role": r[0], "count": r[1]} for r in role_rows]

        # Unknowns timeline
        unknown_daily = cur.execute(
            """
            SELECT substr(timestamp, 1, 10) as unk_date, COUNT(*) as count
            FROM unknown_persons
            GROUP BY unk_date
            ORDER BY unk_date DESC
            LIMIT 30
            """
        ).fetchall()
        daily_unknowns = [{"date": r[0], "count": r[1]} for r in reversed(unknown_daily)]

        return {
            "daily_visits": daily_visits,
            "cam_visits": cam_visits,
            "role_visits": role_visits,
            "daily_unknowns": daily_unknowns,
        }
    finally:
        conn.close()
