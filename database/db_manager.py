"""
database/db_manager.py

"""

import sqlite3
from datetime import datetime

import numpy as np

from config import settings

EMBEDDING_DTYPE = np.float32


# Connection & schema
# ---------------------------------------------------------------------------
def get_connection():
    """Open a SQLite connection with sensible defaults."""
    settings.ensure_directories()
    conn = sqlite3.connect(settings.DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Create all tables. Idempotent — safe to call at every startup."""
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
                verified        INTEGER NOT NULL DEFAULT 0   -- 0 = pending review
            );

            CREATE INDEX IF NOT EXISTS idx_visits_time  ON visit_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_unknown_time ON unknown_persons(timestamp);
            """
        )
        conn.commit()
    finally:
        conn.close()


# Embedding (de)serialization
# ---------------------------------------------------------------------------
def embedding_to_blob(embedding):
    """np.ndarray -> bytes for BLOB storage."""
    return np.asarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def blob_to_embedding(blob):
    """bytes -> 1-D float32 np.ndarray."""
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


# Stakeholders
# ---------------------------------------------------------------------------
def add_stakeholder(stakeholder_uid, name, role, embedding, image_path=None):
    """
    Insert or update (by UID) a stakeholder. Returns the stakeholder row id.
    """
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stakeholders (stakeholder_uid, name, role, embedding,
                                      image_path, registered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(stakeholder_uid) DO UPDATE SET
                name=excluded.name, role=excluded.role,
                embedding=excluded.embedding, image_path=excluded.image_path
            """,
            (stakeholder_uid, name, role, embedding_to_blob(embedding),
             image_path, now),
        )
        conn.commit()
        cur.execute("SELECT id FROM stakeholders WHERE stakeholder_uid=?",
                    (stakeholder_uid,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def load_stakeholder_gallery():
    """
    Load the full recognition gallery.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, role, embedding FROM stakeholders"
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
    """Return all stakeholder rows (without embeddings) for the dashboard."""
    conn = get_connection()
    try:
        return conn.execute(
            """SELECT id, stakeholder_uid, name, role, image_path, registered_at
               FROM stakeholders ORDER BY name"""
        ).fetchall()
    finally:
        conn.close()


def delete_stakeholder(stakeholder_uid):
    """Remove a stakeholder (and cascade-delete their visit logs)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM stakeholders WHERE stakeholder_uid=?",
                     (stakeholder_uid,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Visit logs
# ---------------------------------------------------------------------------
def log_visit(stakeholder_id, camera_location, similarity):
    """Insert one visit record for a recognized stakeholder."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO visit_logs (stakeholder_id, camera_location,
                                    similarity, timestamp)
            VALUES (?, ?, ?, ?)""",
            (stakeholder_id, camera_location, float(similarity), now),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_visits(limit=500, name_filter=None):
    """Recent visits joined with stakeholder info (newest first)."""
    query = """
        SELECT v.timestamp, s.name, s.role, s.stakeholder_uid,
            v.camera_location, ROUND(v.similarity, 3)
        FROM visit_logs v JOIN stakeholders s ON s.id = v.stakeholder_id
    """
    params = []
    if name_filter:
        query += " WHERE s.name LIKE ? "
        params.append(f"%{name_filter}%")
    query += " ORDER BY v.timestamp DESC LIMIT ?"
    params.append(limit)

    conn = get_connection()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unknown persons
# ---------------------------------------------------------------------------
def log_unknown(image_path, embedding, camera_location):
    """Insert an unknown-person record; returns the new row id."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO unknown_persons (image_path, embedding,
                                            camera_location, timestamp)
               VALUES (?, ?, ?, ?)""",
            (image_path, embedding_to_blob(embedding), camera_location, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def fetch_unknowns(limit=200, only_unverified=False):
    """Recent unknown-person records (newest first)."""
    query = """SELECT id, timestamp, camera_location, image_path, verified
            FROM unknown_persons"""
    if only_unverified:
        query += " WHERE verified = 0"
    query += " ORDER BY timestamp DESC LIMIT ?"
    conn = get_connection()
    try:
        return conn.execute(query, (limit,)).fetchall()
    finally:
        conn.close()


def mark_unknown_verified(unknown_id):
    """Flag an unknown-person record as reviewed by security staff."""
    conn = get_connection()
    try:
        conn.execute("UPDATE unknown_persons SET verified=1 WHERE id=?",
                     (unknown_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Simple statistics for the dashboard
# ---------------------------------------------------------------------------
def get_stats():
    """Return headline counts as a dict."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        stats = {
            "stakeholders": cur.execute(
                "SELECT COUNT(*) FROM stakeholders").fetchone()[0],
            "visits": cur.execute(
                "SELECT COUNT(*) FROM visit_logs").fetchone()[0],
            "unknowns": cur.execute(
                "SELECT COUNT(*) FROM unknown_persons").fetchone()[0],
            "visits_today": cur.execute(
                "SELECT COUNT(*) FROM visit_logs WHERE timestamp >= date('now')"
            ).fetchone()[0],
        }
        return stats
    finally:
        conn.close()
