"""
db.py — طبقة قاعدة البيانات.

- لو وُجد متغير البيئة DATABASE_URL → يستخدم Postgres (دائم، مناسب للاستضافة
  المجانية التي تمسح الملفات المحلية مثل Render).
- وإلا → SQLite محلي (للتجربة على جهازك فقط).
"""

import os
import datetime
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_PG = bool(DATABASE_URL)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive.db")

if USE_PG:
    import psycopg2
    from psycopg2.extras import execute_values

    PH = "%s"

    def _connect():
        return psycopg2.connect(DATABASE_URL, connect_timeout=15)

    _DDL = [
        """CREATE TABLE IF NOT EXISTS enf_archive (
               id BIGSERIAL PRIMARY KEY,
               captured_at TEXT NOT NULL,
               freq_hz DOUBLE PRECISION NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_captured_at ON enf_archive (captured_at)",
        """CREATE TABLE IF NOT EXISTS verification_log (
               id BIGSERIAL PRIMARY KEY,
               checked_at TEXT NOT NULL,
               file_name TEXT,
               result TEXT NOT NULL,
               confidence DOUBLE PRECISION,
               details TEXT)""",
    ]
else:
    PH = "?"

    def _connect():
        return sqlite3.connect(DB_PATH)

    _DDL = [
        """CREATE TABLE IF NOT EXISTS enf_archive (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               captured_at TEXT NOT NULL,
               freq_hz REAL NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_captured_at ON enf_archive (captured_at)",
        """CREATE TABLE IF NOT EXISTS verification_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               checked_at TEXT NOT NULL,
               file_name TEXT,
               result TEXT NOT NULL,
               confidence REAL,
               details TEXT)""",
    ]

_initialized = False


def _conn():
    global _initialized
    conn = _connect()
    if not _initialized:
        cur = conn.cursor()
        for stmt in _DDL:
            cur.execute(stmt)
        conn.commit()
        _initialized = True
    return conn


def _fetchall(sql, params=()):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        conn.close()


def _scalar(sql, params=()):
    rows = _fetchall(sql, params)
    return rows[0][0] if rows else 0


def backend_name():
    return "postgres" if USE_PG else "sqlite"


def save_reference_points(start_time: datetime.datetime, rel_times, freqs):
    rows = [
        ((start_time + datetime.timedelta(seconds=float(rt))).isoformat(), float(f))
        for rt, f in zip(rel_times, freqs)
    ]
    conn = _conn()
    try:
        cur = conn.cursor()
        if USE_PG:
            execute_values(cur, "INSERT INTO enf_archive (captured_at, freq_hz) VALUES %s", rows, page_size=2000)
        else:
            cur.executemany("INSERT INTO enf_archive (captured_at, freq_hz) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def load_reference_archive():
    data = _fetchall("SELECT captured_at, freq_hz FROM enf_archive ORDER BY captured_at")
    times = [datetime.datetime.fromisoformat(r[0]) for r in data]
    freqs = [r[1] for r in data]
    return times, freqs


def archive_count():
    return _scalar("SELECT COUNT(*) FROM enf_archive")


def archive_time_range():
    rows = _fetchall("SELECT MIN(captured_at), MAX(captured_at) FROM enf_archive")
    if not rows or rows[0][0] is None:
        return None, None
    return datetime.datetime.fromisoformat(rows[0][0]), datetime.datetime.fromisoformat(rows[0][1])


def log_verification(file_name: str, result: str, confidence: float, details: str):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO verification_log (checked_at, file_name, result, confidence, details) "
            f"VALUES ({PH}, {PH}, {PH}, {PH}, {PH})",
            (datetime.datetime.now().isoformat(), file_name, result, confidence, details),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_verifications(limit: int = 6):
    return _fetchall(
        "SELECT checked_at, file_name, result, confidence, details FROM verification_log "
        f"ORDER BY checked_at DESC LIMIT {PH}",
        (limit,),
    )


def verification_count():
    return _scalar("SELECT COUNT(*) FROM verification_log")


def storage_size_kb():
    """حجم التخزين التقريبي بالـKB."""
    if USE_PG:
        try:
            return round(_scalar("SELECT pg_database_size(current_database())") / 1024, 1)
        except Exception:
            return 0
    return round(os.path.getsize(DB_PATH) / 1024, 1) if os.path.exists(DB_PATH) else 0
