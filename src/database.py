"""SQLite logging for bulk-uploaded retraining data, plus the auto-retrain
trigger condition (in addition to the UI's manual trigger button)."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = "data/uploads.db"
AUTO_RETRAIN_THRESHOLD = 20  # pending uploads before an auto-retrain is recommended


def get_connection(db_path=DEFAULT_DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            used_in_retraining INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def insert_upload(filename, filepath, label, db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO uploaded_data (filename, filepath, label, timestamp, used_in_retraining) "
        "VALUES (?, ?, ?, ?, 0)",
        (filename, filepath, label, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_pending_samples(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM uploaded_data WHERE used_in_retraining = 0"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_as_retrained(ids, db_path=DEFAULT_DB_PATH):
    if not ids:
        return
    conn = get_connection(db_path)
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE uploaded_data SET used_in_retraining = 1 WHERE id IN ({placeholders})", ids
    )
    conn.commit()
    conn.close()


def mark_all_pending_as_retrained(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    conn.execute("UPDATE uploaded_data SET used_in_retraining = 1 WHERE used_in_retraining = 0")
    conn.commit()
    conn.close()


def get_upload_stats(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) AS c FROM uploaded_data").fetchone()["c"]
    pending = conn.execute(
        "SELECT COUNT(*) AS c FROM uploaded_data WHERE used_in_retraining = 0"
    ).fetchone()["c"]
    by_label = conn.execute(
        "SELECT label, COUNT(*) AS c FROM uploaded_data GROUP BY label"
    ).fetchall()
    conn.close()
    return {
        "total_uploads": total,
        "pending_uploads": pending,
        "by_label": {row["label"]: row["c"] for row in by_label},
        "auto_retrain_recommended": pending >= AUTO_RETRAIN_THRESHOLD,
    }
