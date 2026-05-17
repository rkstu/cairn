"""Local SQLite store for offline triage cases — syncs to CouchDB when online."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class LocalStore:
    """SQLite-backed offline store for triage encounters.

    Each case is stored as a JSON document with a sync status flag.
    When connectivity returns, unsynced cases are pushed to a remote CouchDB.
    """

    def __init__(self, db_path: str | Path = "./data/cairn.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS triage_cases (
                case_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                patient_description TEXT,
                triage_level TEXT,
                confidence REAL,
                assessment TEXT,
                vitals TEXT,
                medications TEXT,
                safety_flags TEXT,
                fhir_resource TEXT,
                flagged_for_human INTEGER DEFAULT 0,
                synced INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_synced ON triage_cases(synced);
            CREATE INDEX IF NOT EXISTS idx_flagged ON triage_cases(flagged_for_human);
        """)
        self._conn.commit()

    def save_case(self, case_data: dict[str, Any]) -> str:
        """Save a triage case to the local store."""
        case_id = case_data.get("case_id", "")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO triage_cases
            (case_id, timestamp, patient_description, triage_level, confidence,
             assessment, vitals, medications, safety_flags, fhir_resource,
             flagged_for_human, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                case_id,
                case_data.get("timestamp", ""),
                case_data.get("patient_description", ""),
                case_data.get("triage_level", "YELLOW"),
                case_data.get("confidence", 0.0),
                case_data.get("assessment", ""),
                json.dumps(case_data.get("vitals", {})),
                json.dumps(case_data.get("medications", [])),
                json.dumps(case_data.get("safety_flags", [])),
                json.dumps(case_data.get("fhir_resource", {})),
                1 if case_data.get("flagged_for_human") else 0,
            ),
        )
        self._conn.commit()
        return case_id

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Retrieve a single case by ID."""
        row = self._conn.execute(
            "SELECT * FROM triage_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_unsynced_cases(self) -> list[dict[str, Any]]:
        """Get all cases that haven't been synced to remote."""
        rows = self._conn.execute(
            "SELECT * FROM triage_cases WHERE synced = 0 ORDER BY timestamp"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_flagged_cases(self) -> list[dict[str, Any]]:
        """Get all cases flagged for human review."""
        rows = self._conn.execute(
            "SELECT * FROM triage_cases WHERE flagged_for_human = 1 ORDER BY timestamp"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_synced(self, case_id: str):
        """Mark a case as synced to remote."""
        self._conn.execute(
            "UPDATE triage_cases SET synced = 1 WHERE case_id = ?", (case_id,)
        )
        self._conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Get summary stats for the local store."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN triage_level = 'RED' THEN 1 ELSE 0 END) as red,
                SUM(CASE WHEN triage_level = 'YELLOW' THEN 1 ELSE 0 END) as yellow,
                SUM(CASE WHEN triage_level = 'GREEN' THEN 1 ELSE 0 END) as green,
                SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as unsynced,
                SUM(CASE WHEN flagged_for_human = 1 THEN 1 ELSE 0 END) as flagged
            FROM triage_cases
        """).fetchone()
        return dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("vitals", "medications", "safety_flags", "fhir_resource"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        self._conn.close()
