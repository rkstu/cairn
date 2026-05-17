"""Tests for the local SQLite triage store."""

import os
import tempfile

import pytest

from src.sync.store import LocalStore


@pytest.fixture
def store():
    """Create a temporary store for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        s = LocalStore(db_path)
        yield s
        s.close()


class TestLocalStore:
    def test_save_and_retrieve(self, store):
        case = {
            "case_id": "test-001",
            "timestamp": "2026-04-12T10:00:00Z",
            "patient_description": "45yo male, chest pain",
            "triage_level": "RED",
            "confidence": 0.85,
            "assessment": "Possible MI",
            "vitals": {"spo2": 94, "heart_rate": 110},
            "medications": ["aspirin"],
            "safety_flags": [],
            "flagged_for_human": False,
        }
        store.save_case(case)
        result = store.get_case("test-001")
        assert result is not None
        assert result["triage_level"] == "RED"
        assert result["confidence"] == 0.85
        assert result["vitals"] == {"spo2": 94, "heart_rate": 110}

    def test_unsynced_cases(self, store):
        for i in range(3):
            store.save_case({
                "case_id": f"case-{i}",
                "timestamp": f"2026-04-12T10:0{i}:00Z",
                "triage_level": "YELLOW",
            })
        unsynced = store.get_unsynced_cases()
        assert len(unsynced) == 3

        store.mark_synced("case-1")
        unsynced = store.get_unsynced_cases()
        assert len(unsynced) == 2

    def test_flagged_cases(self, store):
        store.save_case({"case_id": "a", "timestamp": "", "flagged_for_human": True})
        store.save_case({"case_id": "b", "timestamp": "", "flagged_for_human": False})
        flagged = store.get_flagged_cases()
        assert len(flagged) == 1
        assert flagged[0]["case_id"] == "a"

    def test_stats(self, store):
        store.save_case({"case_id": "r", "timestamp": "", "triage_level": "RED"})
        store.save_case({"case_id": "y", "timestamp": "", "triage_level": "YELLOW"})
        store.save_case({"case_id": "g", "timestamp": "", "triage_level": "GREEN"})
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["red"] == 1
        assert stats["yellow"] == 1
        assert stats["green"] == 1
        assert stats["unsynced"] == 3

    def test_upsert(self, store):
        store.save_case({"case_id": "x", "timestamp": "", "triage_level": "YELLOW"})
        store.save_case({"case_id": "x", "timestamp": "", "triage_level": "RED"})
        result = store.get_case("x")
        assert result["triage_level"] == "RED"  # Updated

    def test_nonexistent_case(self, store):
        assert store.get_case("does-not-exist") is None
