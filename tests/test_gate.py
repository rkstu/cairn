"""Tests for the integrated safety gate."""

from src.config import CairnConfig
from src.safety.gate import SafetyGate


class TestSafetyGate:
    def test_passes_with_normal_vitals(self):
        gate = SafetyGate(CairnConfig(onsides_db_path="/nonexistent"))
        result = gate.check(vitals={"spo2": 98, "heart_rate": 72})
        assert result.passed
        assert not result.blocked
        assert not result.emergency
        assert result.flags == []

    def test_emergency_on_critical_vitals(self):
        gate = SafetyGate(CairnConfig(onsides_db_path="/nonexistent"))
        result = gate.check(vitals={"spo2": 80})
        assert result.emergency
        assert len(result.flags) == 1

    def test_no_crash_without_onsides_db(self):
        gate = SafetyGate(CairnConfig(onsides_db_path="/nonexistent"))
        result = gate.check(medications=["aspirin", "warfarin"])
        assert result.passed  # No drug DB → no flags → passes

    def test_no_vitals_no_meds_passes(self):
        gate = SafetyGate(CairnConfig(onsides_db_path="/nonexistent"))
        result = gate.check()
        assert result.passed
        assert result.summary == "All safety checks passed."

    def test_multiple_emergency_flags(self):
        gate = SafetyGate(CairnConfig(onsides_db_path="/nonexistent"))
        result = gate.check(vitals={"spo2": 75, "heart_rate": 180, "glucose": 20})
        assert result.emergency
        assert len(result.flags) == 3
