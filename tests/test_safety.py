"""Tests for the deterministic safety gate — vitals and drug checks."""

import pytest

from src.config import VitalBounds
from src.safety.vital_check import Severity, check_vitals


class TestVitalChecks:
    def test_normal_vitals_no_flags(self):
        vitals = {"spo2": 98, "heart_rate": 72, "systolic_bp": 120, "glucose": 100}
        flags = check_vitals(vitals)
        assert flags == []

    def test_critical_spo2(self):
        flags = check_vitals({"spo2": 85})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY
        assert flags[0].action == "ESCALATE"
        assert "85%" in flags[0].detail

    def test_warning_spo2(self):
        flags = check_vitals({"spo2": 92})
        assert len(flags) == 1
        assert flags[0].severity == Severity.WARNING

    def test_normal_spo2_no_flag(self):
        flags = check_vitals({"spo2": 97})
        assert flags == []

    def test_tachycardia(self):
        flags = check_vitals({"heart_rate": 160})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY
        assert "tachycardia" in flags[0].detail.lower()

    def test_bradycardia(self):
        flags = check_vitals({"heart_rate": 35})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY
        assert "bradycardia" in flags[0].detail.lower()

    def test_hypertensive_crisis_systolic(self):
        flags = check_vitals({"systolic_bp": 200})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY

    def test_hypertensive_crisis_diastolic(self):
        flags = check_vitals({"diastolic_bp": 130})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY

    def test_hyperthermia(self):
        flags = check_vitals({"temperature": 41.2})
        assert len(flags) == 1
        assert flags[0].severity == Severity.CRITICAL

    def test_hypothermia(self):
        flags = check_vitals({"temperature": 34.0})
        assert len(flags) == 1
        assert flags[0].severity == Severity.CRITICAL

    def test_severe_hyperglycemia(self):
        flags = check_vitals({"glucose": 450})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY

    def test_severe_hypoglycemia(self):
        flags = check_vitals({"glucose": 40})
        assert len(flags) == 1
        assert flags[0].severity == Severity.EMERGENCY

    def test_multiple_critical_vitals(self):
        vitals = {"spo2": 80, "heart_rate": 170, "systolic_bp": 200}
        flags = check_vitals(vitals)
        assert len(flags) == 3
        assert all(f.severity == Severity.EMERGENCY for f in flags)

    def test_empty_vitals_no_flags(self):
        assert check_vitals({}) == []

    def test_missing_keys_ignored(self):
        assert check_vitals({"unknown_field": 42}) == []

    def test_custom_bounds(self):
        custom = VitalBounds(spo2_critical=85, spo2_warning=90)
        flags = check_vitals({"spo2": 87}, bounds=custom)
        assert len(flags) == 1
        assert flags[0].severity == Severity.WARNING  # Between 85-90 = warning
