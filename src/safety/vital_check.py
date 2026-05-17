"""Deterministic vital sign safety checks — no LLM involved."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import VitalBounds


class Severity(str, Enum):
    EMERGENCY = "EMERGENCY"
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    NORMAL = "NORMAL"


@dataclass
class SafetyFlag:
    flag_type: str
    severity: Severity
    detail: str
    action: str = "INFORM"  # INFORM | WARN | BLOCK | ESCALATE


def check_vitals(
    vitals: dict,
    bounds: VitalBounds | None = None,
) -> list[SafetyFlag]:
    """Run deterministic safety checks on patient vital signs.

    Returns a list of SafetyFlags. Empty list = all vitals within normal bounds.
    """
    bounds = bounds or VitalBounds()
    flags: list[SafetyFlag] = []

    spo2: Optional[int] = vitals.get("spo2")
    if spo2 is not None:
        if spo2 <= bounds.spo2_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_SPO2",
                severity=Severity.EMERGENCY,
                detail=f"SpO2 {spo2}% ≤ {bounds.spo2_critical}% — immediate intervention required",
                action="ESCALATE",
            ))
        elif spo2 < bounds.spo2_warning:
            flags.append(SafetyFlag(
                flag_type="VITAL_SPO2",
                severity=Severity.WARNING,
                detail=f"SpO2 {spo2}% below normal range",
                action="WARN",
            ))

    hr: Optional[int] = vitals.get("heart_rate")
    if hr is not None:
        if hr >= bounds.hr_high_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_HR",
                severity=Severity.EMERGENCY,
                detail=f"Heart rate {hr} bpm ≥ {bounds.hr_high_critical} — tachycardia",
                action="ESCALATE",
            ))
        elif hr <= bounds.hr_low_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_HR",
                severity=Severity.EMERGENCY,
                detail=f"Heart rate {hr} bpm ≤ {bounds.hr_low_critical} — bradycardia",
                action="ESCALATE",
            ))

    systolic: Optional[int] = vitals.get("systolic_bp")
    diastolic: Optional[int] = vitals.get("diastolic_bp")
    if systolic is not None and systolic >= bounds.systolic_bp_critical:
        flags.append(SafetyFlag(
            flag_type="VITAL_BP",
            severity=Severity.EMERGENCY,
            detail=f"Systolic BP {systolic} mmHg ≥ {bounds.systolic_bp_critical} — hypertensive crisis",
            action="ESCALATE",
        ))
    if diastolic is not None and diastolic >= bounds.diastolic_bp_critical:
        flags.append(SafetyFlag(
            flag_type="VITAL_BP",
            severity=Severity.EMERGENCY,
            detail=f"Diastolic BP {diastolic} mmHg ≥ {bounds.diastolic_bp_critical} — hypertensive crisis",
            action="ESCALATE",
        ))

    temp: Optional[float] = vitals.get("temperature")
    if temp is not None:
        if temp >= bounds.temp_high_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_TEMP",
                severity=Severity.CRITICAL,
                detail=f"Temperature {temp}°C ≥ {bounds.temp_high_critical}°C — hyperthermia",
                action="ESCALATE",
            ))
        elif temp <= bounds.temp_low_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_TEMP",
                severity=Severity.CRITICAL,
                detail=f"Temperature {temp}°C ≤ {bounds.temp_low_critical}°C — hypothermia",
                action="ESCALATE",
            ))

    glucose: Optional[int] = vitals.get("glucose")
    if glucose is not None:
        if glucose >= bounds.glucose_high_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_GLUCOSE",
                severity=Severity.EMERGENCY,
                detail=f"Glucose {glucose} mg/dL ≥ {bounds.glucose_high_critical} — severe hyperglycemia",
                action="ESCALATE",
            ))
        elif glucose <= bounds.glucose_low_critical:
            flags.append(SafetyFlag(
                flag_type="VITAL_GLUCOSE",
                severity=Severity.EMERGENCY,
                detail=f"Glucose {glucose} mg/dL ≤ {bounds.glucose_low_critical} — severe hypoglycemia",
                action="ESCALATE",
            ))

    return flags
