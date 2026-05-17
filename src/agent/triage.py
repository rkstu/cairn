"""Cairn triage agent — LangGraph workflow orchestrating the full pipeline."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agent.tools import TRIAGE_TOOLS
from src.config import CairnConfig
from src.models.cascade import ModelCascade
from src.safety.gate import SafetyGate


@dataclass
class TriageCase:
    """A single triage encounter."""

    case_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    patient_description: str = ""
    image_path: str | None = None
    vitals: dict = field(default_factory=dict)
    medications: list[str] = field(default_factory=list)
    # Filled by pipeline
    cascade_result: dict = field(default_factory=dict)
    safety_result: dict = field(default_factory=dict)
    final_output: str = ""
    triage_level: str = "YELLOW"
    confidence: float = 0.0
    flagged_for_human: bool = False


class TriageAgent:
    """Full triage pipeline: cascade → safety gate → output.

    Usage:
        agent = TriageAgent()
        result = agent.run(
            patient_description="45yo male, chest pain radiating to left arm...",
            vitals={"spo2": 94, "heart_rate": 110, "systolic_bp": 160},
            medications=["aspirin", "warfarin"],
            image_path="wound.jpg",
        )
    """

    def __init__(self, config: CairnConfig | None = None):
        self.config = config or CairnConfig()
        self.cascade = ModelCascade(self.config)
        self.gate = SafetyGate(self.config)

    def run(
        self,
        patient_description: str,
        vitals: dict | None = None,
        medications: list[str] | None = None,
        image_path: str | None = None,
    ) -> TriageCase:
        """Execute the full triage pipeline for a patient encounter."""
        case = TriageCase(
            patient_description=patient_description,
            image_path=image_path,
            vitals=vitals or {},
            medications=medications or [],
        )

        # Step 1: Model cascade (screen → reason if needed)
        case.cascade_result = self.cascade.triage(
            patient_description=patient_description,
            image_path=image_path,
            vitals=vitals,
            tools=TRIAGE_TOOLS,
        )

        case.triage_level = case.cascade_result.get("triage_level", "YELLOW")
        case.confidence = case.cascade_result.get("confidence", 0.0)
        case.flagged_for_human = case.cascade_result.get("flagged_for_human", False)

        # Step 2: Deterministic safety gate
        gate_result = self.gate.check(
            vitals=vitals,
            medications=medications,
        )
        case.safety_result = {
            "passed": gate_result.passed,
            "blocked": gate_result.blocked,
            "emergency": gate_result.emergency,
            "flags": [
                {
                    "type": f.flag_type,
                    "severity": f.severity.value,
                    "detail": f.detail,
                    "action": f.action,
                }
                for f in gate_result.flags
            ],
        }

        # Step 3: Compose final output
        case.final_output = self._compose_output(case, gate_result)

        # Override triage level if safety gate found emergency
        if gate_result.emergency:
            case.triage_level = "RED"

        return case

    def _compose_output(self, case: TriageCase, gate_result: Any) -> str:
        """Compose the final output shown to the community health worker."""
        parts: list[str] = []

        # Safety alerts first (most important)
        if gate_result.emergency:
            parts.append("*** EMERGENCY ALERT ***")
            for flag in gate_result.flags:
                if flag.severity.value == "EMERGENCY":
                    parts.append(f"  ! {flag.detail}")
            parts.append("")

        if gate_result.blocked:
            parts.append("*** SAFETY BLOCK — Recommendation withheld ***")
            for flag in gate_result.flags:
                if flag.action == "BLOCK":
                    parts.append(f"  BLOCKED: {flag.detail}")
            parts.append("This case has been flagged for physician review.")
            parts.append("")

        # Triage assessment (only if not blocked)
        if not gate_result.blocked:
            parts.append(f"Triage Level: {case.triage_level}")
            parts.append(f"Confidence: {case.confidence:.0%}")
            if case.cascade_result.get("escalated"):
                reason = case.cascade_result.get("escalation_reason", "")
                parts.append(f"(Escalated to advanced model — reason: {reason})")
            parts.append("")
            parts.append(case.cascade_result.get("assessment", ""))

        # Warnings (non-blocking)
        warnings = [f for f in gate_result.flags if f.action == "WARN"]
        if warnings:
            parts.append("\n--- Safety Warnings ---")
            for w in warnings:
                parts.append(f"  Warning: {w.detail}")

        # Human review flag
        if case.flagged_for_human:
            parts.append("\n>>> LOW CONFIDENCE — Queued for physician review <<<")

        return "\n".join(parts)

    def to_fhir_encounter(self, case: TriageCase) -> dict:
        """Convert a triage case to a FHIR-compatible Encounter resource."""
        triage_code_map = {
            "RED": {"code": "1", "display": "Immediate"},
            "YELLOW": {"code": "2", "display": "Urgent"},
            "GREEN": {"code": "3", "display": "Non-urgent"},
        }
        triage_code = triage_code_map.get(
            case.triage_level, {"code": "2", "display": "Urgent"}
        )
        return {
            "resourceType": "Encounter",
            "id": case.case_id,
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "EMER",
                "display": "emergency",
            },
            "priority": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority",
                        "code": triage_code["code"],
                        "display": triage_code["display"],
                    }
                ]
            },
            "period": {"start": case.timestamp},
            "reasonCode": [{"text": case.patient_description[:200]}],
            "extension": [
                {
                    "url": "cairn:confidence",
                    "valueDecimal": round(case.confidence, 3),
                },
                {
                    "url": "cairn:model_used",
                    "valueString": case.cascade_result.get("model_used", ""),
                },
                {
                    "url": "cairn:assessment",
                    "valueString": case.cascade_result.get("assessment", "")[:1000],
                },
                {
                    "url": "cairn:safety_flags",
                    "valueString": json.dumps(case.safety_result.get("flags", [])),
                },
            ],
        }
