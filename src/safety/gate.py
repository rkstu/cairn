"""Safety gate — deterministic checks that run BEFORE any LLM output reaches the user."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import CairnConfig
from src.safety.drug_check import DrugSafetyChecker
from src.safety.vital_check import SafetyFlag, Severity, check_vitals


@dataclass
class GateResult:
    """Result of running the safety gate on a triage recommendation."""

    passed: bool
    flags: list[SafetyFlag] = field(default_factory=list)
    blocked: bool = False
    emergency: bool = False

    @property
    def summary(self) -> str:
        if not self.flags:
            return "All safety checks passed."
        parts = []
        for f in self.flags:
            parts.append(f"[{f.severity.value}] {f.detail}")
        return "\n".join(parts)


class SafetyGate:
    """Deterministic safety gate — no LLM involved.

    Checks vital bounds, drug interactions, and adverse events
    BEFORE the triage recommendation is shown to the user.
    """

    def __init__(self, config: CairnConfig | None = None):
        self.config = config or CairnConfig()
        db_path = Path(self.config.onsides_db_path)
        self.drug_checker = DrugSafetyChecker(db_path) if db_path.exists() else None

    def check(
        self,
        vitals: dict | None = None,
        medications: list[str] | None = None,
    ) -> GateResult:
        """Run all deterministic safety checks."""
        flags: list[SafetyFlag] = []

        # 1. Vital sign bounds
        if vitals:
            flags.extend(check_vitals(vitals, self.config.vitals))

        # 2. Drug safety (if OnSIDES DB is available)
        if medications and self.drug_checker:
            flags.extend(self.drug_checker.check_all(medications))

        blocked = any(f.action == "BLOCK" for f in flags)
        emergency = any(f.severity == Severity.EMERGENCY for f in flags)

        return GateResult(
            passed=not blocked,
            flags=flags,
            blocked=blocked,
            emergency=emergency,
        )
